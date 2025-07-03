# Part of code is adapted from the following sources:
# https://github.com/huggingface/transformers/blob/caa5c65db1f4db617cdac2ad667ba62edf94dd98/examples/pytorch/multiple-choice/run_swag_no_trainer.py

import os
from functools import partial
from itertools import chain
from dataclasses import dataclass
from typing import Optional, Union

os.environ['HF_HOME'] = 'data/huggingface'
os.environ['HF_DATASETS_CACHE'] = 'data/huggingface'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import LlamaTokenizer, GPTNeoXTokenizerFast, GPT2Tokenizer, CodeGenTokenizerFast
from transformers.utils import PaddingStrategy
from datasets import load_dataset, Dataset
import pandas as pd
from accelerate.logging import get_logger


logger = get_logger('dpm_revisited', log_level='INFO')


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


class Dataloader:

    def __init__(self, data_dir, max_length, max_target_length=None, hf_token=None): 
        self.data_dir = data_dir
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.tokenizer_dict = {}
        self.necessary_items = None
        self.dataset_info = None
        self.train_dataset_name = None
        self.hf_token = hf_token
        self.load_func_dict = {
            'json': self._load_json_dataset,
            'tsv': self._load_tsv_dataset,
            'csv': self._load_csv_dataset,
        }

    def _load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=self.hf_token)

        # no default pad token for llama!
        # here we add all special tokens again, because the default ones are not in the special_tokens_map
        if isinstance(tokenizer, LlamaTokenizer):
            num_added_tokens = tokenizer.add_special_tokens({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            })
            assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        elif isinstance(tokenizer, GPTNeoXTokenizerFast) or isinstance(tokenizer, CodeGenTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
        elif isinstance(tokenizer, GPT2Tokenizer) and 'opt' in model_name.lower():
            num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

        if 'qwen2' in model_name.lower():
            tokenizer.padding_side = 'left'

        self.tokenizer_dict[model_name] = tokenizer
        return tokenizer

    @staticmethod
    def _load_json_dataset(dataset_path):
        return load_dataset('json', data_files=dataset_path)['train']

    @staticmethod
    def _load_tsv_dataset(dataset_path):
        df = pd.read_csv(dataset_path, sep='\t')
        return Dataset.from_pandas(df.dropna(axis=0).reset_index(drop=True))
    
    @staticmethod
    def _load_csv_dataset(dataset_path):
        df = pd.read_csv(dataset_path)
        return Dataset.from_pandas(df.dropna(axis=0).reset_index(drop=True))

    def _tokenize_func(self):
        raise NotImplementedError
    
    def reindex_data(self, data):
        raise NotImplementedError
    
    def _process(self, dataset, dataset_name, model_name, uid=None):
        # rename the original name from the dataset to the standard name
        for item in [*self.necessary_items, 'labels', 'uid']:
            if item not in self.dataset_info[dataset_name]:
                continue
            if self.dataset_info[dataset_name][item] == item:
                continue
            dataset = dataset.rename_column(self.dataset_info[dataset_name][item], item)
        
        # select the rows with uid in the list
        if uid is not None:
            uid = set(uid)
            dataset = dataset.filter(lambda example: example['uid'] in uid)

        # add data indices 
        dataset = dataset.add_column('data_idx', list(range(len(dataset))))
        # convert string labels to class indices
        if 'labels' in dataset[0] and isinstance(dataset[0]['labels'], str):
            assert 'label_name_to_label' in self.dataset_info[dataset_name], \
                '`label_name_to_label` must be provided for string labels'
            dataset = dataset.map(lambda example: {
                'labels': self.dataset_info[dataset_name]['label_name_to_label'][example['labels']]})

        # remove additional columns
        add_info_columns = self.dataset_info[dataset_name]['additional_information'] \
            if 'additional_information' in self.dataset_info[dataset_name] else []
        extra_columns = [item for item in dataset.column_names if item not in
                         [*self.necessary_items, *add_info_columns, 'labels', 'data_idx', 'uid']]

        # process data
        # instruction tuning data cannot be tokenized in batch so we use multiprocessing
        use_batch = False if isinstance(self, Seq2SeqForCausalLMDataloader) else True
        num_proc = 18 if isinstance(self, Seq2SeqForCausalLMDataloader) else None
        dataset = dataset.map(partial(
            self._tokenize_func,
            tokenizer=self.tokenizer_dict[model_name],
            max_length=self.max_length),
            batched=use_batch, 
            num_proc=num_proc,
            remove_columns=extra_columns
        )

        return dataset

    def _load_dataset(self, dataset_name, split, model_name, uid=None, noise_rate=None):
        dataset_path, dataset_type = self.dataset_info[dataset_name]['dataset_path_type'][split]
        dataset_path = os.path.join(self.data_dir, dataset_path)
        if (
            (dataset_name.startswith('cad') 
             or dataset_name.startswith('winogrande') 
             or dataset_name.startswith('dialog_sum')) 
            and (noise_rate is not None)
        ):
            dataset = self.load_func_dict[dataset_type](dataset_path, noise_rate=noise_rate)
        else:
            dataset = self.load_func_dict[dataset_type](dataset_path)
        if model_name not in self.tokenizer_dict:
            _ = self._load_tokenizer(model_name)
        dataset = self._process(dataset, dataset_name, model_name, uid=uid)
        add_info_columns = self.dataset_info[dataset_name]['additional_information'] \
            if 'additional_information' in self.dataset_info[dataset_name] else []
        forward_columns = set(dataset.column_names) - set([*self.necessary_items, *add_info_columns, 'uid'])
        info_columns = set(dataset.column_names) - forward_columns
        return dataset.remove_columns(list(info_columns)), dataset.remove_columns(list(forward_columns))

    def load_train(self, model_name='microsoft/deberta-v3-small', uid=None, noise_rate=None):
        return self._load_dataset(
            self.train_dataset_name, 'train', model_name, uid=uid, noise_rate=noise_rate)

    def load_dev(self, dataset_names=None, model_name='microsoft/deberta-v3-small', uid=None):
        if dataset_names is None:
            dataset_names = [dataset_name for dataset_name in self.dataset_info
                             if 'dev' in self.dataset_info[dataset_name]['dataset_path_type']]
            dataset_names = [dataset_name for dataset_name in dataset_names if 'grad_only' not in dataset_name]
        return {dataset_name: self._load_dataset(dataset_name, 'dev', model_name, uid=uid) 
                for dataset_name in dataset_names}

    def load_test(self, dataset_names=None, model_name='microsoft/deberta-v3-small', uid=None):
        if dataset_names is None:
            dataset_names = [dataset_name for dataset_name in self.dataset_info
                             if 'test' in self.dataset_info[dataset_name]['dataset_path_type']]
            dataset_names = [dataset_name for dataset_name in dataset_names if 'grad_only' not in dataset_name]
        return {dataset_name: self._load_dataset(dataset_name, 'test', model_name, uid=uid) 
                for dataset_name in dataset_names}


class ClassificationDataloader(Dataloader):

    def __init__(self, data_dir, max_length, hf_token=None): 
        super().__init__(data_dir, max_length, hf_token=hf_token)

    def _tokenize_func(self, examples, tokenizer, max_length):
        text = [examples[text] for text in self.necessary_items]
        result = tokenizer(*text, padding=False, max_length=max_length, truncation=True)
        return result
    
    def _lm_tokenizer_func(self, examples, tokenizer, max_length):
        return self._tokenize_func(examples, tokenizer, max_length)
    
    def reindex_data(self, data):
        return data.map(lambda _, idx: {'data_idx': idx}, with_indices=True)


class MultipleChoiceDataloader(Dataloader):
    
    def __init__(self, data_dir, max_length, hf_token=None): 
        super().__init__(data_dir, max_length, hf_token=hf_token)
    
    def _tokenize_func(self, examples, tokenizer, max_length):
        num_choices = len(examples['choices'][0])
        questions = list(chain(*[[example_question] * num_choices for example_question in examples['question']]))
        choices = list(chain(*[example_choices for example_choices in examples['choices']]))
        labels, data_indices = examples['labels'], examples['data_idx']
        result = tokenizer(questions, choices, padding=False, max_length=max_length, truncation=True)
        result = {k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)] for k, v in result.items()}
        result['labels'] = labels
        result['data_idx'] = [[data_indices[i]] * num_choices for i in range(len(data_indices))]
        return result
    
    def reindex_data(self, data):
        num_choices = len(data[0]['data_idx'])
        return data.map(lambda _, idx: {'data_idx': [idx] * num_choices}, with_indices=True)
    

class Seq2SeqForCausalLMDataloader(Dataloader):

    def __init__(self, data_dir, max_length, hf_token=None):
        super().__init__(data_dir, max_length, hf_token=hf_token)

    def _tokenize_func(self, examples, tokenizer, max_length):
        '''
        Original implementation of the function: 
        1. https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238
        2. https://github.com/princeton-nlp/LESS/blob/b7ace0633fb8e4b606547fbeeaa9324084f05de1/less/data_selection/get_training_dataset.py#L96
        '''
        # if prompt doesn't end with space and completion doesn't start with space, add space
        if not examples['source'].endswith((' ', '\n', '\t')) and not examples['target'].startswith((' ', '\n', '\t')):
            example_text = examples['source'] + ' ' + examples['target']
        else:
            example_text = examples['source'] + examples['target']
        example_text = example_text + tokenizer.eos_token
        tokenized_example = tokenizer(
            example_text, return_tensors='pt', max_length=max_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()
        tokenized_prompt = tokenizer(
            examples['source'], return_tensors='pt', max_length=max_length, truncation=True)
        # mask the prompt part for avoiding loss
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }


class CADDataloader(ClassificationDataloader):
    # required fields: text, labels
    def __init__(self, data_dir='data/datasets/hsd', max_length=None, train_dataset_name='cad', hf_token=None):
        super().__init__(data_dir, max_length, hf_token=hf_token)
        self.train_dataset_name = train_dataset_name
        self.necessary_items = ['text']
        self.dataset_info = {
            'cad': {
                'dataset_path_type': {
                    'train': ('cad/cad_train_regenerated_uids.tsv', 'hsd_tsv'),
                    'dev': ('cad/cad_dev_regenerated_uids.tsv', 'hsd_tsv'),
                    'test': ('cad/cad_test_regenerated_uids.tsv', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
                'uid': 'uid', 
            },
            'dynahate_r2_original': {
                'dataset_path_type': {
                    'test': ('dynahate/r2_original.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r2_perturbation': {
                'dataset_path_type': {
                    'test': ('dynahate/r2_perturbation.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r3_original': {
                'dataset_path_type': {
                    'test': ('dynahate/r3_original.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r3_perturbation': {
                'dataset_path_type': {
                    'test': ('dynahate/r3_perturbation.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r4_original': {
                'dataset_path_type': {
                    'test': ('dynahate/r4_original.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
            'dynahate_r4_perturbation': {
                'dataset_path_type': {
                    'test': ('dynahate/r4_perturbation.test', 'hsd_tsv'),
                },
                'text': 'text',
                'labels': 'labels',
                'label_name_to_label': {'non-toxic': 0, 'toxic': 1},
                'label_names': ['non-toxic', 'toxic'],
            },
        }
        self.load_func_dict['hsd_tsv'] = self._load_hsd_tsv

    def _load_hsd_tsv(self, dataset_path, noise_rate=None):
        if noise_rate is not None and 'train' in dataset_path:
            dataset_path = os.path.join(self.data_dir, f'cad/flipped_labels/{noise_rate}/cad_train_flipped.tsv')
            print('Loading noise injected dataset:', dataset_path)
        data = pd.read_csv(dataset_path, sep='\t', names=['labels', 'text', 'uid'])
        return Dataset.from_pandas(data)


class WinoGrandeDataloader(MultipleChoiceDataloader):

    def __init__(self, data_dir='data/datasets/csr', max_length=None, train_dataset_name='winogrande-xl', hf_token=None):
        super().__init__(data_dir, max_length, hf_token=hf_token)
        self.train_dataset_name = train_dataset_name
        self.necessary_items = ['question', 'choices']
        self.dataset_info = {
            'winogrande-xl': {
                'dataset_path_type': {
                    'train': ('winogrande/xl-train.jsonl', 'winogrande_json'),
                }, 
                'question': 'sentence',
                'choices': 'choices', 
                'labels': 'labels',
                'uid': 'uid',
            }, 
            'winogrande-l': {
                'dataset_path_type': {
                    'train': ('winogrande/l-train.jsonl', 'winogrande_json'),
                }, 
                'question': 'sentence',
                'choices': 'choices', 
                'labels': 'labels',
                'uid': 'uid',
            }, 
            'winogrande-debiased': {
                'dataset_path_type': {
                    'train': ('winogrande/debiased-train.jsonl', 'winogrande_json'),
                }, 
                'question': 'sentence',
                'choices': 'choices', 
                'labels': 'labels',
                'uid': 'uid',
            }, 
            'winogrande': {
                'dataset_path_type': {
                    'dev': ('winogrande/validation.jsonl', 'winogrande_json'),
                    'test': ('winogrande/test.jsonl', 'winogrande_json'),
                }, 
                'question': 'sentence',
                'choices': 'choices', 
                'labels': 'labels',
                'uid': 'uid',
            }, 
        }
        self.load_func_dict['winogrande_json'] = self._load_winogrande_json_dataset
    
    def _load_winogrande_json_dataset(self, dataset_path, noise_rate=None):
        if noise_rate is not None and 'train' in dataset_path:
            dataset_name = (dataset_path.split(".")[0]).split('/')[-1]
            dataset_path = os.path.join(self.data_dir, f'winogrande/flipped_labels/{dataset_name}-{noise_rate}.jsonl')
            print('Loading noise injected dataset:', dataset_path)

        data = self._load_json_dataset(dataset_path)
        data = data.map(lambda example: {
            'choices': [example['option1'], example['option2']], 
            'labels': int(example['answer']) - 1, 
        })
        return data


class SumDataloader(Seq2SeqForCausalLMDataloader):

    def __init__(
            self,
            data_dir='data/datasets/sum/dialog_sum',
            max_length=512,
            train_dataset_name='dialog_sum', 
            hf_token=None, 
        ):
        super().__init__(data_dir, max_length, hf_token=hf_token)
        self.train_dataset_name = train_dataset_name
        self.necessary_items = ['source', 'target']
        self.dataset_info = {
            'dialog_sum': {'dataset_path_type': {
                'train': ('train.json', 'dialog_sum_json'),
                'dev': ('validation.json', 'dialog_sum_json'),
                'test': ('test.json', 'dialog_sum_json'),
            }},
        }
        self.load_func_dict['dialog_sum_json'] = self._load_dialog_sum_json_dataset
    
    def _tokenize_func(self, examples, tokenizer, max_length):
        source_text = examples['source'].strip() + '\ntl;dr\n'
        target_text = examples['target'].strip() + tokenizer.eos_token
        
        # Tokenize the source and target text together
        tokenized_example = tokenizer(
            source_text + target_text, 
            return_tensors='pt', 
            max_length=max_length, 
            truncation=True
        )
        
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()
        
        # Tokenize the source text alone
        tokenized_source = tokenizer(
            source_text, 
            return_tensors='pt', 
            max_length=max_length, 
            truncation=True
        )
        
        # Mask the prompt part for avoiding loss
        labels[:, :tokenized_source.input_ids.shape[1]] = -100
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }
    
    def _load_dialog_sum_json_dataset(self, dataset_path, noise_rate=None):
        if noise_rate is not None and 'train' in dataset_path:
            dataset_path = os.path.join(self.data_dir, f'flipped_labels/train_{noise_rate}.json')
            print('Loading noise injected dataset:', dataset_path)

        dataset = self._load_json_dataset(dataset_path)
        dataset = dataset.map(lambda example: {
            'source': example['dialogue'].strip(),
            'target': example['summary'].strip(), 
            'uid': example['id']}
        )
        return dataset


DATALOADER_DICT = {
    'cad': CADDataloader,
    'winogrande': WinoGrandeDataloader,
    'sum': SumDataloader,
}

NUM_LABELS_DICT = {
    'cad': 2,
    'winogrande': 2,
}
