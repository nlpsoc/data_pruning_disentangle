# coding=utf-8
# This file has been modified by Yupei Du from Utrecht University. 
# The original file is licensed under the Apache License Version 2.0. 
# The modifications by YD are licensed under the MIT license.
# 
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

os.environ['HF_HOME'] = 'data/huggingface'
os.environ['HF_DATASETS_CACHE'] = 'data/huggingface'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BITSANDBYTES_USE_BF16"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.distributed as dist
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    HfArgumentParser,
    BitsAndBytesConfig, 
    DataCollatorWithPadding, 
    DataCollatorForSeq2Seq, 
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from dataloader import DATALOADER_DICT, DataCollatorForMultipleChoice, NUM_LABELS_DICT
from trainers import (
    TrainingDynamicTrainer, 
    compute_classification_metrics, 
    DataTrainingArguments,
    ModelArguments
)

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    with training_args.main_process_first(desc="create target directory"):
        os.makedirs(training_args.output_dir, exist_ok=True)
        
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(os.path.join(training_args.output_dir, 'train.log'))
        ],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        + f"device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        + f"mixed precision training: {training_args.fp16 or training_args.bf16}, "
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading data
    logger.info('Loading data')
    dataloader = DATALOADER_DICT[data_args.task_name](
        max_length=data_args.max_seq_length, 
        train_dataset_name=data_args.train_dataset_name, 
        hf_token=model_args.token,
    )

    train_data, train_info, eval_data = None, None, None
    if training_args.do_train:
        with training_args.main_process_first(desc='Loading training datasets'):  
            if data_args.selected_uid_path is not None:
                selected_uids = []
                with open(data_args.selected_uid_path, 'r') as f_selected_uids:
                    for line in f_selected_uids:
                        selected_uids.append(line.strip())
            else:
                selected_uids = None

            train_data, train_info = dataloader.load_train(
                model_name=model_args.tokenizer_name if model_args.tokenizer_name 
                else model_args.model_name_or_path,
                uid=selected_uids, 
                noise_rate=data_args.noise_rate,  
            )

    if training_args.do_eval:
        with training_args.main_process_first(desc='Loading evaluation datasets'):
            eval_data_info = dataloader.load_dev(
                model_name=model_args.tokenizer_name if model_args.tokenizer_name
                else model_args.model_name_or_path
            )
            test_data_info = dataloader.load_test(
                model_name=model_args.tokenizer_name if model_args.tokenizer_name
                else model_args.model_name_or_path
            )
            eval_data = {
                f'val_{dataset_name}': eval_data_info[dataset_name][0] 
                for dataset_name in eval_data_info
            }
            eval_data.update({
                f'test_{dataset_name}': test_data_info[dataset_name][0] 
                for dataset_name in test_data_info
            })

    # Initialize model and trainer
    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name \
        else model_args.model_name_or_path
    tokenizer = dataloader.tokenizer_dict[tokenizer_name]
    torch_dtype = torch.float32
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    compute_metrics = None
    pad_to_multiple_of=8  # tensor cores
    
    # sequence classification tasks
    if data_args.task_name in ['cad', 'uni_nli']:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            token=model_args.token,
            num_labels=NUM_LABELS_DICT[data_args.task_name],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            token=model_args.token,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        compute_metrics = compute_classification_metrics


    # encoder multiple choice tasks
    elif data_args.task_name in ['winogrande']:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            token=model_args.token,
        )
        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            token=model_args.token,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        data_collator = DataCollatorForMultipleChoice(
            tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        compute_metrics = compute_classification_metrics

    # instruction following tasks
    # so far all causal llms we use support flash_attention_2
    elif data_args.task_name in ['tulu', 'sum']:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            token=model_args.token,
        )
        # QLoRA models, only support causal LM for now
        if model_args.lora and model_args.qlora:
            logger.info('Using QLoRA')
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool('.ckpt' in model_args.model_name_or_path),
                config=config,
                token=model_args.token,
                quantization_config=bnb_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                attn_implementation='flash_attention_2',
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool('.ckpt' in model_args.model_name_or_path),
                config=config,
                token=model_args.token,
                trust_remote_code=True, 
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                attn_implementation='flash_attention_2',
            )

        # resize embeddings for causal LM
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model, 
            label_pad_token_id=-100,
            pad_to_multiple_of=pad_to_multiple_of, 
            padding='longest',
        )
    
    # https://github.com/princeton-nlp/LESS/blob/b7ace0633fb8e4b606547fbeeaa9324084f05de1/less/train/train.py#L88
    if not isinstance(model, PeftModel) and model_args.lora:
        model.get_input_embeddings().weight.requires_grad = False
        if model.get_output_embeddings() is not None:
            model.get_output_embeddings().weight.requires_grad = False
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            modules_to_save=model_args.lora_modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA to model.")
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # trainer = Trainer(
    trainer = TrainingDynamicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=eval_data if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
        
    # Saving run training data uids
    if training_args.do_train and trainer.is_world_process_zero:
        uid = train_info['uid']
        if not isinstance(uid[0], str):
            uid = [str(i) for i in uid]
        with open(os.path.join(training_args.output_dir, 'train_uids.txt'), 'w') as f:
            for i in uid:
                f.write(f'{i}\n')

    # Print model parameters before training
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")
    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)

    # print the first training example
    example_input_ids = train_data[0]['input_ids']
    if isinstance(example_input_ids[0], list):
        example_input_ids = example_input_ids[0]
    logger.info(f"train_example: {tokenizer.decode(example_input_ids)}")
    
    # Training
    train_result = trainer.train()
    # trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_data))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == '__main__':
    main()
