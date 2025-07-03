import os
from typing import Optional, Dict, List
from collections import defaultdict
from packaging import version
from dataclasses import dataclass, field
import inspect

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import datasets
from transformers import Trainer
from transformers.utils import logging
from transformers.trainer_utils import speed_metrics
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

logger = logging.get_logger(__name__)


# Metric
def compute_classification_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(label_ids, preds),
        "f1": f1_score(label_ids, preds, average='macro'),
    }


# Forward
def forward_model(model, batch, output_hidden_states=False):
    model_signature = list(inspect.signature(model.forward).parameters.keys()) + ['labels', 'labels_ids']
    outputs = model(**{k: batch[k] for k in model_signature if k in batch}, 
                    output_hidden_states=output_hidden_states)
    return outputs


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    token: str = field(
        default='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',
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    # https://github.com/princeton-nlp/LESS/blob/b7ace0633fb8e4b606547fbeeaa9324084f05de1/less/train/model_arguments.py#L65
    lora: Optional[bool] = field(default=False, metadata={"help": "whether to use lora"})
    qlora: Optional[bool] = field(default=False, metadata={"help": "whether to use qlora"})
    lora_r: Optional[int] = field(default=8, metadata={"help": ("r for lora")})
    lora_alpha: Optional[float]=field(default=32, metadata={"help": ("alpha for lora")})
    lora_dropout: Optional[float]=field(default=0.1, metadata={"help": ("dropout for lora")})
    lora_target_modules: List[str]=field(default_factory=list, metadata={"help": ("target modules for lora")})
    lora_modules_to_save: List[str]=field(default_factory=list, metadata={"help": ("extra modules to save for lora")})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: str = field(
        # choices=["mnli", "cad", "hellaswag", "winogrande"], 
        default="mnli", 
        metadata={"help": "The name of the task to train on"}
    )
    train_dataset_name: str = field(
        # choices=["mnli", "cad", "hellaswag", "winogrande-xl", "winogrande-debiased"],
        default="mnli", 
        metadata={"help": "The name of the training dataset"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    selected_uid_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the selected uid file for data pruning"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_generation_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum generation sequence length."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    selected_uid_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the selected uid file for data pruning"}
    )   
    noise_rate: Optional[float] = field(
        default=None, metadata={"help": "The injected noise rate."}
    )


class TrainingDynamicTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_dynamics = defaultdict(list)
        self.gather_func = self.accelerator.gather_for_metrics
    
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        # Change: keep the data_idx column
        ignored_columns = list(set(dataset.column_names) - set(signature_columns) - set(["data_idx"]))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        data_idx = inputs.pop('data_idx') if 'data_idx' in inputs else None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # unwrapped_model = unwrap_model(model) 
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        
        # Change: save the probability of the correct token
        shift_masks = None
        if self.model.training:
            logits = outputs.logits.detach()

            is_classification = inputs['labels'] is not None and len(inputs['labels'].size()) == 1
            if is_classification:  # classification task
                prob = logits.softmax(dim=-1)[torch.arange(logits.size(0), device=logits.device), inputs['labels']]
                log_prob = logits.log_softmax(dim=-1)[torch.arange(logits.size(0), device=logits.device), inputs['labels']]
                if len(data_idx.size()) == 3:  # multiple choices data format
                    data_idx = data_idx[:, 0, 0]

            else:  # seq2seq task using causal language models, e.g., summarization, translation, and instruction generation
                shift_logits = logits[..., :-1, :]  # batch_size, seq_len, vocab_size
                shift_labels = inputs["input_ids"][..., 1:]  # batch_size, seq_len
                shift_masks = (inputs['labels'][..., 1:] != -100)
                # CHIA: prob, INV-PPL: log_prob.exp(), NLL: -log_prob
                prob = torch.gather(shift_logits.softmax(dim=-1), dim=-1, index=shift_labels[..., None])[..., 0]
                prob = (prob * (1.0 * shift_masks)).sum(dim=-1) / shift_masks.sum(dim=-1)
                log_prob = torch.gather(shift_logits.log_softmax(dim=-1), dim=-1, index=shift_labels[..., None])[..., 0]
                log_prob = (log_prob * (1.0 * shift_masks)).sum(dim=-1) / shift_masks.sum(dim=-1)
            
            # if device rank is 0
            if self.is_world_process_zero:
                step_training_dynamics = {
                    'data_idx': self.gather_func(data_idx).cpu(),
                    'prob': self.gather_func(prob).cpu(), 
                    'log_prob': self.gather_func(log_prob).cpu(),
                    'epoch': self.state.epoch, 
                    'global_step': self.state.global_step,
                }
                for k, v in step_training_dynamics.items():
                    self.training_dynamics[k].append(v)

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        if self.is_world_process_zero:
            if self.state.global_step > 1:
                if len(self.training_dynamics) > 0:
                    training_dynamics_dir = os.path.join(self.args.output_dir, 'training_dynamics')
                    os.makedirs(training_dynamics_dir, exist_ok=True)
                    torch.save(dict(self.training_dynamics), os.path.join(
                        training_dynamics_dir, f'step_{self.state.global_step}.pt'))
                    self.training_dynamics = defaultdict(list)
 

