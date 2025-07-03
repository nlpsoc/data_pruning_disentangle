# This script collects gradients and representations from a model during evaluation, 
# which is heavily based on: https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/get_info.py
import os
import sys
import json
import yaml
import logging
import argparse
import time
import hashlib
from typing import List, Optional

os.environ['HF_HOME'] = 'data/huggingface'
os.environ['HF_DATASETS_CACHE'] = 'data/huggingface'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BITSANDBYTES_USE_BF16"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.utils
from tqdm.auto import tqdm
import torch
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
import transformers
from transformers import (
    set_seed, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoModelForMultipleChoice,
    DataCollatorForSeq2Seq, 
    DataCollatorWithPadding
)
from transformers.optimization import AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from peft import PeftModel, AutoPeftModelForCausalLM
import datasets
import wandb

from trainers import forward_model
from dataloader import DATALOADER_DICT, DataCollatorForMultipleChoice 

logger = logging.getLogger(__name__)


def get_trak_projector(device: torch.device):
    ''' Get trak projectors (see https://github.com/MadryLab/trak for details) '''
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print('Using CudaProjector')
    except:
        projector = BasicProjector
        print('Using BasicProjector')
    return projector


def get_number_of_params(model):
    ''' Make sure that only lora parameters require gradients in peft models. '''
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and 'lora' not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f'Total number of parameters that require gradients: {num_params}')
    return num_params


def obtain_gradients_with_adam(vectorized_grads, adam_avg, adam_avg_sq):
    ''' obtain gradients with adam optimizer states. '''
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    updated_avg = beta1 * adam_avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * adam_avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def obtain_vectorized_grads(model):
    ''' obtain vectorized gradients '''
    vectorized_grads = torch.cat(
        [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_gradients(vectorized_grads, gradient_type='sgd', adam_avg=None, adam_avg_sq=None):
    if gradient_type == 'sgd':
        grads = vectorized_grads
    elif gradient_type == 'sign':
        grads = torch.sign(vectorized_grads)
    elif gradient_type == 'adam':
        assert adam_avg is not None and adam_avg_sq is not None
        grads = obtain_gradients_with_adam(vectorized_grads, adam_avg, adam_avg_sq)
    else:
        raise ValueError(f'Unsupported gradient type: {gradient_type}')
    
    if len(grads.size()) == 1:
        grads = grads[None, :]

    return grads


def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def load_pt_adam_state(model, optimizer_state_path):
    opt_grouped_parameters = [{'weight_decay': 0.0}, {'weight_decay': 0.0}]
    opt_grouped_parameter_names = [None, None]

    decay_parameters = [name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if 'bias' not in name]
    opt_grouped_parameters[0]['params'], opt_grouped_parameter_names[0] = zip(*[
        (p, n) for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad])
    param_name_to_size_dict = {n: p.size() for n, p in model.named_parameters() if p.requires_grad}
    if len(param_name_to_size_dict) != len(opt_grouped_parameter_names[0]):
        opt_grouped_parameters[1]['params'], opt_grouped_parameter_names[1] = zip(*[
            (p, n) for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad])
    else:
        opt_grouped_parameters[1]['params'], opt_grouped_parameter_names[1] = [], []

    optimizer = AdamW(opt_grouped_parameters)
    optimizer.load_state_dict(torch.load(optimizer_state_path, map_location='cpu'))
    saved_state_dict = optimizer.state_dict()

    param_name_to_saved_state_dict = {}
    for group_idx in range(len(saved_state_dict['param_groups'])):
        group_param_indices = saved_state_dict['param_groups'][group_idx]['params']
        group_param_names = opt_grouped_parameter_names[group_idx]
        for param_idx, param_name in zip(group_param_indices, group_param_names):
            param_size = param_name_to_size_dict[param_name]
            exp_avg = saved_state_dict['state'][param_idx]['exp_avg'].to(model.device)
            exp_avg_sq = saved_state_dict['state'][param_idx]['exp_avg_sq'].to(model.device)
            assert exp_avg.size() == param_size
            param_name_to_saved_state_dict[param_name] = {'exp_avg': exp_avg, 'exp_avg_sq': exp_avg_sq}
    
    return param_name_to_saved_state_dict


def collect_info(dataloader, 
                 model, 
                 output_dir, 
                 dataset_uid_list, 
                 info_to_collect=['grads', 'reps', 'losses'],
                 proj_dim: List[int] = [1024, 2048, 4096, 8192], 
                 rep_layers: List[int] = [-1],
                 projector_batch_size: int = 16, 
                 save_interval: int = None, 
                 adam_optimizer_state: Optional[dict] = None, 
                 model_type: str = 'clm',
                 grad_types: List[str] = ['sgd', 'adam'], 
                 rng_seed: int = 0):
    '''
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
    '''

    torch.random.manual_seed(rng_seed)  # set the random seed for torch

    os.makedirs(output_dir, exist_ok=True)

    save_interval = save_interval if save_interval is not None else projector_batch_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    number_of_params = get_number_of_params(model)

    # prepare optimization states
    adam_avg, adam_avg_sq = None, None
    if 'grads' in info_to_collect and 'adam' in grad_types:
        assert adam_optimizer_state is not None
        # first and second moment estimates
        adam_avg, adam_avg_sq = prepare_optimizer_state(model, adam_optimizer_state, device)
        assert adam_avg.size() == adam_avg_sq.size()
        assert adam_avg.size(0) == number_of_params

    # initialize a project for each target projector dimension
    projector_class, block_size = get_trak_projector(device), 128
    projectors = {}
    for dim in proj_dim:
        projectors[dim] = projector_class(
            grad_dim=number_of_params,
            proj_dim=dim, 
            seed=rng_seed, 
            proj_type=ProjectionType.rademacher, 
            device=device, 
            dtype=dtype, 
            block_size=block_size, 
            max_batch_size=projector_batch_size
        )
        
    # projected_gradients
    info_to_collect_str = []
    for info_item in info_to_collect:
        if info_item == 'grads':
            assert len(grad_types) > 0, 'At least one gradient type should be provided.'
            for grad_type in grad_types:
                info_to_collect_str.append(f'{info_item}-{grad_type}')
        else:
            info_to_collect_str.append(info_item)
    logger.info(f'Collecting items: {" ".join(info_to_collect_str)}...')
            
    full_grads = {grad_type: [] for grad_type in grad_types}
    projected_grads = {grad_type: {dim: [] for dim in proj_dim} for grad_type in grad_types}
    uids, reps, log_probs, probs = [], [], [], []
    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        if batch_idx == 0:
            if 'grads' in info_to_collect: 
                assert batch['input_ids'].size(0) == 1, 'Only support batch size 1 for collecting gradients'

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = forward_model(model, batch, output_hidden_states='reps' in info_to_collect)
        # collect data uids and losses
        batch_uids = []
        for data_idx in batch['data_idx']:
            if data_idx.dim() == 0 or data_idx.size(0) == 1:
                batch_uids.append(dataset_uid_list[int(data_idx)])
            else:
                # select the first value in the tensor
                batch_uids.append(dataset_uid_list[int(data_idx[0])])
        uids.extend(batch_uids)

        if 'losses' in info_to_collect:
            logits = outputs.logits.detach()  # batch_size (x seq_len) x vocab_size
            if model_type == 'clm':
                # code for collect full log_probs and probs for each token

                # eop_pos = ((batch['labels'] == -100) & (batch['attention_mask'] == 1)).sum(dim=1)
                # eor_pos = (batch['attention_mask'] == 1).sum(dim=1)
                # for instance_idx in range(logits.size(0)):
                #     instance_logits = logits[instance_idx, eop_pos[instance_idx]:eor_pos[instance_idx]]  # seq_len x vocab_size
                #     instance_labels = batch['labels'][instance_idx, eop_pos[instance_idx]:eor_pos[instance_idx]]  # seq_len
                #     log_probs.append(torch.nn.functional.log_softmax(
                #         instance_logits, dim=-1)[torch.arange(instance_logits.size(0)), instance_labels])
                #     probs.append(torch.nn.functional.softmax(
                #         instance_logits, dim=-1)[torch.arange(instance_logits.size(0)), instance_labels])

                # code for collect average log_probs and probs for each instance

                mask = (batch['attention_mask'] == 1) & (batch['labels'] != -100)
                pseudo_labels = torch.where(mask, batch['labels'], torch.zeros_like(batch['labels']))
                log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
                log_prob = torch.gather(log_prob, 2, pseudo_labels[:, :, None])[:, :, 0]
                log_prob = (log_prob * (1.0 * mask)).sum(dim=1) / mask.sum(dim=1)
                prob = torch.nn.functional.softmax(logits, dim=-1)
                prob = torch.gather(prob, 2, pseudo_labels[:, :, None])[:, :, 0]
                prob = (prob * (1.0 * mask)).sum(dim=1) / mask.sum(dim=1)

            elif model_type == 'seq_cls' or model_type == 'mc':
                log_prob = torch.nn.functional.log_softmax(logits, dim=-1)[
                    torch.arange(logits.size(0), device=logits.device), batch['labels']]
                prob = torch.nn.functional.softmax(logits, dim=-1)[
                    torch.arange(logits.size(0), device=logits.device), batch['labels']]

            log_probs.append(log_prob.cpu())  # num_instances,
            probs.append(prob.cpu())

        if 'grads' in info_to_collect:
            # collect full gradients
            outputs.loss.backward()
            vectorized_grads = obtain_vectorized_grads(model)
            for grad_type in grad_types:
                full_grads[grad_type].append(obtain_gradients(
                    vectorized_grads, gradient_type=grad_type, adam_avg=adam_avg, adam_avg_sq=adam_avg_sq))
            model.zero_grad()

            # projection
            if (batch_idx + 1) % projector_batch_size or batch_idx == len(dataloader) - 1:
                for grad_type in grad_types:
                    full_grads[grad_type] = torch.cat(full_grads[grad_type], dim=0).type(torch.float32)
                    for dim in proj_dim:
                        projected_grads[grad_type][dim].append(
                            (projectors[dim].project(
                            full_grads[grad_type], model_id=rng_seed)).cpu())
                    full_grads[grad_type] = []

        if 'reps' in info_to_collect:
            rep = torch.stack([(outputs.hidden_states[layer_idx]).detach() for layer_idx in rep_layers], dim=0)
            if model_type == 'clm':
                # code for collect full representations for each token

                # eop_pos = ((batch['labels'] == -100) & (batch['attention_mask'] == 1)).sum(dim=1) - 1
                # eor_pos = (batch['attention_mask'] == 1).sum(dim=1) - 1
                # for instance_idx in range(batch_reps.size(1)):
                #     instance_reps = batch_reps[:, instance_idx, eop_pos[instance_idx]:eor_pos[instance_idx]]  # num_layers x seq_len x hidden_size
                #     reps.append(instance_reps.type(torch.float32).cpu())

                # code for collect the last representations for each instance for CLM
                rep = rep[
                    torch.arange(rep.size(0), device=rep.device).view(-1, 1, 1), 
                    torch.arange(rep.size(1), device=rep.device).view(1, -1, 1),
                    ((batch['attention_mask'] == 1).sum(dim=1) - 1).view(1, -1, 1).expand(
                        len(rep_layers), rep.size(1), 1)
                ][:, :, 0, :]

            elif model_type == 'seq_cls':
                # collect the CLS token representations for sequence classification models
                rep = rep[:, :, 0, :]  
            
            elif model_type == 'mc':
                rep = rep[:, :, 0, :] 
                num_choices = batch['input_ids'].size(1)
                # average the representations of the choices
                rep = rep.view(rep.size(0), -1, num_choices, rep.size(2)).mean(dim=2)

            else:
                raise ValueError(f'model type {type(model)} is not supported for representation collection yet.')

            reps.append(rep.type(torch.float32).cpu())  # num_layers, num_instances x hidden_size
            
        # save the projected gradients and representations
        if (batch_idx + 1) % save_interval == 0 or batch_idx == len(dataloader) - 1:
            logger.info(f'Saving data at batch {batch_idx + 1}...')
            saved_data = {'uid': uids}

            if 'losses' in info_to_collect:
                saved_data['log_prob'] = torch.cat(log_probs, dim=0)
                saved_data['prob'] = torch.cat(probs, dim=0)

            if 'reps' in info_to_collect:
                saved_data['reps'] = torch.cat(reps, dim=1)

            if 'grads' in info_to_collect:
                if len(full_grads[grad_types[0]]) > 0:
                    for grad_type in grad_types:
                        full_grads[grad_type] = torch.cat(full_grads[grad_type], dim=0).type(torch.float32)
                        for dim in proj_dim:
                            projected_grads[grad_type][dim].append((projectors[dim].project(
                                full_grads[grad_type], model_id=rng_seed)).cpu())
                        full_grads[grad_type] = []

                for grad_type in grad_types:
                    saved_data[f'grads-{grad_type}'] = {
                        dim: torch.cat(v, dim=0) for dim, v in projected_grads[grad_type].items()}
            
            hashed_filename = hashlib.md5(f"{time.time()}_{os.getpid()}".encode()).hexdigest()
            torch.save(saved_data, os.path.join(output_dir, f'{hashed_filename}.pt'))

            projected_grads = {grad_type: {dim: [] for dim in proj_dim} for grad_type in grad_types}
            uids, reps, log_probs, probs = [], [], [], []

    logger.info('Finished')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the YAML configuration file')
    cli_args = parser.parse_args()
    
    with open(cli_args.config, 'r') as fin:
        config = yaml.safe_load(fin)
    
    # Explicitly assign each argument from the YAML config
    args = argparse.Namespace()
    args.seed = config.get('seed', 42)
    args.model_name = config['model_name']
    args.model_weights_dir = config.get('model_weights_dir', None)
    args.use_lora = config.get('use_lora', False)
    args.model_type = config.get('model_type', 'clm')

    args.task_name = config['task_name']
    args.dataset_name = config['dataset_name']
    args.dataset_split = config['dataset_split']
    args.max_seq_length = config.get('max_seq_length', 512)
    args.batch_size = config.get('batch_size', 1)
    args.selected_uid_path = config.get('selected_uid_path', None)

    args.info_to_collect = config.get('info_to_collect', ['grads', 'reps', 'loss'])
    args.grad_types = config.get('grad_types', ['sgd', 'adam'])
    args.grad_proj_dim = config.get('grad_proj_dim', [1024, 2048, 4096, 8192])
    args.proj_batch_size = config.get('proj_batch_size', 16)
    args.rep_layers = config.get('rep_layers', [-1])
    args.output_dir = config['output_dir']
    args.save_interval = config.get('save_interval', None)

    args.log_level = config.get('log_level', 'INFO')
    args.wandb_name = config.get('wandb_name', None)
    args.noise_rate = config.get('noise_rate', None)

    return args


def main():
    args = parse_args()
    assert torch.cuda.is_available(), 'CUDA is required for this script'
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(os.path.join(args.output_dir, 'info_collection.log'))
        ],
    )
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(args.log_level)
    datasets.utils.logging.set_verbosity(args.log_level)
    transformers.utils.logging.set_verbosity(args.log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Setup WandB
    wandb.init(
        name=args.wandb_name,
        config=vars(args)
    )

    set_seed(args.seed)

    # Write the arguments to a file in the output directory
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Load the model
    if args.model_type == 'clm':
        model_class = AutoModelForCausalLM
    elif args.model_type == 'seq_cls':
        model_class = AutoModelForSequenceClassification
    elif args.model_type == 'mc':
        model_class = AutoModelForMultipleChoice
    else:
        raise ValueError(f'Unsupported model type: {args.model_type}')

    if args.use_lora: 
        if args.model_type == 'clm':
            model = AutoPeftModelForCausalLM.from_pretrained(args.model_weights_dir, torch_dtype=torch.bfloat16)
        else:
            model = model_class.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(model, args.model_weights_dir)
    else:
        model = model_class.from_pretrained(
            args.model_weights_dir if args.model_weights_dir is not None else args.model_name, 
            torch_dtype=torch.bfloat16)
    model.to('cuda')

    # Enable gradient computation when collecting gradients
    adam_optimizer_state = None
    if 'grads' in args.info_to_collect:
        if args.use_lora:
            for n, p in model.named_parameters():
                if 'lora' in n or 'Lora' in n:
                    p.requires_grad = True 
            model.print_trainable_parameters()
        else:
            for n, p in model.named_parameters():
                # freeze embeddings
                if 'embeddings' in n:
                    p.requires_grad = False
            print('All parameters require gradients')
        
        # load adam states
        if 'adam' in args.grad_types:
            assert args.model_weights_dir is not None, 'model_weights_dir is required for collecting adam gradients'
            bin_optimizer_state_path = os.path.join(args.model_weights_dir, 'optimizer.bin')
            pt_optimizer_state_path = os.path.join(args.model_weights_dir, 'optimizer.pt')
            assert os.path.exists(bin_optimizer_state_path) or os.path.exists(pt_optimizer_state_path), \
                'optimizer.bin or optimizer.pt is required for collecting adam gradients'
            if os.path.exists(bin_optimizer_state_path):
                adam_optimizer_state = torch.load(bin_optimizer_state_path)['state']
            else:
                adam_optimizer_state = load_pt_adam_state(model, pt_optimizer_state_path)

    # Load the dataset
    if args.dataset_split == 'train':
        # load selected uids if provided
        if args.selected_uid_path is not None:
            selected_uids = []
            with open(args.selected_uid_path) as f_selected_uids:
                for line in f_selected_uids:
                    selected_uids.append(line.strip())
        else:
            selected_uids = None
        dataloader = DATALOADER_DICT[args.task_name](
            max_length=args.max_seq_length, train_dataset_name=args.dataset_name)
    else:
        dataloader = DATALOADER_DICT[args.task_name](max_length=args.max_seq_length)
    
    # Read the data uids to collect
    if args.dataset_split == 'train':
        data, data_info = dataloader.load_train(model_name=args.model_name, uid=selected_uids, noise_rate=args.noise_rate)
    elif args.dataset_split == 'dev':
        data, data_info = dataloader.load_dev(
            model_name=args.model_name, dataset_names=[args.dataset_name])[args.dataset_name]
    elif args.dataset_split == 'test':
        data, data_info = dataloader.load_test(
            model_name=args.model_name, dataset_names=[args.dataset_name])[args.dataset_name]
    assert 'uid' in data_info[0], 'uid is required in data_info for collecting data'
    uid_list = data_info['uid']
    tokenizer = dataloader.tokenizer_dict[args.model_name]

    # Resize token embeddings if the tokenizer has more tokens than the model
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            if model.get_output_embeddings() is not None:
                model.get_output_embeddings().weight.requires_grad = False

    # Create batch data collator and loader
    if args.model_type == 'clm':
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding='longest', label_pad_token_id=-100, pad_to_multiple_of=8)
    elif args.model_type == 'seq_cls':
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8) 
    elif args.model_type == 'mc':
        data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8)

    if 'grads' in args.info_to_collect or 'loss' in args.info_to_collect:
        batch_size = 1
    else:
        batch_size = args.batch_size
    data = torch.utils.data.DataLoader(data, batch_size=batch_size, collate_fn=data_collator)

    # Collect the information
    grad_context = torch.inference_mode() if 'grads' not in args.info_to_collect else torch.enable_grad()
    with grad_context:
        collect_info(
            dataloader=data, 
            model=model, 
            output_dir=args.output_dir, 
            dataset_uid_list=uid_list, 
            info_to_collect=args.info_to_collect, 
            proj_dim=args.grad_proj_dim,
            rep_layers=args.rep_layers,
            projector_batch_size=args.proj_batch_size,
            save_interval=args.save_interval,
            adam_optimizer_state=adam_optimizer_state, 
            model_type=args.model_type, 
            grad_types=args.grad_types,
            rng_seed=args.seed
        )

    wandb.finish()


if __name__ == '__main__':
    main()
