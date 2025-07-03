import os
import hashlib
import time 
import random
import yaml
import json


def construct_train_yaml(
        output_dir, 
        run_name,
        do_eval,
        eval_strategy,
        batch_size_per_device,
        gradient_accumulation_steps,
        learning_rate,
        num_train_epochs,
        warmup_ratio,
        save_strategy,
        save_only_model,
        random_seed, 
        model_name,
        task_name,
        train_dataset_name,
        selected_uid_path,
        max_seq_length,
        use_lora,
        save_lora_classifiers=False,
        noise_rate=None,
    ):
    if isinstance(use_lora, str):
        use_lora = use_lora.lower()
        assert use_lora in ['true', 'false']
        use_lora = True if use_lora == 'true' else False
    else:
        assert isinstance(use_lora, bool)
    
    data = {
        # training arguments
        'output_dir': output_dir,
        'run_name': run_name,
        'overwrite_output_dir': True,
        'do_train': True,
        'do_eval': do_eval,
        'eval_strategy': eval_strategy,
        'per_device_train_batch_size': batch_size_per_device,
        'per_device_eval_batch_size': batch_size_per_device,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'learning_rate': learning_rate,
        'weight_decay': 0,
        'max_grad_norm': 1.0,
        'num_train_epochs': num_train_epochs,
        'lr_scheduler_type': 'linear',
        'warmup_ratio': warmup_ratio,
        'log_level': 'passive',
        'logging_strategy': 'steps',
        'logging_first_step': False,
        'logging_steps': 10,
        'save_strategy': save_strategy,
        'save_only_model': save_only_model,
        'use_cpu': False,
        'seed': random_seed,
        'data_seed': random_seed,
        'bf16': True,
        'tf32': True,
        'ddp_backend': None,
        'debug': '',
        'optim': 'adamw_torch',
        'report_to': 'wandb',
        'skip_memory_metrics': True,
        'resume_from_checkpoint': False,
        'gradient_checkpointing': False,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},
        # model arguments
        'model_name_or_path': model_name,
        'config_name': model_name,
        'tokenizer_name': model_name,
        'token': 'your huggingface token here',
        # data arguments
        'task_name': task_name,
        'train_dataset_name': train_dataset_name,
        'overwrite_cache': False,
        'selected_uid_path': selected_uid_path,
        'max_seq_length': max_seq_length,
        'lora': use_lora,
        'qlora': False,
        'lora_r': 64,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'lora_target_modules': 'all-linear',
    }
    if use_lora and save_lora_classifiers:
        data['lora_modules_to_save'] = [
            'classifier.bias',
            'classifier.weight',
            'pooler.dense.bias',
            'pooler.dense.weight',
        ]
    if noise_rate is not None:
        data['noise_rate'] = noise_rate
    yaml_content = yaml.dump(data)

    return yaml_content


def construct_feature_yaml(
        random_seed, 
        model_name,
        model_weights_dir,
        output_dir, 
        use_lora,
        model_type,
        task_name,
        dataset_name,
        dataset_split,
        max_seq_length,
        batch_size,
        info_to_collect,
        grad_types,
        grad_proj_dim,
        run_name,
    ):
    if isinstance(use_lora, str):
        use_lora = use_lora.lower()
        assert use_lora in ['true', 'false']
        use_lora = True if use_lora == 'true' else False
    else:
        assert isinstance(use_lora, bool)
    
    data = {
        'seed': random_seed,
        'model_name': model_name,
        'model_weights_dir': model_weights_dir,
        'output_dir': output_dir,
        'use_lora': use_lora,
        'model_type': model_type,
        'task_name': task_name,
        'dataset_name': dataset_name,
        'dataset_split': dataset_split,
        'max_seq_length': max_seq_length,
        'batch_size': batch_size,
        'info_to_collect': info_to_collect,
        'grad_types': grad_types,
        'grad_proj_dim': grad_proj_dim,
        'proj_batch_size': 16,
        'rep_layers': [-1],
        'save_interval': 160,
        'log_level': 'INFO',
        'wandb_name': run_name,
    }
    yaml_content = yaml.dump(data)

    return yaml_content


def compute_batch_size(batch_size, max_batch_size):
    gradient_accumulation_steps = 1
    while batch_size > max_batch_size:
        assert batch_size % 2 == 0, f'batch_size {batch_size} is not divisible'
        batch_size = batch_size // 2
        gradient_accumulation_steps *= 2
    return batch_size, gradient_accumulation_steps
        

def exe_gpu_job(
        commands, 
        job_name, 
        job_time_length='1:00:00', 
        num_gpus=1, 
        script_dir='temp_sbatch_scripts'
    ):
    '''
    Execute a job on a GPU node using SLURM.
    commands: list of strings, each string is a command to be executed.
    job_name: str, name of the job.
    job_time_length: str, time length of the job.
    num_gpus: int, number of GPUs to be used.
    script_dir: str, directory to save the temporary SLURM script.
    '''

    header = f'''#!/bin/bash -x
#SBATCH --cpus-per-task=18
#SBATCH --job-name={job_name}
#SBATCH -p gpu
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --time={job_time_length}
#SBATCH --error=sbatch_files/errors/{job_name}.txt
#SBATCH --output=sbatch_files/outputs/{job_name}.txt

module load 2024
module load cuDNN/9.5.0.50-CUDA-12.6.0
module load Python/3.12.3-GCCcore-13.3.0

        '''
    if len(commands) == 0 or (len(commands) == 1 and commands[0].startswith('source')):
        print(f'No commands to execute for {job_name}.')
        return

    os.makedirs(script_dir, exist_ok=True)
    file_path = os.path.join(script_dir, f'{job_name}.sh')

    with open(file_path, 'w') as f:
        f.write(header.strip())
        f.write("\n\n")
        f.write("\n".join(commands))

    num_train_commands = len([cmd for cmd in commands if 'train.py' in cmd])
    print(f'Number of train commands: {num_train_commands}')

    bash_command = f'sbatch {file_path}'
    print(bash_command)
    # os.system(bash_command)


def create_experiment_id():
    random_number = str(random.randint(0, 10000)).encode('utf-8')
    current_time = str(time.time()).encode('utf-8')
    return hashlib.sha256(current_time + random_number).hexdigest()


def args_to_cmd(args, header=None):
    cmd = header if header else ''
    for key, value in args.items():
        # if value is bool value, add flag only
        if isinstance(value, bool):
            if value:
                cmd += f' --{key}'
        else: 
            if isinstance(value, list):
                value = ' '.join(map(str, value))
            cmd += f' --{key} {value}'
    return cmd


def create_ckpt_epoch_to_lr(ckpt_epochs=None, peak_lr=2e-5, num_train_epochs=15, warmup_ratio=0.1):
    if ckpt_epochs is None:
        ckpt_epochs = list(range(num_train_epochs))
    warmup_epochs = num_train_epochs * warmup_ratio
    warmup_lr_inc = peak_lr / warmup_epochs
    anneal_epochs = num_train_epochs - warmup_epochs
    anneal_lr_dec = peak_lr / anneal_epochs

    lr_schedule = []
    start_lr = 0
    for start_epoch in range(num_train_epochs):
        end_epoch = start_epoch + 1
        if end_epoch < warmup_epochs:
            end_lr = warmup_lr_inc * end_epoch
            avg_lr = (start_lr + end_lr) / 2
        elif end_epoch >= warmup_epochs:
            end_lr = peak_lr - anneal_lr_dec * (end_epoch - warmup_epochs)
            if start_epoch >= warmup_epochs:
                avg_lr = (start_lr + end_lr) / 2
            else:
                avg_lr = (peak_lr + start_lr) / 2 * (warmup_epochs - start_epoch) \
                    + (peak_lr + end_lr) / 2 * (end_epoch - warmup_epochs)
        start_lr = end_lr
        lr_schedule.append(avg_lr)

    ckpt_epoch_to_lr = {}
    last_ckpt_epoch = 0
    for ckpt_epoch in ckpt_epochs:
        ckpt_epoch_span = ckpt_epoch - last_ckpt_epoch + 1
        span_avg_lr = sum(lr_schedule[last_ckpt_epoch:ckpt_epoch + 1]) / ckpt_epoch_span
        ckpt_epoch_to_lr[ckpt_epoch] = span_avg_lr

        last_ckpt_epoch = ckpt_epoch + 1

    return ckpt_epoch_to_lr
    

def check_training_completeness(train_dataset, run_dir):
    assert train_dataset in ['cad', 'winogrande-xl', 'dialog_sum']
    check_func = check_vllm_task_completeness if train_dataset == 'dialog_sum' \
        else check_trainer_task_completeness
    return check_func(run_dir)
        

def check_trainer_task_completeness(run_dir):
    filenames = os.listdir(run_dir)
    try:
        assert 'trainer_state.json' in filenames
        with open(os.path.join(run_dir, 'trainer_state.json')) as fin:
            trainer_state = json.load(fin)
        num_train_epochs = trainer_state['num_train_epochs']
        log_history = trainer_state['log_history']
        assert int(log_history[-1]['epoch']) == num_train_epochs
    except:
        return False
    return True


def check_vllm_task_completeness(run_dir):
    filenames = os.listdir(run_dir)
    try:
        assert 'trainer_state.json' in filenames
        with open(os.path.join(run_dir, 'trainer_state.json')) as fin:
            trainer_state = json.load(fin)
        num_train_epochs = trainer_state['num_train_epochs']
        assert 'eval_results' in filenames
        eval_result_dirs = os.listdir(os.path.join(run_dir, 'eval_results'))
        assert len(eval_result_dirs) >= num_train_epochs
        for eval_result_dir in eval_result_dirs:
            eval_result_files = os.listdir(os.path.join(run_dir, 'eval_results', eval_result_dir))
            assert 'metrics.json' in eval_result_files
    except:
        return False
    
    return True