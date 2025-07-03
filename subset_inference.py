import os
import sys
import json
import argparse
import logging

import torch

from algorithms import (
    DatasetCartographyPruning,
    S2LPruning,
    HiddenStatePruning,
    RandomPruning, 
    GradientBasedPruning, 
    S2LModifiedPruning,
    PrototypicalityModifiedPruning,
    LessModifiedPruning,
)

logger = logging.getLogger(__name__)

PRUNING_METRICS = {
    's2l': S2LPruning,
    'dataset_cartography': DatasetCartographyPruning,
    'prototypicality': HiddenStatePruning,
    'semdedup': HiddenStatePruning,
    'random': RandomPruning,
    'less': GradientBasedPruning,
    'memorization': GradientBasedPruning, 
    # neo metrics
    'proto-tl': PrototypicalityModifiedPruning,
    'less-tl': LessModifiedPruning,
    's2l-hs': S2LModifiedPruning,
    'less-hs': LessModifiedPruning,
    's2l-grad': S2LModifiedPruning,
    'proto-grad': PrototypicalityModifiedPruning,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Subset Inference')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--selection_ratios', type=float, nargs='+', default=[
        round(x, 2) for x in torch.arange(0.05, 1.01, 0.05).numpy().tolist()])

    parser.add_argument('--pruning_metric', type=str, required=True,
                        choices=[
                            'random', 
                            'dataset_cartography', 's2l', 
                            'prototypicality', 'semdedup', 
                            'less', 'memorization', 'reverse_less', 
                            'proto-tl', 'less-tl',
                            's2l-hs', 'less-hs',
                            's2l-grad', 'proto-grad',
                        ],
                        help='Pruning metric to use')

    # RandomPruning & TrainingDynamicPruning
    parser.add_argument('--ref_run_dir', type=str, help='Reference run directory')
    parser.add_argument('--num_instances', type=int, default=None)
    
    # TrainingDynamicPruning
    parser.add_argument('--predicted_training_dynamic_dirs', type=str, nargs='+', default=None)
    parser.add_argument('--load_predicted_training_dynamic', action='store_true')
    parser.add_argument('--not_override_training_dynamic', action='store_true')
    parser.add_argument('--train_loss_dirs', type=str, nargs='+', default=None)
    parser.add_argument('--val_loss_dirs', type=str, nargs='+', default=None)

    # S2L & HiddenStatePruning
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--max_kmeans_iter', type=int, default=20)

    # HiddenStatePruning
    parser.add_argument('--hidden_state_dir', type=str, help='Hidden states directory')
    parser.add_argument('--val_hidden_state_dir', type=str, default=None, help='Validation hidden states directory')

    # HiddenStatePruning & GradientBasedPruning
    parser.add_argument('--normalize_features', action='store_true')

    # GradientBasedPruning
    parser.add_argument('--train_grad_dirs', type=str, nargs='+', default=None)
    parser.add_argument('--val_grad_dirs', type=str, nargs='+', default=None)
    parser.add_argument('--select_by_subject', action='store_true')
    parser.add_argument('--train_lrs', type=float, nargs='+', default=None)
    parser.add_argument('--grad_dim', type=int, default=8192)
    parser.add_argument('--grad_type', type=str, default='sgd')

    # Label balancing
    parser.add_argument('--balance_label', action='store_true')
    # Only works for CAD
    parser.add_argument('--uid_to_label_file', type=str, default='data/datasets/hsd/cad/cad_train_uid_to_label.json')

    return parser.parse_args()


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout), 
        ],
    )
    args = parse_args()

    # Load the pruning metric
    if args.pruning_metric in ['random', 'dataset_cartography', 's2l']:
        pruning_metric = PRUNING_METRICS[args.pruning_metric](args.ref_run_dir)
    elif args.pruning_metric in ['proto-tl', 'less-tl']:
        pruning_metric = PRUNING_METRICS[args.pruning_metric](
            ref_run_dir=args.ref_run_dir, 
            train_loss_dirs=args.train_loss_dirs, 
            val_loss_dirs=args.val_loss_dirs,
        )
    elif args.pruning_metric in ['prototypicality', 'semdedup']:
        pruning_metric = PRUNING_METRICS[args.pruning_metric](hidden_states_dir=args.hidden_state_dir)
    elif args.pruning_metric in ['s2l-hs', 'less-hs']:
        pruning_metric = PRUNING_METRICS[args.pruning_metric](
            ref_run_dir=args.ref_run_dir, 
            train_hidden_state_dirs=args.hidden_state_dir, 
            val_hidden_state_dirs=args.val_hidden_state_dir, 
        )
    elif args.pruning_metric in ['influential', 'memorization', 'less', 'reverse_less']:
        pruning_metric = PRUNING_METRICS[args.pruning_metric](
            args.train_lrs, args.train_grad_dirs, args.val_grad_dirs)
    elif args.pruning_metric in ['s2l-grad', 'proto-grad']:
        pruning_metric = PRUNING_METRICS[args.pruning_metric](
            ref_run_dir=args.ref_run_dir, 
            train_grad_dirs=args.train_grad_dirs, 
            val_grad_dirs=args.val_grad_dirs,
            train_lrs=[float(train_lr) for train_lr in args.train_lrs],
        )
    else:
        raise ValueError(f'Unsupported pruning metric: {args.pruning_metric}')
    
    
    min_length = args.num_instances if args.num_instances is not None else 0
    # Compute scores for each sample
    if args.pruning_metric == 'random':
        scores = pruning_metric.scoring_func()

    elif args.pruning_metric == 'dataset_cartography':
        override_training_dynamic = not args.not_override_training_dynamic
        if not override_training_dynamic or not args.load_predicted_training_dynamic:
            pruning_metric.load_training_dynamic_uid(min_length=min_length)
        if args.load_predicted_training_dynamic:
            pruning_metric.load_predicted_training_dynamic_uid(
                args.predicted_training_dynamic_dirs, override=override_training_dynamic)
        # scores = pruning_metric.scoring_func(min_length=min_length)
        scores = pruning_metric.scoring_func(min_length=min_length, scoring_rule='prob', method='ambiguous')

    elif args.pruning_metric == 's2l':
        pruning_metric.load_training_dynamic_uid(min_length=min_length)
        if args.load_predicted_training_dynamic:
            pruning_metric.load_predicted_training_dynamic_uid(args.predicted_training_dynamic_dirs)
        scores = pruning_metric.scoring_func(
            num_clusters=args.num_clusters, 
            max_kmeans_iter=args.max_kmeans_iter, 
            min_length=min_length,
        )
    elif args.pruning_metric in ['s2l-hs', 's2l-grad']:
        scores = pruning_metric.scoring_func(
            feature_name='hidden_states' if 'hs' in args.pruning_metric else 'grads',
            num_clusters=args.num_clusters, 
            max_kmeans_iter=args.max_kmeans_iter, 
            normalize=args.normalize_features,
        )

    elif args.pruning_metric in ['prototypicality', 'semdedup']:
        pruning_metric.load_hidden_states()
        scores = pruning_metric.scoring_func(
            num_clusters=args.num_clusters, 
            max_kmeans_iter=args.max_kmeans_iter, 
            method=args.pruning_metric, 
            normalize=args.normalize_features, 
        )
    elif args.pruning_metric in ['proto-tl', 'proto-grad']:
        scores = pruning_metric.scoring_func(
            feature_name='losses' if 'tl' in args.pruning_metric else 'grads',
            num_clusters=args.num_clusters, 
            max_kmeans_iter=args.max_kmeans_iter, 
            normalize=args.normalize_features,
        )

    elif args.pruning_metric in ['less', 'reverse_less', 'memorization']:
        pruning_metric.load_train_grad_uid(
            grad_dirs=args.train_grad_dirs, grad_dim=args.grad_dim, grad_type=args.grad_type)
        if args.pruning_metric in ['less', 'reverse_less']:
            pruning_metric.load_val_grad_uid(
                grad_dirs=args.val_grad_dirs, grad_dim=args.grad_dim)
            assert pruning_metric.train_grads.size(1) == pruning_metric.val_grads.size(1), \
                f'Number of checkpoints mismatch: {pruning_metric.train_grads.size(1)} vs {pruning_metric.val_grads.size(1)}' 
        scores = pruning_metric.scoring_func(
            grad_dim=args.grad_dim, 
            grad_type=args.grad_type, 
            method=args.pruning_metric, 
            normalize=args.normalize_features, 
            select_by_subject=args.select_by_subject,
        )
    elif args.pruning_metric in ['less-tl', 'less-hs']:
        scores = pruning_metric.scoring_func(
            feature_name='losses' if 'tl' in args.pruning_metric else 'hidden_states',
            normalize=args.normalize_features,
        )
    
    if 's2l' not in args.pruning_metric:
        # fill nan values with the smallest value
        logger.info(f'Filling {100 * (scores.isnan().sum().item() / scores.size(0))}\% nan values with the smallest value')
        scores[scores.isnan()] = scores[~scores.isnan()].min()
    
    # The UIDs to select from
    train_uids = pruning_metric.train_uids \
        if args.pruning_metric in [
            'less', 'reverse_less', 'influential', 'memorization', 'less-tl', 'less-hs'] \
        else pruning_metric.uids

    os.makedirs(args.output_dir, exist_ok=True)
    # Save the arguments and scores
    with open(os.path.join(args.output_dir, 'data_selection_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    torch.save(scores, os.path.join(args.output_dir, 'scores.pt'))
    # save all uids
    with open(os.path.join(args.output_dir, 'all_uids.txt'), 'w') as f:
        for uid in train_uids:
            f.write(f'{uid}\n')

    # Select the data and save the UIDs
    if args.balance_label:
        assert args.uid_to_label_file is not None, 'UID to label mapping is required for label balancing'
        with open(args.uid_to_label_file) as f:
            uid_to_label = json.load(f)
    else:
        uid_to_label = None

    for selection_ratio in args.selection_ratios:
        selected_uids = pruning_metric.select_data_from_score(
            train_uids, scores, selection_ratio, 
            uid_to_label=uid_to_label, balance_label=args.balance_label,
        )
        with open(os.path.join(args.output_dir, f'uid_{selection_ratio}.txt'), 'w') as f:
            for uid in selected_uids:
                f.write(f'{uid}\n')


if __name__ == '__main__':
    main()