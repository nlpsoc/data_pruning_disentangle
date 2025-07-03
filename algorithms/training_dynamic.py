import os
import torch
from transformers.utils import logging

from .dpm import (
    DataPruningMetric, 
    kmeans_clustering, 
    load_training_dynamic, 
    load_predicted_training_dynamic,
    load_info_from_multi_saved_dirs, 
    filter_data_with_uids
)

logger = logging.get_logger(__name__)


# proto-tl, less-tl (need to collect static training dynamics)
class TrainingDynamicPruning(DataPruningMetric):

    def __init__(self, ref_run_dir):
        super().__init__(ref_run_dir=ref_run_dir)
        self.uids = None
        self.training_dynamic = None

    def load_training_dynamic_uid(self, min_length=0):
        training_dynamic, data_indices = load_training_dynamic(
            os.path.join(self.ref_run_dir, 'training_dynamics'), min_length=min_length)
        uids = self.load_uids(os.path.join(self.ref_run_dir, 'train_uids.txt'))
        assert len(data_indices) == len(uids), \
            f'data instance num mismatch: {len(data_indices)} vs {len(uids)}'
        self.training_dynamic, self.uids = training_dynamic, uids
    
    def load_predicted_training_dynamic_uid(self, predicted_training_dynamic_dirs, override=True):
        predicted_training_dynamic, uids = load_info_from_multi_saved_dirs(
            predicted_training_dynamic_dirs, load_predicted_training_dynamic)
        
        if override:
            self.training_dynamic, self.uids = predicted_training_dynamic, uids
            return

        if self.uids is not None and set(uids) == set(self.uids):
            logger.info('UIDs are the same as the reference run. No need to update the training dynamics')
            return
        
        if self.uids is not None or self.training_dynamic is not None:
            logger.info('Existing training dynamics found. Updating the predicted training dynamics...')
            assert len(self.uids) == self.training_dynamic['prob'].size(0), \
                f'Existing UIDs-training dynamic instance num mismatch: '\
                f'{len(self.uids)} vs {self.training_dynamic["prob"].size(0)}'
            for k in self.training_dynamic:
                assert k in predicted_training_dynamic, \
                    f'training dynamics {k} not found in predicted training dynamic'
                assert self.training_dynamic[k].size(1) == predicted_training_dynamic[k].size(1), \
                    f'Training dynamic {k} num epochs mismatch: '\
                    f'{self.training_dynamic[k].size(1)} vs {predicted_training_dynamic[k].size(1)}'
            # select the missing UIDs and append the corresponding predicted training dynamics
            missing_uids_indices, missing_uids = zip(*[(
                idx, uid) for idx, uid in enumerate(uids) if uid not in set(self.uids)])
            missing_uids_indices = torch.LongTensor(missing_uids_indices)
            self.uids.extend(missing_uids)
            self.training_dynamic = {
                k: torch.cat([
                    self.training_dynamic[k], 
                    predicted_training_dynamic[k][missing_uids_indices]], 
                    dim=0) 
                for k in self.training_dynamic}

        else:
            logger.info('No existing training dynamics. Loading the predicted training dynamics...')
            self.training_dynamic, self.uids = predicted_training_dynamic, uids

    def filter_data_with_uids(self, target_uids):
        self.training_dynamic, self.uids = filter_data_with_uids(
            self.uids, target_uids, self.training_dynamic)
    

class DatasetCartographyPruning(TrainingDynamicPruning):
    # https://arxiv.org/abs/2009.10795
    # https://aclanthology.org/2022.findings-emnlp.540
    # https://arxiv.org/abs/2310.12118

    def __init__(self, ref_run_dir):
        super().__init__(ref_run_dir=ref_run_dir)
    
    def scoring_func(self, scoring_rule='log_prob', method='hard', min_length=0):
        if self.training_dynamic is None:
            logger.warning('warning: training dynamics are not loaded. loading from reference run dir...')
            self.load_training_dynamic_uid(min_length=min_length)
        
        if scoring_rule == 'prob':  # Original dataset cartography or CHIA scores for generation tasks
            raw_scores = self.training_dynamic['prob']
        elif scoring_rule == 'log_prob':  # Inverse PPL for generation tasks
            raw_scores = self.training_dynamic['log_prob'].exp()
        
        if method == 'hard':  
            scores = 1 - raw_scores.mean(dim=1)
        elif method == 'ambiguous':
            scores = raw_scores.std(dim=1, unbiased=False)
        else: 
            raise NotImplementedError(f'{method} is not supported by dataset cartography')

        return scores


class S2LPruning(TrainingDynamicPruning):
    # An implementation of the S2L pruning algorithm
    # Reference: https://arxiv.org/abs/2403.07384, Algorithm 1

    def __init__(self, ref_run_dir):
        super().__init__(ref_run_dir=ref_run_dir)
    
    def scoring_func(self, num_clusters=100, max_kmeans_iter=20, min_length=0):
        if self.training_dynamic is None:
            logger.warning('warning: training dynamics are not loaded. loading from reference run dir...')
            self.load_training_dynamic_uid(min_length=min_length)

        assert 'log_prob' in self.training_dynamic, 'log_prob is not found in training dynamics'
        assert self.training_dynamic['log_prob'].size(1) > 1, \
            'At least 2 epochs are required for clustering in S2L pruning'
        cluster_labels = kmeans_clustering(
            -self.training_dynamic['log_prob'], num_clusters, max_kmeans_iter)[1]
        return cluster_labels
    
    @classmethod
    def select_data_from_score(cls, uids, cluster_labels, selection_ratio, **kwargs):
        num_clusters = int(cluster_labels.max().item() + 1)
        cluster_sizes = torch.bincount(cluster_labels)
        sorted_cluster_indices = torch.argsort(cluster_sizes, descending=True)

        logger.info('Sampling from clusters...')
        num_selected, keep_data_indices, budget_size = 0, [], int(selection_ratio * len(uids))
        for sample_idx, cluster_idx in enumerate(sorted_cluster_indices):
            cluster_size = cluster_sizes[cluster_idx].item()
            cluster_indices = torch.arange(cluster_labels.size(0))[cluster_labels == cluster_idx]
            sample_size = min(int((budget_size - num_selected) / (num_clusters - sample_idx)), cluster_size)
            num_selected += sample_size
            keep_data_indices.append(cluster_indices[torch.randperm(cluster_size)[:sample_size]])

        keep_data_indices = torch.cat(keep_data_indices, dim=0)
        keep_uids = set([uids[idx.item()] for idx in keep_data_indices])
        logger.info(f'{keep_data_indices.size(0)} examples are selected')

        return keep_uids

