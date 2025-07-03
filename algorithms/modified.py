import os
from typing import List, Tuple
import torch
from algorithms import (
    DataPruningMetric, 
    load_last_layer_hidden_states, 
    load_grads, 
    load_losses, 
    load_info_from_multi_saved_dirs,
    filter_data_with_uids, 
    kmeans_clustering, 
    S2LPruning, 
)


class BaseModifiedPruning(DataPruningMetric):

    def __init__(
            self, 
            ref_run_dir: str = None,
            train_loss_dirs: Tuple[str, List[str]] = None,
            val_loss_dirs: Tuple[str, List[str]] = None,
            train_hidden_state_dirs: Tuple[str, List[str]] = None,
            val_hidden_state_dirs: Tuple[str, List[str]] = None,
            train_grad_dirs: Tuple[str, List[str]] = None,
            val_grad_dirs: Tuple[str, List[str]] = None,
            train_lrs: Tuple[float, List[float]] = None,
            grad_dim: int = 1024
        ):
        super().__init__(ref_run_dir=ref_run_dir)

        self.train_uids, self.val_uids = self.load_uids(os.path.join(ref_run_dir, 'train_uids.txt')), None
        self.train_loss_dirs = train_loss_dirs
        self.val_loss_dirs = val_loss_dirs
        self.train_hidden_state_dirs = train_hidden_state_dirs
        self.val_hidden_state_dirs = val_hidden_state_dirs
        self.train_grad_dirs = train_grad_dirs
        self.val_grad_dirs = val_grad_dirs
        self.train_lrs = train_lrs
        self.grad_dim = grad_dim
    
    def load_features(self, feature_name='losses', data_split='train'):
        assert feature_name in ['losses', 'hidden_states', 'grads'], \
            'feature name must be one of [losses, hidden_states, grads]'
        assert data_split in ['train', 'val'], 'data split must be one of [train, val]'

        feature_dirs = {
            ('losses', 'train'): self.train_loss_dirs,
            ('losses', 'val'): self.val_loss_dirs,
            ('hidden_states', 'train'): self.train_hidden_state_dirs,
            ('hidden_states', 'val'): self.val_hidden_state_dirs,
            ('grads', 'train'): self.train_grad_dirs,
            ('grads', 'val'): self.val_grad_dirs
        }[(feature_name, data_split)]
        assert feature_dirs is not None, f'{feature_name} directories for {data_split} data is not provided'

        grad_type = 'adam' if data_split == 'train' else 'sgd'
        grad_dim = self.grad_dim if feature_name == 'grads' else None
        kwargs = {'grad_type': grad_type, 'grad_dim': grad_dim} if feature_name == 'grads' else {}
        load_func = {
            'losses': load_losses,
            'hidden_states': load_last_layer_hidden_states,
            'grads': load_grads
        }[feature_name]

        features, uids = load_info_from_multi_saved_dirs(
            feature_dirs, load_func, **kwargs)
        if data_split == 'val' and self.val_uids is None:
            self.val_uids = uids

        if data_split == 'train':
            features, uids = filter_data_with_uids(uids, self.train_uids, features)
        elif data_split == 'val':
            features, uids = filter_data_with_uids(uids, self.val_uids, features)
        
        return features, uids
    
    @classmethod
    def normalize_features(cls, features):
        feature_norm = torch.norm(features, dim=-1, keepdim=True)
        feature_norm = torch.where(feature_norm == 0, torch.ones_like(feature_norm), feature_norm)
        return features / feature_norm
    
    
class S2LModifiedPruning(BaseModifiedPruning):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uids = self.train_uids
    
    def load_features(self, feature_name='hidden_states', data_split='train'):
        features, uids =  super().load_features(feature_name, data_split)
        assert len(features.size()) in [2, 3], 'features must be 2D or 3D tensor'
        if len(features.size()) == 3:
            if self.train_lrs:
                assert len(self.train_lrs) == features.size(1), 'number of learning rates must match the number of epochs of features'
                print(f'Multiple epochs are offered, averaging features across epochs according to learning rates')
            else:
                self.train_lrs = [1.0] * features.size(1)
                print(f'Multiple epochs are offered but learning rates are not offered, averaging features across epochs, with equal weights')
            # average features across epochs according to learning rates
            features = torch.einsum('ijk,j->ik', features, torch.FloatTensor(
                self.train_lrs, device=features.device))
        return features, uids
    
    def scoring_func(self, feature_name, num_clusters=100, max_kmeans_iter=100, normalize=True):
        assert feature_name in ['hidden_states', 'grads'], \
            'feature name must be one of [hidden_states, grads]'
        features, _ = self.load_features(feature_name, data_split='train')
        features = self.normalize_features(features) if normalize and feature_name != 'losses' else features
        cluster_labels = kmeans_clustering(features, num_clusters, max_kmeans_iter)[1]
        return cluster_labels
    
    def select_data_from_score(cls, uids, cluster_labels, selection_ratio, **kwargs):
        return S2LPruning.select_data_from_score(uids, cluster_labels, selection_ratio, **kwargs)


class PrototypicalityModifiedPruning(BaseModifiedPruning):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uids = self.train_uids
    
    def load_features(self, feature_name='losses', data_split='train'):
        features, uids = super().load_features(feature_name, data_split)
        assert len(features.size()) in [2, 3], 'features must be 2D or 3D tensor'
        if len(features.size()) == 3:
            if self.train_lrs:
                assert len(self.train_lrs) == features.size(1), 'number of learning rates must match the number of epochs of features'
                print(f'Multiple epochs are offered, averaging features across epochs according to learning rates')
            else:
                self.train_lrs = [1.0] * features.size(1)
                print(f'Multiple epochs are offered but learning rates are not offered, averaging features across epochs, with equal weights')
            # average features across epochs according to learning rates
            features = torch.einsum('ijk,j->ik', features, torch.FloatTensor(
                self.train_lrs, device=features.device))
        return features, uids
    
    def scoring_func(self, feature_name, num_clusters=100, max_kmeans_iter=100, normalize=True):
        assert feature_name in ['losses', 'grads'], \
            'feature name must be one of [losses, grads]'
        features, _ = self.load_features(feature_name, data_split='train')
        features = self.normalize_features(features) if normalize and feature_name != 'losses' else features
        return kmeans_clustering(features, num_clusters, max_kmeans_iter)[2]


class LessModifiedPruning(BaseModifiedPruning):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def scoring_func(self, feature_name, normalize=True):
        assert feature_name in ['losses', 'hidden_states'], \
            'feature name must be one of [losses, hidden_states]'
        train_features, _ = self.load_features(feature_name, data_split='train')
        val_features, _ = self.load_features(feature_name, data_split='val')
        train_features = self.normalize_features(train_features) \
            if normalize and feature_name != 'losses' else train_features
        val_features = self.normalize_features(val_features) \
            if normalize and feature_name != 'losses' else val_features
        
        if len(train_features.size()) == 2:
            train_features = train_features.unsqueeze(1)  # (train_size, num_features) or (train_size, num_epochs, num_features)
            val_features = val_features.unsqueeze(1)
        val_features = val_features.sum(dim=0, keepdim=True)  # sum across all validation data, (1, 1, num_features) or (1, num_epochs, num_features) 
        if self.train_lrs and feature_name != 'losses':
            val_features = torch.einsum('ijk,j->ijk', val_features, torch.FloatTensor(
                self.train_lrs, device=val_features.device))
        
        # calculate the dot product between train and validation data
        return torch.einsum('ijk,ljk->il', train_features, val_features).sum(dim=-1)
    