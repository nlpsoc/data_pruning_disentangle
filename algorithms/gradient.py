from typing import List, Union
import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)
from .dpm import (
    DataPruningMetric, 
    load_grads, 
    load_info_from_multi_saved_dirs, 
    filter_data_with_uids
)

logger = logging.get_logger(__name__)

# s2l-grad, proto-grad
class GradientBasedPruning(DataPruningMetric):
    
    def __init__(
            self, 
            train_lrs: Union[List[str], List[float]], 
            train_grad_dirs: List[str], 
            val_grad_dirs: List[str] = None, 
        ):
        super().__init__(ref_run_dir=None)

        self.train_grad_dirs, self.val_grad_dirs = train_grad_dirs, val_grad_dirs
        self.train_lrs = [float(lr) for lr in train_lrs]

        self.train_grads = None
        self.val_grads = None
        self.train_uids = None
        self.val_uids = None
    
    def load_train_grad_uid(self, grad_dirs, grad_dim, grad_type):
        self.train_grads, self.train_uids = load_info_from_multi_saved_dirs(
            grad_dirs, load_grads, grad_dim=grad_dim, grad_type=grad_type)
        assert self.train_grads.dim() == 3 or self.train_grads.dim() == 2, \
            f'Unsupported grad dim: {self.train_grads.dim()}'
        if self.train_grads.dim() == 2:
            self.train_grads = self.train_grads[:, None, :]
            print('Warning: train_grads dim is 2. Adding a new dim at dim=1')
    
    def load_val_grad_uid(self, grad_dirs, grad_dim):
        self.val_grads, self.val_uids = load_info_from_multi_saved_dirs(
            grad_dirs, load_grads, grad_dim=grad_dim, grad_type='sgd')
        assert self.val_grads.dim() == 3 or self.val_grads.dim() == 2, \
            f'Unsupported grad dim: {self.val_grads.dim()}'
        if self.val_grads.dim() == 2:
            self.val_grads = self.val_grads[:, None, :]
            print('Warning: val_grads dim is 2. Adding a new dim at dim=1')
    
    def filter_data_with_uids(self, train_target_uids, val_target_uids=None):
        self.train_grads, self.train_uids = filter_data_with_uids(
            self.train_uids, train_target_uids, self.train_grads)
        if val_target_uids is not None:
            self.val_grads, self.val_uids = filter_data_with_uids(
                self.val_uids, val_target_uids, self.val_grads)

    def scoring_func(
            self, 
            grad_dim=8192, 
            grad_type='sgd', 
            method='less', 
            normalize=False, 
            select_by_subject=False,
        ):
        assert method in ['less', 'influential', 'memorization', 'reverse_less'], \
            f'Unsupported method: {method}'

        if self.train_grads is None or self.train_uids is None:
            logger.warning('Training grads are not provided. Loading training grads...')
            assert self.train_grad_dirs is not None, 'Training grad dir is not provided'
            self.load_train_grad_uid(self.train_grad_dirs, grad_dim=grad_dim, grad_type=grad_type)
        
        if method in ['less', 'reverse_less']:
            if self.val_grads is None or self.val_uids is None:
                logger.warning('Validation grads are not provided. Loading validation grads...')
                assert self.val_grad_dirs is not None, 'Validation grad dir is not provided'
                self.load_val_grad_uid(self.val_grad_dirs, grad_dim=grad_dim)
                assert self.train_grads.size(1) == self.val_grads.size(1), \
                    f'Number of checkpoints mismatch: {self.train_grads.size(1)} vs {self.val_grads.size(1)}' 

            scores = 0
            for epoch_idx in range(self.train_grads.size(1)):
                train_grads = self.train_grads[:, epoch_idx]  # (num_train, grad_dim)
                val_grads = self.val_grads[:, epoch_idx]  # (num_val, grad_dim)
                train_lr = self.train_lrs[epoch_idx]

                if normalize:
                    train_grad_norm = train_grads.norm(dim=-1, keepdim=True)
                    train_grad_norm = torch.where(train_grad_norm == 0, torch.ones_like(train_grad_norm), train_grad_norm)
                    train_grads = train_grads / train_grad_norm

                    val_grad_norm = val_grads.norm(dim=-1, keepdim=True)
                    val_grad_norm = torch.where(val_grad_norm == 0, torch.ones_like(val_grad_norm), val_grad_norm)
                    val_grads = val_grads / val_grad_norm

                if not select_by_subject:
                    summed_val_grads = val_grads.T.sum(dim=1, keepdim=True)  # (grad_dim, 1)
                    
                else:
                    # bbh_dev_{task}_{i}
                    subjects = [val_uid.split('_')[2] for val_uid in self.val_uids]
                    # convert string to int according to the order of appearance
                    subject2idx = {subject: idx for idx, subject in enumerate(set(subjects))}
                    subjects = torch.LongTensor([subject2idx[subject] for subject in subjects])
                    summed_val_grads = []
                    for subject_idx in range(len(subject2idx)):
                        subject_mask = subjects == subject_idx
                        summed_val_grads.append(val_grads[subject_mask].T.sum(dim=1, keepdim=True))
                    summed_val_grads = torch.cat(summed_val_grads, dim=1)

                scores += train_lr * torch.matmul(train_grads, summed_val_grads)  # (num_train, num_subjects)

            if method == 'reverse_less':
                scores = -scores

            scores = scores.max(dim=1).values

        elif method in ['influential', 'memorization']:
            scores = 0
            for epoch_idx in range(self.train_grads.size(1)):
                train_grads = self.train_grads[:, epoch_idx]
                train_lr = self.train_lrs[epoch_idx]
                if normalize:
                    train_grads = train_grads / train_grads.norm(dim=-1, keepdim=True)
                if method == 'influential':
                    scores += train_lr * torch.matmul(train_grads, train_grads.T.sum(dim=1, keepdim=True))[:, 0]
                else:
                    scores += train_lr * (train_grads ** 2).sum(dim=1)

        else:
            raise ValueError(f'Unsupported method: {method}')

        return scores
