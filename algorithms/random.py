import os
import torch
from transformers.utils import logging
from .dpm import DataPruningMetric

logger = logging.get_logger(__name__)


class RandomPruning(DataPruningMetric):

    def __init__(self, ref_run_dir):
        super().__init__(ref_run_dir=ref_run_dir)
        self.uids = None

    def scoring_func(self):
        if self.uids is None:
            logger.warning('UIDs are not loaded yet. Loading UIDs from reference run...')
            self.uids = self.load_uids(os.path.join(self.ref_run_dir, 'train_uids.txt'))
        # Assign a random score from 0 to 1 to each sample
        return torch.rand(len(self.uids))
    