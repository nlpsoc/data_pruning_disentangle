import torch
from transformers.utils import logging
from .dpm import (
    DataPruningMetric, 
    kmeans_clustering, 
    load_last_layer_hidden_states, 
    filter_data_with_uids
)

logger = logging.get_logger(__name__)


# s2l-hs, less-hs (need to read validation hidden states)
class HiddenStatePruning(DataPruningMetric):

    def __init__(self, hidden_states_dir):
        super().__init__(ref_run_dir=None)
        self.hidden_states_dir = hidden_states_dir
        self.uids = None
        self.hidden_states = None

        self.cluster_centroids = None
        self.cluster_labels = None
        self.centroid_distances = None
        self.semdedup_max_sim = None
    
    def load_hidden_states(self):
        self.hidden_states, self.uids = load_last_layer_hidden_states(self.hidden_states_dir)
    
    def select_data_from_uids(self, target_uids):
        self.hidden_states, self.uids = filter_data_with_uids(
            self.uids, target_uids, self.hidden_states)
    
    def scoring_func(
            self, 
            num_clusters=10, 
            max_kmeans_iter=20, 
            method='prototypicality', 
            normalize=False, 
        ):
        if self.hidden_states is None or self.uids is None:
            logger.warning('Hidden states are not provided. Loading hidden states...')
            self.load_hidden_states()
        if normalize:
            self.hidden_states = self.hidden_states / self.hidden_states.norm(dim=-1, keepdim=True)

        # Prototypicality: https://arxiv.org/abs/2206.14486
        if method == 'prototypicality':
            scores = kmeans_clustering(self.hidden_states, num_clusters, max_kmeans_iter)[2]

        # SemDeDup: https://arxiv.org/abs/2303.09540
        elif method == 'semdedup':
            _, cluster_labels, centroid_distances = kmeans_clustering(
                self.hidden_states, num_clusters, max_kmeans_iter)
            scores = self._compute_semdedup_max_sim(cluster_labels, centroid_distances)

        else:
            raise NotImplementedError(f'Unsupported method: {method} for feature space pruning metrics')

        return scores
    
    def _compute_semdedup_cluster_max_sim(
            self, cluster_hidden_states, cluster_centroid_distances):
        # reference: https://github.com/facebookresearch/SemDeDup/blob/main/semdedup.py
        ordered_cluster_indices = torch.argsort(cluster_centroid_distances)
        inverse_ordered_cluster_indices = torch.argsort(ordered_cluster_indices)
        ordered_cluster_hidden_states = cluster_hidden_states.clone()[ordered_cluster_indices, :]
        if torch.cuda.is_available():
            ordered_cluster_hidden_states = ordered_cluster_hidden_states.to('cuda:0')
        pair_w_sim_matrix = ordered_cluster_hidden_states @ (ordered_cluster_hidden_states.T)
        del cluster_hidden_states
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
        max_sim = torch.max(triu_sim_mat, dim=1)[0].cpu()
        max_sim = max_sim[inverse_ordered_cluster_indices]
        return -max_sim
    
    def _compute_semdedup_max_sim(self, cluster_labels, centroid_distances):
        semdedup_max_sim = torch.zeros(self.hidden_states.size(0))
        for label in cluster_labels.unique():
            cluster_indices = (cluster_labels == label)
            cluster_hidden_states = self.hidden_states[cluster_indices]
            cluster_centroid_distances = centroid_distances[cluster_indices]
            cluster_max_sim = self._compute_semdedup_cluster_max_sim(
                cluster_hidden_states, cluster_centroid_distances)
            semdedup_max_sim[cluster_indices] = cluster_max_sim
        return semdedup_max_sim

 