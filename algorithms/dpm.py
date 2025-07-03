import os
from typing import Union 
from collections import defaultdict

from tqdm.auto import tqdm
import torch
import faiss
from transformers.utils import logging

logger = logging.get_logger(__name__)


def kmeans_clustering(features, num_clusters, max_iter=20, verbose=False, gpu=False, seed=42, spherical=False):
    # reference: https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
    kmeans = faiss.Kmeans(
        features.shape[1], num_clusters, 
        niter=max_iter, verbose=verbose, gpu=gpu, seed=seed, 
        spherical=spherical, 
    )
    kmeans.train(features)
    centroids = kmeans.centroids
    distances, labels = kmeans.index.search(features, 1)
    return torch.FloatTensor(centroids), \
    torch.LongTensor(labels[:, 0]), \
    torch.FloatTensor(distances[:, 0])


def load_training_dynamic(training_dynamic_dir, min_length=0):
    # load training dynamics, 
    # min_length: minimum number of data instances (applicable only when training for one epoch)
    training_dynamic = defaultdict(list)
    for filename in os.listdir(training_dynamic_dir):
        if not filename.endswith('.pt'):
            continue
        file_training_dynamic = torch.load(
            os.path.join(training_dynamic_dir, filename), map_location='cpu')
        
        for batch_idx in range(len(file_training_dynamic['data_idx'])):
            batch_size = file_training_dynamic['data_idx'][batch_idx].size(0)
            for k in file_training_dynamic:
                batch_v = file_training_dynamic[k][batch_idx]
                if k in ['global_step', 'epoch']:
                    batch_v = [batch_v for _ in range(batch_size)]
                    if k == 'epoch':
                        batch_v = torch.floor(torch.FloatTensor(batch_v)).long()
                    else:
                        batch_v = torch.LongTensor(batch_v)
                training_dynamic[k].append(batch_v)
    training_dynamic = {k: torch.cat(v, dim=0) for k, v in training_dynamic.items()}

    # sort by global step
    global_step = training_dynamic['global_step']
    training_dynamic = {k: v[global_step.argsort()] for k, v in training_dynamic.items()}

    # restructuring by data_idx
    log_probs, probs = {}, {} 
    for i, data_idx in enumerate(training_dynamic['data_idx']):
        data_idx = data_idx.item()
        if data_idx not in log_probs:
            log_probs[data_idx] = []
            probs[data_idx] = []
        log_probs[data_idx].append(training_dynamic['log_prob'][i])
        probs[data_idx].append(training_dynamic['prob'][i])

    # fill missing epochs with the value of the last epoch
    mode_num_epochs = torch.LongTensor([len(log_probs[data_idx]) for data_idx in log_probs]).mode().values.item()
    data_idx_bin_count = torch.bincount(training_dynamic['data_idx'], minlength=min_length)
    last_epoch_avg_prob, last_epoch_avg_log_prob = None, None
    missing_data_indices = set((data_idx_bin_count < mode_num_epochs).nonzero().squeeze().numpy().tolist())
    if len(missing_data_indices) > 0:
        for data_idx in missing_data_indices:
            if data_idx not in log_probs:
                log_probs[data_idx], probs[data_idx] = [], []
            if data_idx in log_probs and len(log_probs[data_idx]) != 0:
                sur_log_prob, sur_prob = log_probs[data_idx][-1], probs[data_idx][-1]
            else:
                if last_epoch_avg_log_prob is None:
                    last_epoch_avg_log_prob, last_epoch_avg_prob = [], []
                    for i in probs:
                        if i in missing_data_indices or probs[i][-1].isnan() or log_probs[i][-1].isnan():
                            continue
                        last_epoch_avg_log_prob.append(log_probs[i][-1])
                        last_epoch_avg_prob.append(probs[i][-1])
                    last_epoch_avg_log_prob, last_epoch_avg_prob = torch.stack(last_epoch_avg_log_prob).mean(), torch.stack(last_epoch_avg_prob).mean()
                sur_log_prob, sur_prob = last_epoch_avg_log_prob, last_epoch_avg_prob
                
            log_probs[data_idx].append(sur_log_prob)
            probs[data_idx].append(sur_prob)
    
    data_indices, training_dynamic = [], {'log_prob': [], 'prob': []}
    for data_idx in sorted(log_probs.keys()):
        data_indices.append(data_idx)
        training_dynamic['log_prob'].append(torch.FloatTensor(log_probs[data_idx]))
        training_dynamic['prob'].append(torch.FloatTensor(probs[data_idx]))
    training_dynamic['log_prob'] = torch.stack(training_dynamic['log_prob'], dim=0)  # num_instances x num_epochs
    training_dynamic['prob'] = torch.stack(training_dynamic['prob'], dim=0)

    # fill nan with 0
    training_dynamic['log_prob'][training_dynamic['log_prob'].isnan()] = 0
    training_dynamic['prob'][training_dynamic['prob'].isnan()] = 0

    return training_dynamic, data_indices


def load_info_from_multi_saved_dirs(
        saved_dirs: Union[str, list[str]], 
        load_fn: callable,
        **kwargs
    ):
    
    def reindex_data(info, order_index):
        if isinstance(info, dict):
            return {k: reindex_data(v, order_index) for k, v in info.items()}
        else:
            assert torch.is_tensor(info), f'Unsupported data type: {type(info)}'
            return info[order_index]
    
    if isinstance(saved_dirs, str):
        saved_dirs = [saved_dirs]

    if len(saved_dirs) == 0:
        return None, None
    elif len(saved_dirs) == 1:
        return load_fn(saved_dirs[0], **kwargs)

    uids, uid_to_idx, info = None, None, []
    for saved_dir in saved_dirs:
        info_part, uids_part = load_fn(saved_dir, **kwargs)
        if uids is None:
            uids = uids_part
            uid_to_idx = {uid: idx for idx, uid in enumerate(uids)}
        else:
            assert len(uids) == len(uids_part), 'UID mismatch'
            uids_order_index = torch.argsort(torch.LongTensor([uid_to_idx[uid] for uid in uids_part]))
            info_part = reindex_data(info_part, uids_order_index)
        
        # training dynamic: {prob: num_instances,  log_prob: num_instances}
        # hidden_states: num_instances x hidden_dim
        # grads: num_instances x grad_dim
        info.append(info_part)  
    
    # concat info
    if isinstance(info[0], dict):  # [{k: v}, {k: v}, ...] -> {k: stacked_v}
        info = {k: torch.stack([info_part[k] for info_part in info], dim=1) for k in info[0]}
    else:
        info = torch.stack(info, dim=1)  # num_instances x num_saved_dirs (num_epochs) x hidden_dim/grad_dim
    
    return info, uids


def load_predicted_training_dynamic(saved_dir):
    uids, probs, log_probs = [], [], []
    for filename in os.listdir(saved_dir):
        if not filename.endswith('.pt'):
            continue
        saved_data = torch.load(os.path.join(saved_dir, filename), map_location='cpu')

        file_uids = saved_data['uid']
        file_log_probs = saved_data['log_prob']
        file_probs = saved_data['prob']
        assert file_log_probs.size(0) == len(file_uids), \
            f'num instances log prob mismatch: {file_log_probs.size(0)} vs {len(file_uids)}'
        assert file_probs.size(0) == len(file_uids), \
            f'num instances prob mismatch: {file_probs.size(0)} vs {len(file_uids)}'
        
        uids.extend(file_uids)
        log_probs.append(file_log_probs)
        probs.append(file_probs)

    log_probs = torch.cat(log_probs, dim=0)  # num_instances 
    probs = torch.cat(probs, dim=0)

    # fill nan with 0
    log_probs[log_probs.isnan()] = 0
    probs[probs.isnan()] = 0

    return {'prob': probs, 'log_prob': log_probs}, uids


def load_losses(saved_dir):
    prob_dict, uids = load_predicted_training_dynamic(saved_dir)
    return -prob_dict['log_prob'].float(), uids


def load_last_layer_hidden_states(saved_dir):
    uids, hidden_states = [], []
    for filename in os.listdir(saved_dir):
        if not filename.endswith('.pt'):
            continue
        saved_data = torch.load(os.path.join(saved_dir, filename), map_location='cpu')
        file_uids = saved_data['uid']
        file_reps = saved_data['reps'][-1]
        assert file_reps.size(0) == len(file_uids), f'num instances mismatch: {file_reps.size(0)} vs {len(file_uids)}'

        uids.extend(file_uids)
        hidden_states.append(file_reps)  # num_instances x hidden_dim

    hidden_states = torch.cat(hidden_states, dim=0)
    
    return hidden_states, uids


def load_grads(saved_dir, grad_dim=8192, grad_type='sgd'):
    uids, grads = [], []
    for filename in os.listdir(saved_dir):
        if not filename.endswith('.pt'):
            continue
        saved_data = torch.load(os.path.join(saved_dir, filename), map_location='cpu')

        file_uids = saved_data['uid']
        file_grads = saved_data[f'grads-{grad_type}'][grad_dim]
        assert file_grads.size(0) == len(file_uids), f'num instances mismatch: {file_grads.size(0)} vs {len(file_uids)}'
        assert file_grads.size(1) == grad_dim, f'grad dim mismatch: {file_grads.size(1)} vs {grad_dim}'

        uids.extend(file_uids)
        grads.append(file_grads)
        
    grads = torch.cat(grads, dim=0)  # num_instances x grad_dim

    if len(set(uids)) != len(uids):
        logger.warning('Duplicate UIDs found in the grad data, deduplicating...')
        uid_to_idx = {}
        for uid in uids:
            if uid not in uid_to_idx:
                uid_to_idx[uid] = len(uid_to_idx)
        uids = list(uid_to_idx.keys())
        grads = grads[torch.LongTensor([uid_to_idx[uid] for uid in uids], device=grads.device)]

    return grads, uids



def filter_data_with_uids(source_uids, target_uids, data):
    # assert self.uids is not None, 'UIDs are not loaded yet' 
    assert set(target_uids).issubset(set(source_uids)), 'UIDs mismatch'
    uid_to_idx = {uid: idx for idx, uid in enumerate(source_uids)}
    selected_indices = torch.LongTensor([
        uid_to_idx[uid] for uid in target_uids])
    if isinstance(data, dict):
        return {k: data[k][selected_indices] for k in data}, target_uids
    else:
        assert torch.is_tensor(data), f'Unsupported data type: {type(data)}'
        return data[selected_indices], target_uids


class DataPruningMetric():

    def __init__(self, ref_run_dir=None):
        self.ref_run_dir = ref_run_dir
    
    def scoring_func(
            self, 
            **kwargs,
    ):
        raise NotImplementedError
    
    @classmethod
    def load_uids(cls, train_uid_path):
        return [str(line.strip()) for line in open(train_uid_path)]
    
    @classmethod
    def select_data_from_score(cls, uids, scores, selection_ratio, uid_to_label=None, balance_label=False):
        assert scores.size(0) == len(uids), 'Mismatch between scores and UIDs'

        if selection_ratio == 1.0:
            return set(uids)
        elif selection_ratio == 0.0:
            return set()

        if balance_label:
            uid_to_score = {uid: score for uid, score in zip(uids, scores)}
            assert uid_to_label is not None, 'UID to label mapping is required for label balancing'
            label_to_uid_score = {label: [[], []] for label in set(uid_to_label.values())}
            for uid, label in uid_to_label.items():
                label_to_uid_score[label][0].append(uid)
                label_to_uid_score[label][1].append(uid_to_score[uid])
            uid_scores = [[label_uids, torch.FloatTensor(label_scores, device=scores.device)] 
                        for label_uids, label_scores in label_to_uid_score.values()]
        else:
            uid_scores = [[uids, scores]]

        keep_uids = set()
        for label_uids, label_scores in uid_scores:
            label_keep_data_indices = torch.arange(label_scores.size(0))[
                label_scores >= torch.quantile(label_scores, 1 - selection_ratio)]
            label_keep_uids = set([label_uids[idx.item()] for idx in label_keep_data_indices])
            keep_uids.update(label_keep_uids)

        logger.info(f'{len(keep_uids)} examples are selected')

        return keep_uids


