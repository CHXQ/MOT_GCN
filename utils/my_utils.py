from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d, bbox_overlaps_nearest_3d
import torch

def my_collate_fn(batch):
    batch.sort(key=lambda x : len(x[0]), reverse=True)
    max_len = len(batch[0][0])
    node_list, A_list, id_list, score_list = [], [], [], []
    for sample in batch:
        nodes, A, ids, scores = sample
        node_num, fdim = nodes.shape
        node_ = torch.cat([nodes, torch.zeros(max_len - node_num, fdim)], dim=0)
        A_ = torch.zeros(max_len, max_len)
        A_[:node_num, :node_num] = A
        ids_ = -torch.ones(max_len)
        ids_[:node_num] = ids
        scores_ = -torch.ones(max_len)
        scores_[:node_num] = scores
        node_list.append(node_)
        A_list.append(A_)
        id_list.append(ids_)
        score_list.append(scores_)
    
    batch_data = (torch.stack(node_list, 0), torch.stack(A_list, 0), torch.stack(id_list), torch.stack(score_list))
    return batch_data