###################################################################
#  mmot_data.py
#  Created by XQ at 2023/9/20 17:05
###################################################################

import os
import torch
from torch import nn
import json
import torch.utils.data as data
from nuscenes import NuScenes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.loaders import load_gt
from torch.utils.data import DataLoader
import scipy.sparse as sp
import numpy as np

class MMotDataset(data.Dataset):
    '''
        Multi-Model Multi Object Detection Dataset
    '''
    def __init__(self,
                 info_path,
                 train_flag = True,
                 max_shift = [10, 5],
                 maxlen_time_window = 5,
                 max_num_nodes = 16000
                 ):
        
        super().__init__()
        self.infos = self.read_json(info_path)
        self.train_flag = train_flag
        self.scence_num = len(self.infos)
        self.max_shift = max_shift
        self.maxlen_time_window = maxlen_time_window
        self.max_num_nodes = max_num_nodes
        self.loop = True
        
    def read_json(self, info_path):
        with open(info_path, 'r') as f:
            infos = json.load(f)
        return infos
    
    def row_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1), dtype=np.float)
        # if rowsum <= 0, keep its previous value
        # rowsum[rowsum <= 0] = 1
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
        
    def __len__(self):
        return self.scence_num
    
    def __getitem__(self, index):
        '''
        [x, y, z, l, w, h, theta, v_x, v_y, time, score, category, model_index]
        return the node features, adjacent matrix A, ids and scores together
        '''
        info = self.infos[index]
        nodes = torch.tensor(info['bbox_tensor'])
        ids = torch.tensor(info['bbox_id'])
        scores = torch.tensor(info['bbox_score'])

        # Adjacency matrix
        node_num = len(nodes)
        A = torch.ones(node_num, node_num)
        # cal shift
        centers = nodes[:, :3]
        t = nodes[:, 9][:, None]
        t = t.repeat(1, node_num)
        shift = torch.cdist(centers, centers)
        # t_map  true: ti == tj   false: ti != tj   i,j is the index of bbox
        t_map = (t - t.T) == 0
        t_window_map = (abs(t - t.T)) <= self.maxlen_time_window
        
        A_x = (shift <= self.max_shift[0]) & (t_map == False) & t_window_map
        A_y = (shift <= self.max_shift[1]) & (t_map)
        A = A * (A_x | A_y)
        
        # if self.loop:
        #     A = np.array(A) + sp.eye(A.shape[0])
        
        A = torch.tensor(self.row_normalize(A), dtype=torch.float32)
        
        # padding
        # _, fdim = node.shape
        # node_ = torch.cat([node, torch.zeros(self.max_num_nodes - node_num, fdim)], dim=0)
        # A_ = torch.zeros(self.max_num_nodes, self.max_num_nodes)
        # A_[:node_num, :node_num] = A
        # scores_ = torch.zero_(self.)
        
        # gt_id_ = -1 * torch.ones(self.max_num_nodes)
        # gt_id_[:node_num] = gt_id
        return nodes, A, ids, scores
        
from tqdm import tqdm     

def main():
    info_path = 'datasets/ensemble_info/ensemble_info_train.json'
    
    data = MMotDataset(info_path)
    for i in tqdm(range(data.__len__())):
        nodes, A, ids, scores = data.__getitem__(i)
        
        t =  nodes[:, 9]
        model_index = nodes[:, 12]
        t_unique = set(t.tolist())
        frame_num = len(t_unique)
        
        for frame in t_unique:
            bbox_inds = torch.bitwise_and(t == frame, model_index != 0)
            gt_bbox_inds = torch.bitwise_and(t == frame, model_index == 0)
            
            if (bbox_inds.sum() == 0 or gt_bbox_inds.sum() == 0):
                frame_num -= 1
                assert frame_num != 0
                continue

if __name__ == "__main__":
    main()