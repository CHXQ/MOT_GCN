import torch
import torch.nn as nn
import numpy as np
import time

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_batch, conf_batch, ids_batch, scores_batch, t_batch, model_index_batch):
        loss = 0
        B, N, D  = x_batch.shape
        for x, conf, ids, scores, t, model_index in zip(x_batch, conf_batch, ids_batch, scores_batch, t_batch, model_index_batch):
            node_num = (ids >= 0).sum()
            x = x[:node_num, :]
            conf = conf.view(-1)[:node_num]
            ids = ids[:node_num]
            scores = scores[:node_num]
            t = t[:node_num]
            model_index = model_index[:node_num]
            t_unique = set(t.tolist())
            frame_num = len(t_unique)
            scene_loss = 0
            
            for frame in t_unique:
                bbox_inds = torch.bitwise_and(t == frame, model_index != 0)
                gt_bbox_inds = torch.bitwise_and(t == frame, model_index == 0)
                
                if (bbox_inds.sum() == 0 or gt_bbox_inds.sum() == 0):
                    frame_num -= 1
                    assert frame_num != 0
                    continue
                
                bbox_x = x[bbox_inds]
                bbox_conf = conf[bbox_inds]
                bbox_score = scores[bbox_inds]
                bbox_id = ids[bbox_inds]
                
                gt_bbox_x = x[gt_bbox_inds]
                gt_bbox_id = ids[gt_bbox_inds]
                
                id2num = {id : i for i, id in enumerate(gt_bbox_id.tolist())}
                gt_bbox_id_num = [id2num[id] for id in gt_bbox_id.tolist()]
                bbox_id_num = torch.tensor([id2num[id] for id in bbox_id.tolist()]).cuda()
                
                bbox_x_norm = bbox_x / torch.norm(bbox_x, dim=1, keepdim=True)
                gt_bbox_x_norm = gt_bbox_x / torch.norm(gt_bbox_x, dim=1, keepdim=True)
            
                similarity = torch.mm(bbox_x_norm, gt_bbox_x_norm.T) * 10
                
                feat_loss_func = nn.CrossEntropyLoss()
                feat_loss = feat_loss_func(similarity, bbox_id_num.cuda())
                conf_loss_func = nn.MSELoss()
                conf_loss = conf_loss_func(bbox_conf, bbox_score.cuda())
                frame_loss = feat_loss + conf_loss
                scene_loss += frame_loss
            loss += scene_loss / frame_num
                # test
                # t1 = torch.softmax(similarity, dim=-1)
                # t2 = torch.log(t1)
                # frame_loss = -torch.sum(torch.mul(t2, y_score.cuda())) / similarity.shape[0]
                # if math.isinf(frame_loss):
                #     print('bug')
                # loss = loss + frame_loss / len(t_unique)
        
        return loss / B


# loss_fun = My_loss()
# x = torch.randn(1, 3, 4)
# id_gt = torch.tensor([0, 2, 0])
# loss = loss_fun(x, id_gt)
# print(loss)