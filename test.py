import argparse
import os
import pickle
import time
from data.mmot_data import MMotDataset
from models.gcn import GCN
from torch.utils.data import DataLoader
import numpy as np
from utils.meters import AverageMeter
from utils.serialization import load_checkpoint
from utils.loss import My_loss
from tqdm import tqdm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from utils.gen_results import gen_results_json
import json

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc_dataroot', type=str, default='/home/zhangxq/datasets/nuscenes')
    parser.add_argument('--info_path', type=str, default='datasets/ensemble_info/ensemble_info_val.json')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default="logs/1213/epoch_50.ckpt")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--dis_th', type=float, default=0.5)
    parser.add_argument('--save_path', type=str, default='save_results')
    parser.add_argument('--origin_info_path', type=str, default='datasets/ensemble_info/new_ensemble_origin_info_val.json')
    parser.add_argument('--save_flag', type=bool, default=True)
    args = parser.parse_args()
    
    return args

def predict_id(node_list, dis_th):
    pred_list = []
    for nodes in tqdm(node_list):
        Z = linkage(nodes, 'average')
        f = fcluster(Z, dis_th, 'distance')
        pred_list.append(f)
    return pred_list

def gen_sample_token_result(nodes, token):
    result = []
    for node in nodes:
        sample_result = dict()
        sample_result['sample_token'] = token
        sample_result['translation'] = node[:3]
        sample_result['size'] = node[3:6]
        sample_result['rotation'] = node
        result.append(sample_result)

def gen_result(nodes, pred, scenes_token):
    # [x, y, z, l, w, h, theta, v_x, v_y, time, score, category, model_index, id]
    print("finish")
    results = []
    time = nodes[:, 9]
    nodes = np.concatenate([nodes, pred[:, None]], axis=1)
    for i, token in enumerate(scenes_token):
        node = nodes[time == i]
        results.append(gen_sample_token_result(node, token))
        

def val(loader, net):
    data_time = AverageMeter()
    node_list = []
    conf_list = []
    
    for i, (node, adj, ids, scores) in tqdm(enumerate(loader)):
        node, adj = map(lambda x: x.cuda(), (node, adj))
        output, conf = net(node, adj)
            
        node_list.append(output[0].detach().cpu().numpy())
        conf_list.append(conf[0].detach().cpu().numpy())
    return node_list, conf_list


def main():
    
    args = parse_config()
    # load data
    # data = MMotDataset(args.info_path, train_flag=False)
    # valloader = DataLoader(
    #             data, batch_size=args.batch_size,
    #             num_workers=args.workers, shuffle=False, pin_memory=True)

    # # init model
    # net = GCN()
    # # load checkpoint
    # ckpt = load_checkpoint(args.checkpoint)
    # net.load_state_dict(ckpt['state_dict'])
    # net = net.cuda()
    # criterion = My_loss()
    
    # node_list, conf_list = val(valloader, net)
    # if args.save_flag:
    #     node_list_path = os.path.join(args.save_path, 'node_list.pkl')
    #     conf_list_path = os.path.join(args.save_path, 'conf_list.pkl')
    #     with open(node_list_path, 'wb') as f:
    #         pickle.dump(node_list, f)
    #     with open(conf_list_path, 'wb') as f:
    #         pickle.dump(conf_list, f)
    
    with open('save_results/node_list.pkl', 'rb') as f:
        node_list = pickle.load(f)
        
    with open('save_results/conf_list.pkl', 'rb') as f:
        conf_list = pickle.load(f)
    
    # pred_lists = predict_id(node_list, args.dis_th)
    # if args.save_flag:
    #     pred_list_path = os.path.join(args.save_path, 'new_pred_list_th_'+str(args.dis_th) + '.pkl')
    #     with open(pred_list_path, 'wb') as f:
    #         pickle.dump(pred_lists, f)
    
    with open('save_results/new_pred_list_th_0.5.pkl', 'rb') as f:
        pred_lists = pickle.load(f)
    
    result_infos = gen_results_json(args.origin_info_path, node_list, pred_lists, conf_list)
    if args.save_flag:
        result_save_path = os.path.join(args.save_path, 'tracking_result_th_'+str(args.dis_th) + '.json')
        with open(result_save_path, 'w') as f:
            json.dump(result_infos, f)
    
if __name__ == "__main__":
    main()