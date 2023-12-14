import numpy as np
import pickle
import json
from tqdm import tqdm
import torch

def get_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    results = data['results']
    meta = data['meta']
    return results, meta

def gen_bbox_result(bbox, id, score):
    bbox['tracking_id'] = str(id)
    bbox['tracking_name'] = bbox['detection_name']
    bbox['tracking_score'] = float(score)
    # bbox.pop('detection_name')
    # bbox.pop('detection_score')
    # bbox.pop('attribute_name')
    # bbox.pop('id')
    return bbox

def merge_id(bbox_list, feat, merge_type = 'max'):
    if merge_type == 'max':
        # from ipdb import set_trace
        # set_trace()
        scores = torch.tensor([x['tracking_score'] for x in bbox_list])
        score_sorted, inds = torch.sort(scores, descending=True)
        bbox_sorted = [bbox_list[ind] for ind in inds.tolist()]
        ids = torch.tensor([int(x['tracking_id']) for x in bbox_sorted])
        
        used = torch.zeros_like(scores)
        cluster_bbox_result = []
        while not (used > 0).all():
            usable_inds = torch.where(used < 1)[0].tolist()
            bbox_sorted = [bbox_sorted[ind] for ind in usable_inds]
            ids = ids[usable_inds]
            used = used[usable_inds]
            cluster_id = int(bbox_sorted[0]['tracking_id'])
            cluster_ind = torch.where(ids == cluster_id)[0].tolist()
            used[cluster_ind] = 1
            cluster_bbox_result.append(bbox_sorted[0])
    elif merge_type == 'feat_max':
        cluster_bbox_result = []
        if feat.dim() == 1:
            feat = feat[None, :]
        ids = torch.tensor([int(x['tracking_id']) for x in bbox_list])
        id_set = ids.unique()
        # id_centers = torch.zeros(len(id_set), D).cuda()
        for i, id in enumerate(id_set):
            inds = np.where(ids == id)[0]
            id_feat = feat[ids == id]
            id_center_feat = torch.mean(id_feat, dim = 0)
            # id_centers[i] = id_center_feat
            id_feat_norm = id_feat / torch.norm(id_feat, dim=1, keepdim=True)
            id_center_norm = id_center_feat / torch.norm(id_center_feat)
            similarity = torch.mm(id_feat_norm, id_center_norm[None, :].T).squeeze()
            max_ind = similarity.argmax()
            cluster_bbox_result.append(bbox_list[inds[max_ind]])

    return cluster_bbox_result

def gen_results_json(infos_path, node_list, pre_lists, conf_lists):
    results, meta = get_results(infos_path)
    scenes_index = 0
    pre_list = pre_lists[0]
    feat_list = node_list[0]
    conf_list = conf_lists[0].squeeze()
    ind = 0
    
    tracking_results = dict()
    for k, v in tqdm(results.items()):
        tracking_result = []
        feat = []
        for bbox in v:
            bbox_result = gen_bbox_result(bbox, pre_list[ind], conf_list[ind])
            tracking_result.append(bbox_result)
            feat.append(feat_list[ind])
            ind += 1
        tracking_result = merge_id(tracking_result, torch.tensor(feat))
        if (ind == len(pre_list)):
            scenes_index += 1
            ind = 0
            if (scenes_index < len(pre_lists)):
                pre_list = pre_lists[scenes_index]
                feat_list = node_list[scenes_index]
                conf_list = conf_lists[scenes_index].squeeze()
        tracking_results[k] = tracking_result
    
    infos = dict()
    infos['meta'] = meta
    infos['results'] = tracking_results
    
    return infos

def main():
    # model_output_folder = './datasets/new_model_output'   
    # save_info_path = './datasets/ensemble_origin_info.json'
    
    # model_results = []
    # model_output = os.listdir(model_output_folder)
    # model_num = len(model_output)
    # for i in range(model_num):
    #     if model_output[i][:3] == 'val':
    #         model_output_path = os.path.join(model_output_folder, model_output[i])
    #         result, meta = get_results(model_output_path)
    #         model_results.append(result)
    
    # results = dict()
    # tokens = list(result.keys())
    # for token in tokens:
    #     token_result = []
    #     for model_index in range(3):
    #         result = model_results[model_index]
    #         token_result += result[token]
    #     results[token] = token_result

    # infos = dict()
    # infos['meta'] = meta
    # infos['results'] = results
    
    # with open(save_info_path, 'w') as f:
    #     json.dump(infos, f) 
    
    infos_path = 'datasets/ensemble_info/ensemble_origin_info_val.json'
    
    with open('save_results/filter_pred_list_th_4.0.pkl', 'rb') as f:
        pre_lists = pickle.load(f)
    
    with open('save_results/filter_node_list_th_4.0.pkl', 'rb') as f:
        node_list = pickle.load(f)
    
    infos = gen_results_json(infos_path, node_list, pre_lists)
    result_save_path = 'save_results/new_tracking_result.json'
    with open(result_save_path, 'w') as f:
        json.dump(infos, f)
    print('finish')
            
                
if __name__ == "__main__":
    main()