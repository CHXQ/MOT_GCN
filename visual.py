import argparse
import json
import os
from typing import Tuple, List, Dict, Any
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking.render import recall_metric_curve, summary_plot
from nuscenes.eval.tracking.utils import print_final_metrics
from nuscenes.eval.common.data_classes import EvalBoxes
from matplotlib import pyplot as plt
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import pickle
from tqdm import tqdm

def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     colormap: dict,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: float = 50,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    # boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    # boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.tracking_score
        box_est.label = int(box_est_global.tracking_id)
        box_est.name = box_est_global.tracking_name

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.set_facecolor('white')

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')
    

    # Show GT boxes.
    # for box in boxes_gt:
    #     box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            color = colormap[str(box.label)]
            box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

class TrackingEval:
    """
    This is the official nuScenes tracking evaluation code.
    Results are written to the provided output_dir.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/tracking for more details.
    """
    def __init__(self,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 colormap: dict,
                 first_tokens: list,
                 verbose: bool = True,
                 render_classes: List[str] = None):
        """
        Initialize a TrackingEval object.
        :param config: A TrackingConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param nusc_version: The version of the NuScenes dataset.
        :param nusc_dataroot: Path of the nuScenes dataset on disk.
        :param verbose: Whether to print to stdout.
        :param render_classes: Classes to render to disk or None.
        """
        self.cfg = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.render_classes = render_classes

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        # Initialize NuScenes object.
        # We do not store it in self to let garbage collection take care of it and save memory.
        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)

        # Load data.
        if verbose:
            print('Initializing nuScenes tracking evaluation')
        pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose)
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances.
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering tracks')
        pred_boxes = filter_eval_boxes(nusc, pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        gt_boxes = filter_eval_boxes(nusc, gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = pred_boxes.sample_tokens
        
        scence_index = 0
        frame_ind = 0
        for sample_token in self.sample_tokens:
            if (sample_token == first_tokens[scence_index]):
                if scence_index == 30:
                    break
                save_dir = os.path.join(self.output_dir, first_tokens[scence_index])
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                scence_index += 1
                frame_ind = 0
    
            
            bbox_gt_list = gt_boxes.boxes[sample_token]
            bbox_pred_list = pred_boxes.boxes[sample_token]
            
            gt_annotations = EvalBoxes()
            pred_annotations = EvalBoxes()
            gt_annotations.add_boxes(sample_token, bbox_gt_list)
            pred_annotations.add_boxes(sample_token, bbox_pred_list)
            save_path = os.path.join(save_dir, str(frame_ind))
            visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, colormap, savepath=save_path)
            frame_ind += 1

        # # Convert boxes to tracks format.
        # self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        # self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)


def image_to_video(image_path, media_path, fps):
    '''
    图片合成视频函数
    :param image_path: 图片路径
    :param media_path: 合成视频保存路径
    :return:
    '''
    # 获取图片路径下面的所有图片名称
    image_names = os.listdir(image_path)
    # 对提取到的图片名称进行排序
    image_names.sort(key=lambda n: int(n[:-4]))
    # 设置写入格式
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    image = Image.open(image_path + image_names[0])
    # 初始化媒体写入对象
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
    # 遍历图片，将每张图片加入视频当中
    for image_name in image_names:
        im = cv2.imread(os.path.join(image_path, image_name))
        media_writer.write(im)
    # 释放媒体写入对象
    media_writer.release()


def main():
    # Settings.
    eval_set_ = 'val'
    dataroot_ = '/home/zhangxq/datasets/nuscenes'
    version_ = 'v1.0-trainval'
    config_path = ''
    render_curves_ = 1
    verbose_ = 1
    render_classes_ = ''
    result_path_ = 'save_results/filter_tracking_result_th_1.5.json'
    output_dir_ = './visual_result/filter_tracking_result_th_1.5'
    image_save_path = os.path.join(output_dir_, 'images')
    video_save_path = os.path.join(output_dir_, 'videos')
    # 设置每秒帧数
    fps = 3

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))
    with open('datasets/maps/tracking_colormap.pkl', 'rb') as f:
        colormap_ = pickle.load(f)
    with open('datasets/first_frame_token/nusc_first_token_val.json', 'r') as f:
        first_tokens_ = json.load(f)
        
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)

    nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=image_save_path,
                                nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                                render_classes=render_classes_, colormap = colormap_, first_tokens = first_tokens_)
    
    
    # convert images to videos
    scences_list = os.listdir(image_save_path)
    scences_list.sort()
    for i, scence in tqdm(enumerate(scences_list)):
        img_folder_path = os.path.join(image_save_path, scence + '/')
        save_video_path = os.path.join(video_save_path, str(i) + '_' + scence + '.mp4')
        # 调用函数，生成视频
        image_to_video(img_folder_path, save_video_path, fps)

if __name__ == '__main__':
    main()