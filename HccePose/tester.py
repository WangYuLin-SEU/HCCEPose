# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

"""HccePose inference with optional MegaPose or three-mode WAPR integration.
HccePose 推理，可选 MegaPose 精修或三模式 WAPR 接入。
"""

import json
import os
import random
import time
from numbers import Integral, Real
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch
import torchvision.transforms as transforms
import trimesh
from ultralytics import YOLO

# Postprocessing belongs to Tester; RGB users do not need the optional renderer.
# 后处理由 Tester 统一负责；RGB 用户无需安装可选的 WAPR 渲染器。
try:
    from wapr.suppression import rendered_mask_iou_nms_gpu
except ModuleNotFoundError as exc:
    if exc.name != 'wapr':
        raise
    rendered_mask_iou_nms_gpu = None

from HccePose.bop_loader import (
    IMAGENET_MEAN_BGR,
    IMAGENET_STD_BGR,
    bop_dataset,
    inference_image_geometry,
    inout,
)
from HccePose.network_model import HccePose_BF_Net, load_checkpoint
from HccePose.ensure_rgbd_megapose_demo import ensure_acceleration_backend_environment
from HccePose.hccepose_acceleration import HccePoseOnnxRunner
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.visualization import vis_rgb_mask_Coord, vis_rgb_mask_Coord_origin
from Refinement.MegaPose import _megapose_add_title, _megapose_vstack_images
from Refinement.refinement_group import (
    Refinement_MP,
    Refinement_WAPR,
    restore_full_frame_masks,
)

composed_transforms_img = transforms.Compose([
            transforms.Normalize(IMAGENET_MEAN_BGR, IMAGENET_STD_BGR),
            ])

def draw_annotations_on_image_yolo(image, boxes, confidences, cls, class_names, confidence_threshold=0.5):
    '''
    ---
    ---
    Draw YOLO detection boxes on a BGR uint8 image (OpenCV layout).
    ---
    ---
    This helper renders YOLO 2D detections, class labels and confidence values
    on the input BGR image. It is only used for human-readable visualization
    and does not affect subsequent inference.

    Args:
        - image: BGR uint8 image to draw on (modified in place).
        - boxes: Detection boxes in xywh format.
        - confidences: Per-detection confidence scores.
        - cls: Class indices per detection.
        - class_names: Mapping/list used to label each class id.
        - confidence_threshold: Minimum confidence to draw a box.

    Returns:
        - image: Same array after drawing.
    ---
    ---
    在原图上绘制 YOLO 二维检测框、类别编号与置信度。
    ---
    ---
    该函数不参与后续推理，仅用于 HccePose 的 2D 可视化展示。

    参数:
        - image: 待绘制的 BGR uint8 图像（原地修改）。
        - boxes: xywh 格式的检测框。
        - confidences: 每个检测的置信度。
        - cls: 每个检测的类别索引。
        - class_names: 用于显示类别名的映射或列表。
        - confidence_threshold: 绘制框的最低置信度。

    返回:
        - image: 绘制后的图像。
    '''
    for box, confidence, cl in zip(boxes, confidences, cls):
        if confidence >= confidence_threshold:
            x_min, y_min, x_max, y_max = box[0], box[1], box[0] + box[2], box[1] + box[3]
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), random_color, 2)
            text = f"{'obj_%s'%str(class_names[cl]).rjust(2, '0')}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_color = random_color
            thickness = 2
            text_position = (x_min, y_min - 2)
            cv2.putText(image, text, text_position, font, font_scale, font_color, thickness)
    return image

def crop_trans_batch_hccepose(xywh, out_size = [128, 128], padding_ratio = 1.2):
    '''
    ---
    ---
    Build batched crop transforms for HccePose detections.
    ---
    ---
    Converts YOLO `xywh` detections into batched perspective transforms and
    padded square boxes. The crop is enlarged by `padding_ratio` and mapped to
    a fixed `out_size` for the main network.

    Args:
        - xywh: Tensor of shape (B, 4) in top-left xywh format.
        - out_size: Target (width, height) after warp.
        - padding_ratio: Expands the square radius around each box center.

    Returns:
        - tfs: Batch of 3x3 homographies (B, 3, 3).
        - boxes: Corresponding xywh crop boxes in original image space.
    ---
    ---
    根据 YOLO 的 `xywh` 检测框计算 HccePose 批量裁剪变换与方形扩张框。
    ---
    ---
    裁剪区域按 `padding_ratio` 扩张，并映射到固定 `out_size` 供主网络使用。

    参数:
        - xywh: 形状 (B, 4) 的张量，左上角 xywh 格式。
        - out_size: 透视变换后的目标宽高。
        - padding_ratio: 相对检测框对正方形半径的扩张比例。

    返回:
        - tfs: 批量 3x3 单应矩阵 (B, 3, 3)。
        - boxes: 原图空间中的对应 xywh 裁剪框。
    '''

    def compute_tf_batch(left, right, top, bottom):
        '''
        ---
        ---
        Build translation + scale homographies from pixel-aligned crop bounds.
        ---
        ---
        Rounds bounds, subtracts top-left, then scales to `out_size` for each
        batch element.

        Args:
            - left, right, top, bottom: 1D tensors of shape (B,) in pixel coords.

        Returns:
            - tf: (B, 3, 3) homography mapping crop pixels to normalized grid.
            - boxes: xywh boxes `[left, top, w, h]` per row.
        ---
        ---
        由像素对齐的裁剪边界构造平移 + 缩放的单应矩阵。
        ---
        ---
        对边界取整，减去左上角，再按 `out_size` 做各 batch 的缩放。

        参数:
            - left, right, top, bottom: 形状 (B,) 的像素坐标张量。

        返回:
            - tf: 将裁剪像素映射到目标网格的 (B, 3, 3) 单应。
            - boxes: 每行 `[left, top, w, h]` 的 xywh 框。
        '''
        B = len(left)
        left = left.round()
        right = right.round()
        top = top.round()
        bottom = bottom.round()

        tf = torch.eye(3, device=xywh.device)[None].expand(B,-1,-1).contiguous()
        tf[:,0,2] = -left
        tf[:,1,2] = -top
        new_tf = torch.eye(3, device=xywh.device)[None].expand(B,-1,-1).contiguous()
        new_tf[:,0,0] = out_size[0]/(right-left)
        new_tf[:,1,1] = out_size[1]/(bottom-top)
        tf = new_tf@tf
        
        boxes = torch.cat([left[...,None], top[...,None], right[...,None] - left[...,None], bottom[...,None] - top[...,None]], dim=1)
        return tf, boxes

    center = xywh[:, :2].clone().detach()
    center[:, 0] += xywh[:, 2] / 2
    center[:, 1] += xywh[:, 3] / 2
    
    radius, _ = torch.max(xywh[:, 2:] * padding_ratio / 2, dim=1) #
    left = center[:,0]-radius
    right = center[:,0]+radius
    top = center[:,1]-radius
    bottom = center[:,1]+radius
    tfs, boxes = compute_tf_batch(left, right, top, bottom)
    return tfs, boxes

class Tester:
    """Detect objects and handle PnP, MegaPose or WAPR inference and postprocessing.
    统一执行物体检测、PnP、MegaPose 或 WAPR 推理及后处理。
    """

    @staticmethod
    def _check_wapr_records(records):
        """Validate instance identities, meter poses and scores without mutating input.
        校验实例标识、米制位姿及评分，不修改原始输入。
        """
        if not isinstance(records, list):
            raise ValueError('Expected a list of raw WAPR records')
        checked = []
        seen = set()
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f'Record {index} must be a dictionary')
            object_id = record.get('object_id')
            instance_index = record.get('instance_index')
            score = record.get('score_6d')
            if isinstance(object_id, bool) or not isinstance(object_id, Integral) or object_id <= 0:
                raise ValueError(f'Record {index} needs a positive integer object_id')
            if isinstance(instance_index, bool) or not isinstance(instance_index, Integral) or instance_index < 0:
                raise ValueError(f'Record {index} needs a nonnegative integer instance_index')
            if isinstance(score, bool) or not isinstance(score, Real) or not np.isfinite(score):
                raise ValueError(f'Record {index} needs a finite numeric score_6d')
            identity = (int(object_id), int(instance_index))
            if identity in seen:
                raise ValueError(f'Duplicate instance identity {identity}')
            seen.add(identity)
            pose = np.asarray(record.get('pose_4x4_m'), dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f'Record {index} needs a finite pose_4x4_m of shape [4,4]')
            if not np.array_equal(pose[3], [0, 0, 0, 1]):
                raise ValueError(f'Record {index} needs an object-to-camera homogeneous pose')
            row = dict(record)
            row.update(object_id=identity[0], instance_index=identity[1],
                       score_6d=float(score), pose_4x4_m=pose.tolist())
            checked.append(row)
        return checked

    @staticmethod
    def postprocess_wapr(records, meshes_by_id, cam_K, image_shape, wbps_threshold=0.75,
                         nms_iou_threshold=0.5, device='cuda:0'):
        """Apply WBPS filtering and same-category CAD mask NMS inside Tester.
        在 Tester 内统一执行 WBPS 筛选和同类 CAD 掩码 NMS，保持原实例顺序。
        predict calls this automatically; static access does not load pose weights.
        predict 自动调用本方法；静态调用不会加载位姿权重。
        """
        if not np.isfinite(wbps_threshold):
            raise ValueError('wbps_threshold must be finite; scores need not lie in [0,1]')
        if not np.isfinite(nms_iou_threshold) or not 0 <= nms_iou_threshold <= 1:
            raise ValueError('nms_iou_threshold must be between zero and one')
        if len(image_shape) < 2 or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in image_shape[:2]):
            raise ValueError('image_shape must start with positive integer height and width')
        height, width = map(int, image_shape[:2])
        cam_K = np.asarray(cam_K, dtype=np.float32)
        if cam_K.shape != (3, 3) or not np.isfinite(cam_K).all() or cam_K[0, 0] <= 0 or cam_K[1, 1] <= 0:
            raise ValueError('cam_K must contain finite 3 x 3 original-image pixel intrinsics')
        rows = Tester._check_wapr_records(records)
        groups = {}
        statuses = ['below_score_threshold'] * len(rows)
        for index, row in enumerate(rows):
            if row['score_6d'] >= wbps_threshold:
                groups.setdefault(row['object_id'], []).append(index)
                statuses[index] = 'not_retained_by_mask_nms'
        for object_id in groups:
            if object_id not in meshes_by_id:
                raise ValueError(f'Missing meter mesh for object {object_id}')
        if groups:
            if rendered_mask_iou_nms_gpu is None:
                raise ImportError('Install the documented public WAPR version before rendered-mask NMS')
            for object_id, indices in groups.items():
                poses = [{'obj_id': object_id, 'score_6d': rows[index]['score_6d'],
                          'pose_4x4': rows[index]['pose_4x4_m'], 'input_index': index}
                         for index in indices]
                kept = rendered_mask_iou_nms_gpu(
                    poses, meshes_by_id[object_id], cam_K, height, width, nms_iou_threshold, device)
                for pose in kept:
                    statuses[pose['input_index']] = 'kept'
        kept_indices = [index for index, status in enumerate(statuses) if status == 'kept']
        decisions = [{'input_index': index, 'object_id': row['object_id'],
                      'instance_index': row['instance_index'], 'score_6d': row['score_6d'],
                      'status': statuses[index]} for index, row in enumerate(rows)]
        return {'score_threshold': float(wbps_threshold), 'nms_iou_threshold': float(nms_iou_threshold),
                'nms_method': 'same_object_id_rendered_mask_iou', 'raw_records': rows,
                'decisions': decisions, 'kept_input_indices': kept_indices,
                'retained_records': [rows[index] for index in kept_indices]}

    @staticmethod
    def postprocess_saved_wapr(dataset_root, image_path, intrinsics_path, input_path, output_path,
                               wbps_threshold=0.75, nms_iou_threshold=0.5,
                               device='cuda:0', mesh_scale_to_m=1e-3):
        """Recheck saved WAPR records through Tester without loading inference models.
        通过 Tester 重新筛选已保存的 WAPR 记录，不加载推理模型，不覆盖原始 JSON。
        """
        same_file = os.path.realpath(input_path) == os.path.realpath(output_path)
        if os.path.exists(input_path) and os.path.exists(output_path):
            same_file = same_file or os.path.samefile(input_path, output_path)
        if same_file:
            raise ValueError('Choose a separate output file; raw input must not be overwritten')
        if not np.isfinite(wbps_threshold) or not np.isfinite(nms_iou_threshold) or not 0 <= nms_iou_threshold <= 1:
            raise ValueError('Use a finite score threshold and an IoU threshold in [0,1]')
        if not np.isfinite(mesh_scale_to_m) or mesh_scale_to_m <= 0:
            raise ValueError('mesh_scale_to_m must be finite and positive')
        for path in (dataset_root, image_path, intrinsics_path, input_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError('Cannot decode the original image')
        camera = inout.load_json(intrinsics_path)
        cam_K = inout.load_cam_params(intrinsics_path)['K'] if isinstance(camera, dict) else np.asarray(camera, dtype=np.float32)
        records = Tester._check_wapr_records(inout.load_json(input_path))
        meshes = {}
        for row in records:
            object_id = row['object_id']
            if row['score_6d'] < wbps_threshold or object_id in meshes:
                continue
            mesh_path = os.path.join(dataset_root, 'models', f'obj_{object_id:06d}.ply')
            if not os.path.isfile(mesh_path):
                raise FileNotFoundError(mesh_path)
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            mesh.apply_scale(mesh_scale_to_m)
            if mesh.is_empty or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                raise ValueError(f'Expected a finite triangle mesh at {mesh_path}')
            meshes[object_id] = mesh
        report = Tester.postprocess_wapr(records, meshes, cam_K, image.shape, wbps_threshold,
                                         nms_iou_threshold, device)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        return report

    def __init__(self, bop_dataset_item: bop_dataset, show_op=True, hccepose_vis=None,
                 CUDA_DEVICE='0', top_K=None, crop_size=256, efficientnet_key=None,
                 megapose_model_name_rgb='megapose-1.0-RGB-multi-hypothesis',
                 megapose_model_name_rgbd='megapose-1.0-RGBD',
                 hccepose_acceleration='pytorch', hccepose_acceleration_cache_dir=None,
                 acceleration=None, acceleration_cache_dir=None):
        """Load the detector and object checkpoints. MegaPose is registered on request."""
        self.bop_dataset_item = bop_dataset_item
        self.crop_size = crop_size
        self.top_K = top_K
        if hccepose_vis is not None:
            show_op = hccepose_vis
        self.show_op = show_op
        self.CUDA_DEVICE = CUDA_DEVICE
        if torch.cuda.is_available():
            self.device = device = torch.device("cuda:%s"%CUDA_DEVICE)
            print("GPU is available. Using GPU.")
        else:
            self.device = device = torch.device("cpu")
            print("GPU is not available. Using CPU.")

        self.megapose_registered = False
        self.MegaPose_Item = None
        self.wapr_registered = False
        self.WAPR_Item = None
        self.last_wapr_raw_records = None
        if acceleration is not None:
            hccepose_acceleration = acceleration
        if acceleration_cache_dir is not None:
            hccepose_acceleration_cache_dir = acceleration_cache_dir
        self.acceleration = 'pytorch' if hccepose_acceleration is None else str(hccepose_acceleration).lower()
        if self.acceleration not in ['pytorch', 'onnx', 'tensorrt']:
            raise ValueError('Unsupported HccePose acceleration backend: %s' % hccepose_acceleration)
        self.acceleration_cache_dir = hccepose_acceleration_cache_dir

        _repo_root = str(Path(__file__).resolve().parents[1])
        ensure_acceleration_backend_environment(_repo_root, self.acceleration)

        self.model_yolo = YOLO(os.path.join(bop_dataset_item.dataset_path, 'yolo11', 
                                            'train_obj_s', 'detection', 'obj_s', 
                                            'yolo11-detection-obj_s.pt')).to(device).eval()

        BBox_3d = []
        for key_i in self.bop_dataset_item.model_info:
            model_info_i = self.bop_dataset_item.model_info[key_i]
            min_x = model_info_i['min_x']
            min_y = model_info_i['min_y']
            min_z = model_info_i['min_z']
            size_x = model_info_i['size_x']
            size_y = model_info_i['size_y']
            size_z = model_info_i['size_z']
            max_x = min_x + size_x
            max_y = min_y + size_y
            max_z = min_z + size_z
            pts_3d = np.array([
                [min_x, min_y, min_z],
                [max_x, min_y, min_z],
                [max_x, max_y, min_z],
                [min_x, max_y, min_z],
                [min_x, min_y, max_z],
                [max_x, min_y, max_z],
                [max_x, max_y, max_z],
                [min_x, max_y, max_z]
            ])
            edges = np.array([
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ])
            BBox_3d.append(pts_3d[edges])
        self.BBox_3d = np.array(BBox_3d)
            
            
        self.HccePose_Item = {}
        self.HccePose_Item_info = {}
        self.HccePose_Runtime_Item = {}
        for obj_id in bop_dataset_item.obj_id_list:
            obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
            min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
            size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)

            HccePose_BF_Net_i = HccePose_BF_Net(efficientnet_key=efficientnet_key, 
                                                min_xyz = min_xyz, 
                                                size_xyz = size_xyz)
            if torch.cuda.is_available():
                HccePose_BF_Net_i=HccePose_BF_Net_i.to('cuda:'+CUDA_DEVICE)
                HccePose_BF_Net_i.eval()
            best_save_path = os.path.join(bop_dataset_item.dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'), 'best_score')
            info_i = load_checkpoint(
                best_save_path, HccePose_BF_Net_i, CUDA_DEVICE=CUDA_DEVICE, inference_object_id=obj_id,
            )
            self.HccePose_Item[obj_id] = HccePose_BF_Net_i
            self.HccePose_Item_info[obj_id] = info_i
            if self.acceleration in ['onnx', 'tensorrt']:
                cache_dir_i = self.acceleration_cache_dir
                if cache_dir_i is None:
                    cache_dir_i = os.path.join(bop_dataset_item.dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'), 'onnx_cache')
                self.HccePose_Runtime_Item[obj_id] = HccePoseOnnxRunner(
                    HccePose_BF_Net_i,
                    info_i['checkpoint_path'],
                    cache_dir_i,
                    device=str(device),
                    obj_id=obj_id,
                    input_size=self.crop_size,
                    provider=self.acceleration,
                )
            else:
                self.HccePose_Runtime_Item[obj_id] = HccePose_BF_Net_i

        self.megapose_model_name_rgb = megapose_model_name_rgb
        self.megapose_model_name_rgbd = megapose_model_name_rgbd

    def register_megapose(self, megapose_model_name_rgb=None, megapose_model_name_rgbd=None):
        '''
        ---
        ---
        Register the MegaPose refinement backend for the current dataset.
        ---
        ---
        Creates `Refinement_MP`, registers every BOP mesh in millimeters, and
        marks `megapose_registered`. Used from `predict` when `use_megapose=True`.

        Args:
            - megapose_model_name_rgb / megapose_model_name_rgbd: Optional overrides.

        Returns:
            - None.
        ---
        ---
        创建 `Refinement_MP` 并将当前数据集全部 mesh（毫米）注册到 MegaPose；
        仅在 `predict(..., use_megapose=True)` 时使用。

        参数:
            - megapose_model_name_rgb / megapose_model_name_rgbd: 可选模型名覆盖。

        返回:
            - 无。
        '''
        megapose_item = Refinement_MP(
            device=str(self.device),
            model_name_rgb=megapose_model_name_rgb or self.megapose_model_name_rgb,
            model_name_rgbd=megapose_model_name_rgbd or self.megapose_model_name_rgbd,
        )
        megapose_item.register_objects(self.bop_dataset_item.obj_id_list, self.bop_dataset_item.obj_model_list, mesh_units='mm')
        self.MegaPose_Item = megapose_item
        self.megapose_registered = True
        print('MegaPose is registered.')

    def _run_megapose_refinement(self, obj_id, cam_K, img, depth, pred_Rts,
                                  megapose_vis=False, megapose_vis_stages=None):
        '''
        ---
        ---
        Run MegaPose refinement on HccePose pose hypotheses.
        ---
        ---
        Converts translations to meters for MegaPose, calls `inference_batch`,
        then maps refined translations back to millimeters with several fallback
        shapes for partial updates.

        Args:
            - obj_id: Object id for all hypotheses in the batch.
            - cam_K: Camera intrinsics.
            - img: OpenCV BGR uint8 image (converted to RGB inside the MegaPose job).
            - depth: Optional depth map (float32) or None for RGB-only path.
            - pred_Rts: Initial poses (mm translation).
            - megapose_vis, megapose_vis_stages: Visualization flags.

        Returns:
            - Tuple of refined poses, scores, valid indices, visualization image.
        ---
        ---
        在 HccePose 初始位姿上做 MegaPose 微调（不重新检测）；支持 RGB / RGBD，
        成功后返回更新位姿、分数、有效 id 与可视化。

        参数:
            - obj_id: 当前批假设对应的物体 id。
            - cam_K: 相机内参。
            - img: OpenCV BGR uint8 图像（MegaPose job 内会转为 RGB）。
            - depth: 可选深度；无则走纯 RGB。
            - pred_Rts: 初始位姿（平移毫米）。
            - megapose_vis, megapose_vis_stages: 可视化选项。

        返回:
            - 微调位姿、分数、有效索引、可视化图组成的元组。
        '''
        if self.MegaPose_Item is None:
            return pred_Rts, None, np.array([], dtype=np.int64), None

        pred_Rts_np = np.asarray(pred_Rts, dtype=np.float32)
        if pred_Rts_np.shape[0] == 0:
            return pred_Rts, None, np.array([], dtype=np.int64), None

        initial_poses_m = pred_Rts_np.copy()
        initial_poses_m[:, :3, 3] *= 1e-3
        obj_ids_np = np.full((initial_poses_m.shape[0],), int(obj_id), dtype=np.int64)
        megapose_depth = None if depth is None else depth.astype(np.float32).copy()

        self.MegaPose_Item.last_visualization = None
        megapose_Rts_m, megapose_scores, valid_pose_ids, megapose_vis_image = self.MegaPose_Item.inference_batch(
            cam_K.astype(np.float32),
            img.astype(np.uint8),
            megapose_depth,
            None,
            obj_ids_np,
            initial_poses=initial_poses_m,
            megapose_vis=megapose_vis,
            megapose_vis_stages=megapose_vis_stages,
        )
        megapose_vis_image = self.MegaPose_Item.last_visualization

        if megapose_Rts_m is None or megapose_scores is None:
            return pred_Rts, None, np.array([], dtype=np.int64), megapose_vis_image

        refined_Rts = pred_Rts_np.astype(np.float32).copy()
        megapose_subset = megapose_Rts_m.astype(np.float32).copy()
        megapose_subset[:, :3, 3] *= 1000.0

        if megapose_subset.shape[0] == refined_Rts.shape[0]:
            valid_pose_ids = np.arange(refined_Rts.shape[0], dtype=np.int64)
            return megapose_subset, megapose_scores.astype(np.float32), valid_pose_ids, megapose_vis_image

        valid_pose_ids = np.asarray(valid_pose_ids, dtype=np.int64)
        if valid_pose_ids.shape[0] == megapose_subset.shape[0] and valid_pose_ids.size > 0 and np.max(valid_pose_ids) < refined_Rts.shape[0]:
            refined_Rts[valid_pose_ids] = megapose_subset
            return refined_Rts, megapose_scores.astype(np.float32), valid_pose_ids, megapose_vis_image

        return pred_Rts, None, np.array([], dtype=np.int64), megapose_vis_image

    @torch.inference_mode()
    def register_wapr(self):
        """Register public WAPR with BOP millimeter models; weights load on first use.
        用 BOP 毫米制模型注册公开 WAPR；首次非空推理时加载后端权重。
        """
        dataset = self.bop_dataset_item
        diameters_mm = [dataset.model_info[str(obj_id)]['diameter'] for obj_id in dataset.obj_id_list]
        wapr_item = Refinement_WAPR(device=str(self.device))
        wapr_item.register_objects(dataset.obj_id_list, dataset.obj_model_list, diameters_mm)
        self.WAPR_Item = wapr_item
        self.wapr_registered = True

    def _run_wapr_estimation(self, predictions, obj_ids, cam_K, image, depth_m,
                             wbps_threshold, nms_iou_threshold, mode='estimate'):
        """Estimate on the original frame, then filter every per-instance field together.
        在原图上估计，再同步筛选所有逐实例字段；保存原始编号、米制位姿及评分。
        """
        instances = []
        identifiers = []
        skipped = []
        for obj_id in obj_ids:
            if obj_id not in predictions:
                continue
            prediction = predictions[obj_id]
            count = len(prediction['pred_mask'])
            masks = restore_full_frame_masks(prediction, image.shape) if mode != 'refine' else [None] * count
            for index, mask in enumerate(masks):
                seed_m = None
                if mode != 'estimate':
                    seed_mm = prediction['Rts'][index]
                    if (prediction['pnp_success'][index] and np.isfinite(seed_mm).all()
                            and seed_mm[2, 3] > 0 and np.allclose(seed_mm[3], [0, 0, 0, 1])):
                        seed_m = seed_mm.astype(np.float32).copy()
                        seed_m[:3, 3] *= 1e-3
                    if mode == 'refine' and seed_m is None:
                        reason = 'pnp_failed' if not prediction['pnp_success'][index] else 'invalid_pnp_pose'
                        skipped.append({'object_id': int(obj_id), 'instance_index': index, 'status': reason})
                        continue
                if not self.wapr_registered:
                    self.register_wapr()
                item = {'mesh': self.WAPR_Item.meshes_by_id[obj_id],
                        'diameter_m': self.WAPR_Item.diameters_m[obj_id], 'mask': mask}
                if seed_m is not None:
                    item['pose_4x4_m'] = seed_m
                instances.append(item)
                identifiers.append({'object_id': int(obj_id), 'instance_index': index})
        outputs = self.WAPR_Item.inference_batch(cam_K, image, depth_m, instances, mode=mode) if instances else []
        records = []
        for identity, output, item in zip(identifiers, outputs, instances):
            record = {**identity, 'pose_4x4_m': np.asarray(output['pose_4x4']).tolist(),
                      'score_6d': float(output['score_6d']),
                      'hypothesis_source': output.get('hypothesis_source', 'wapr')}
            for name in ('hypothesis_index', 'hypothesis_count'):
                if name in output:
                    record[name] = int(output[name])
            if mode != 'estimate':
                record['pnp_success'] = bool(predictions[identity['object_id']]['pnp_success'][identity['instance_index']])
                if 'pose_4x4_m' in item:
                    record['pnp_pose_4x4_m'] = item['pose_4x4_m'].tolist()
            records.append(record)
        # Preserve raw estimates even when the following rendering/NMS step fails.
        # 即使后续渲染或 NMS 失败，也保留本帧原始估计，便于调用方单独保存。
        self.last_wapr_raw_records = records
        meshes = self.WAPR_Item.meshes_by_id if instances else {}
        report = self.postprocess_wapr(records, meshes, cam_K, image.shape, wbps_threshold,
                                       nms_iou_threshold, device=str(self.device))
        report['wapr_mode'] = mode
        report['skipped_instances'] = skipped
        for obj_id, prediction in predictions.items():
            rows = [row for row in report['retained_records'] if row['object_id'] == obj_id]
            kept_indices = np.asarray([row['instance_index'] for row in rows], dtype=np.int64)
            count = len(prediction['pred_mask'])
            # Runtime outputs are batch-first; scalar or shared metadata stays unchanged.
            # 网络输出均以实例为首维；标量或共享元数据保持不变。
            for name, value in list(prediction.items()):
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
                    indices = torch.as_tensor(kept_indices, device=value.device)
                    prediction[name] = value.index_select(0, indices)
                elif isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
                    prediction[name] = value[kept_indices]
            poses_m = np.asarray([row['pose_4x4_m'] for row in rows], dtype=np.float64).reshape(-1, 4, 4)
            poses_mm = poses_m.copy()
            poses_mm[:, :3, 3] *= 1000.0
            prediction['Rts'] = poses_mm
            prediction['wapr_scores'] = np.asarray([row['score_6d'] for row in rows], dtype=np.float64)
            prediction['wapr_instance_indices'] = kept_indices
            prediction['wapr_mode'] = mode
            prediction['wapr_pose_sources'] = [row['hypothesis_source'] for row in rows]
            prediction['wapr_raw_records'] = [row for row in records if row['object_id'] == obj_id]
            prediction['wapr_decisions'] = [row for row in report['decisions'] if row['object_id'] == obj_id]
            prediction['wapr_skipped_instances'] = [row for row in skipped if row['object_id'] == obj_id]
        return report

    @torch.inference_mode()
    def predict(self, cam_K, img, obj_ids, confidence_threshold=0.75, conf=0.75, iou=0.50, max_det=200, pnp_op='ransac+comb',
                depth=None, use_megapose=False, megapose_vis=False, megapose_vis_stages=None,
                use_wapr=False, wbps_threshold=0.75, wapr_nms_iou_threshold=0.5, wapr_mode='estimate'):
        """Estimate object-to-camera poses from an OpenCV BGR image and original pixel intrinsics.

        Depth uses meters. WAPR modes are estimate (no PnP), hybrid (PnP plus WAPR
        candidates), and refine (PnP seeds only). All apply WBPS and rendered-mask NMS.
        深度使用米。WAPR 支持 estimate（不用 PnP）、hybrid（PnP 与 WAPR 候选混合）
        和 refine（仅精修 PnP 初值）；统一执行 WBPS 筛选与渲染掩码 NMS，Rts 平移仍为毫米。
        """
        if wapr_mode not in ('estimate', 'hybrid', 'refine'):
            raise ValueError('WAPR mode must be estimate, hybrid or refine')
        if not use_wapr and wapr_mode != 'estimate':
            raise ValueError('Set use_wapr=True to select a WAPR mode')
        run_pnp = not use_wapr or wapr_mode != 'estimate'
        if use_wapr and use_megapose:
            raise ValueError('Choose either WAPR mask estimation or MegaPose refinement for one predict call')
        if use_wapr:
            self.last_wapr_raw_records = None
            if not np.isfinite(wbps_threshold) or not np.isfinite(wapr_nms_iou_threshold) or not 0 <= wapr_nms_iou_threshold <= 1:
                raise ValueError('Use a finite WBPS threshold and a WAPR NMS IoU threshold in [0,1]')
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError('WAPR requires a uint8 BGR image')
            if depth is None or np.shape(depth) != img.shape[:2]:
                raise ValueError('WAPR requires registered depth matching the original image in meters')
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise ValueError('WAPR depth must be finite, nonnegative meters')
            if np.shape(cam_K) != (3, 3) or not np.isfinite(cam_K).all() or cam_K[0, 0] <= 0 or cam_K[1, 1] <= 0:
                raise ValueError('WAPR requires finite original-image pixel intrinsics')
            obj_ids = list(obj_ids)
            if (any(isinstance(obj_id, bool) or not isinstance(obj_id, Integral) for obj_id in obj_ids)
                    or len(set(obj_ids)) != len(obj_ids)
                    or any(obj_id not in self.bop_dataset_item.obj_id_list for obj_id in obj_ids)):
                raise ValueError('WAPR object IDs must be unique and registered in the dataset')
            original_image = img.copy()
            original_depth_m = np.asarray(depth, dtype=np.float32).copy()
            original_cam_K = cam_K.copy()
        if use_megapose and not self.megapose_registered:
            self.register_megapose()

        pnp_op_l = [['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb'],[0,2,1]]
        # Combined PnP requires the object prior; single-solver modes can use older
        # checkpoints without it. Report the asset before running the detector.
        if run_pnp and pnp_op not in ('epnp', 'ransac', 'ransac+vvs'):
            for obj_id in obj_ids:
                if obj_id not in self.HccePose_Item_info:
                    continue
                info = self.HccePose_Item_info[obj_id]
                prior = info['keypoints_']
                if prior is None or np.asarray(prior).size == 0:
                    raise ValueError(
                        f"Combined PnP requires keypoints_ for object {obj_id} "
                        f"in checkpoint {info['checkpoint_path']}"
                    )

        depth_m = None if depth is None else depth.astype(np.float32, copy=False)
        height, width = img.shape[:2]
        ratio, scaled_h, scaled_w, padded_h, padded_w, top, left = inference_image_geometry(height, width)
        cam_K = cam_K.copy()
        if ratio > 1:
            img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            if depth_m is not None:
                depth_m = cv2.resize(depth_m, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            cam_K[:2, :] /= ratio
        bottom = padded_h - scaled_h - top
        right = padded_w - scaled_w - left
        if padded_h != scaled_h or padded_w != scaled_w:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if depth_m is not None:
                depth_m = cv2.copyMakeBorder(depth_m, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        cam_K[0, 2] += left
        cam_K[1, 2] += top

        img_torch = torch.from_numpy(img.astype(np.float32)).to(self.device)
        
        img_torch = img_torch[None]

        with torch.amp.autocast('cuda'):
            t1 = time.time()
            stage_times = {
                'yolo': 0.0,
                'hccepose': 0.0,
                'megapose': 0.0,
                'wapr': 0.0,
                'visualization': 0.0,
            }
            
            det_results = {}
            results_dict = {}
            
            t_stage = time.time()
            scaled_img_torch = kornia.geometry.transform.resize(img_torch.permute(0,3,1,2), (int(img_torch.shape[1] * 1.0), int(img_torch.shape[2] * 1.0)), interpolation='bilinear')
            # Tensor inputs bypass Ultralytics' BGR-to-RGB conversion; keep HCCE crops in BGR.
            # Tensor 输入不会经过 Ultralytics 的 BGR→RGB 转换；HCCE 裁剪仍保持 BGR。
            detector_rgb = scaled_img_torch[:, [2, 1, 0]].contiguous()
            det_yolo = self.model_yolo(detector_rgb.clamp(0,255)/255,
                                       conf=conf, iou=iou, max_det=max_det, 
                                    #    imgsz=640 * 1
                                       )
            xywh = det_yolo[0].boxes.xywh.clone().detach()
            cls = det_yolo[0].boxes.cls.clone().detach()
            xywh[:, 0] -= xywh[:, 2] / 2 # left, top, width, height
            xywh[:, 1] -= xywh[:, 3] / 2 # left, top, width, height
            det_results['xywh'] = xywh
            det_results['confs'] = confs = det_yolo[0].boxes.conf.clone().detach()
            det_results['cls'] = cls
            stage_times['yolo'] += time.time() - t_stage
            
            

            for key_i in self.bop_dataset_item.obj_id_list:
                if key_i in obj_ids:
                    obj_id = key_i
                    Rt_list = []
                    padding_ratio = 1.5
                    obj_index = self.bop_dataset_item.obj_id_list.index(obj_id)
                    obj_det_ids = torch.nonzero(cls == obj_index, as_tuple=False).reshape(-1)
                    xywh_s = xywh[obj_det_ids]
                    conf_s = confs[obj_det_ids].clone()
                    if xywh_s.shape[0] > 0:
                        t_stage = time.time()
                        crop_size = self.crop_size
                        Detect_Bbox_tfs, _ = crop_trans_batch_hccepose(xywh_s, out_size=[crop_size,crop_size], padding_ratio=padding_ratio)
                        Detect_Bbox_tfs_128, boxes_128 = crop_trans_batch_hccepose(xywh_s, out_size=[int(crop_size/2),int(crop_size/2)], padding_ratio=padding_ratio)
                        img_torch_hccepose = composed_transforms_img(img_torch.permute(0, 3, 1, 2) / 255.0)
                        crop_rgbs = kornia.geometry.transform.warp_perspective(img_torch_hccepose.repeat(xywh_s.shape[0],1,1,1), Detect_Bbox_tfs, dsize=[crop_size,crop_size], mode='bilinear', align_corners=False)
                        pred_results = self.HccePose_Runtime_Item[obj_id].inference_batch(crop_rgbs, boxes_128)
                        
                        
                        pred_results['Detect_Bbox_tfs_128'] = Detect_Bbox_tfs_128
                        pred_results['conf'] = conf_s
                        pred_results['crop_rgbs'] = crop_rgbs
                        pred_mask = pred_results['pred_mask']
                        coord_image = pred_results['coord_2d_image']
                        pred_front_code_0 = pred_results['pred_front_code_obj']
                        pred_back_code_0 = pred_results['pred_back_code_obj']
                        pred_front_code = pred_results['pred_front_code']
                        pred_back_code = pred_results['pred_back_code']
                        
                        pred_mask_np = pred_mask.detach().cpu().numpy()
                        pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
                        pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
                        results = []
                        coord_image_np = coord_image.detach().cpu().numpy()
                        
                        if not run_pnp:
                            # Only the masks are needed; do not fabricate PnP success flags.
                            # WAPR 只需要掩码，不计算或伪造 PnP 成功标记。
                            pass
                        elif pnp_op in ['epnp', 'ransac', 'ransac+vvs']:
                            pred_m_f_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], coord_image_np[i], cam_K) for i in range(pred_mask_np.shape[0])]
                            for pred_m_f_c_np_i in pred_m_f_c_np:
                                result_i = solve_PnP(pred_m_f_c_np_i, pnp_op=pnp_op_l[1][pnp_op_l[0].index(pnp_op)])
                                results.append(result_i)
                                Rt_i = np.eye(4)
                                Rt_i[:3, :3] = result_i['rot']
                                Rt_i[:3, 3:] = result_i['tvecs']
                                Rt_list.append(Rt_i)
                        else:
                            pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K) for i in range(pred_mask_np.shape[0])]
                            for pred_m_bf_c_np_i in pred_m_bf_c_np:
                                if pnp_op == 'ransac+comb':
                                    pnp_op_0 = 2
                                else:
                                    pnp_op_0 = 1
                                result_i = solve_PnP_comb(pred_m_bf_c_np_i, self.HccePose_Item_info[obj_id]['keypoints_'], pnp_op=pnp_op_0)
                                results.append(result_i)
                                Rt_i = np.eye(4)
                                Rt_i[:3, :3] = result_i['rot']
                                Rt_i[:3, 3:] = result_i['tvecs']
                                Rt_list.append(Rt_i)

                        if run_pnp:
                            pred_results['Rts'] = np.array(Rt_list)
                            pred_results['pnp_success'] = np.asarray([result['success'] for result in results], dtype=bool)
                            if use_wapr:
                                pred_results['Rts_hccepose'] = pred_results['Rts'].copy()
                        stage_times['hccepose'] += time.time() - t_stage

                        if use_megapose:
                            hccepose_Rts = pred_results['Rts'].astype(np.float32).copy()
                            t_stage = time.time()
                            megapose_Rts, megapose_scores, megapose_valid_pose_ids, megapose_vis_image = self._run_megapose_refinement(
                                obj_id,
                                cam_K,
                                img,
                                depth_m,
                                hccepose_Rts,
                                megapose_vis=megapose_vis,
                                megapose_vis_stages=megapose_vis_stages,
                            )
                            if megapose_vis_image is not None:
                                pred_results['megapose_vis'] = megapose_vis_image
                            if megapose_scores is not None:
                                pred_results['Rts_hccepose'] = hccepose_Rts
                                pred_results['Rts'] = megapose_Rts
                                pred_results['megapose_scores'] = megapose_scores
                                pred_results['megapose_valid_pose_ids'] = megapose_valid_pose_ids
                            stage_times['megapose'] += time.time() - t_stage

                        results_dict[obj_id] = pred_results
            
            if use_wapr:
                t_stage = time.time()
                # Preserve the public backend's precision policy outside HccePose AMP.
                # WAPR 不继承 HccePose 的混合精度上下文，保持公开后端自身的精度策略。
                with torch.amp.autocast('cuda', enabled=False):
                    wapr_report = self._run_wapr_estimation(
                        results_dict, obj_ids, original_cam_K, original_image, original_depth_m,
                        wbps_threshold, wapr_nms_iou_threshold, mode=wapr_mode)
                stage_times['wapr'] += time.time() - t_stage
            t2 = time.time()
            
            t_stage = time.time()
            if self.show_op:
                draw_image = draw_annotations_on_image_yolo(img_torch.clone().detach().cpu().numpy().astype(np.uint8)[0], 
                                                    det_results['xywh'].clone().detach().cpu().numpy().astype(np.int32), 
                                                    det_results['confs'].clone().detach().cpu().numpy().astype(np.float32),
                                                    det_results['cls'].clone().detach().cpu().numpy().astype(np.int32),
                                                    self.bop_dataset_item.obj_id_list,
                                                    confidence_threshold = confidence_threshold,
                                                    )
                
            if self.show_op:
                pred_front_code_l = []
                pred_back_code_l = []
                crop_rgbs_l = []
                pred_mask_l = []
                Detect_Bbox_tfs_128_l = []
                obj_ids_l = []
                Rts_l = []
                conf_s_l = []
                for obj_id in results_dict:
                    pred_results = results_dict[obj_id]
                    if len(pred_results['Rts']) == 0:
                        continue
                    pred_front_code_raw = pred_results['pred_front_code_raw'].reshape((-1,128,128,3,8)).permute((0,1,2,4,3)).reshape((-1,128,128,24))
                    pred_back_code_raw = pred_results['pred_back_code_raw'].reshape((-1,128,128,3,8)).permute((0,1,2,4,3)).reshape((-1,128,128,24))
                    pred_front_code_l.append(torch.cat([pred_results['pred_front_code'], pred_front_code_raw], dim=-1))
                    pred_back_code_l.append(torch.cat([pred_results['pred_back_code'], pred_back_code_raw], dim=-1))
                    crop_rgbs_l.append(pred_results['crop_rgbs'])
                    pred_mask_l.append(pred_results['pred_mask'])
                    Detect_Bbox_tfs_128_l.append(pred_results['Detect_Bbox_tfs_128'])
                    obj_ids_l.append(np.ones((pred_results['pred_mask'].shape[0])) * obj_id)
                    Rts_l.append(pred_results['Rts'])
                    conf_s_l.append(pred_results['conf'])
                    
                if len(crop_rgbs_l) == 0:
                    h_i, w_i = int(img_torch.shape[1]), int(img_torch.shape[2])
                    vis0 = np.zeros((h_i, w_i, 3), dtype=np.uint8)
                    vis1 = vis0.copy()
                    vis2 = vis0.copy()
                else:
                    crop_rgbs = torch.cat(crop_rgbs_l, dim = 0)
                    pred_mask = torch.cat(pred_mask_l, dim = 0)
                    pred_front_code = torch.cat(pred_front_code_l, dim = 0)
                    pred_back_code = torch.cat(pred_back_code_l, dim = 0)
                    Detect_Bbox_tfs_128 = torch.cat(Detect_Bbox_tfs_128_l, dim = 0)
                    conf_s = torch.cat(conf_s_l, dim = 0)
                    obj_ids_l = np.concatenate(obj_ids_l, axis = 0)
                    Rts_l = np.concatenate(Rts_l, axis = 0)
                    
                    vis0 = vis_rgb_mask_Coord(crop_rgbs, pred_mask, pred_front_code, pred_back_code)
                    
                    pred_mask_origin = kornia.geometry.transform.warp_perspective(pred_mask[:,None,...], 
                                                                            torch.linalg.inv(Detect_Bbox_tfs_128.to(torch.float32)), 
                                                                            dsize=[img_torch[0].shape[0],img_torch[0].shape[1]], mode='nearest', align_corners=False)
                    pred_front_code_origin = kornia.geometry.transform.warp_perspective(pred_front_code.permute(0,3,1,2), 
                                                                            torch.linalg.inv(Detect_Bbox_tfs_128.to(torch.float32)), 
                                                                            dsize=[img_torch[0].shape[0],img_torch[0].shape[1]], mode='nearest', align_corners=False)
                    pred_back_code_origin = kornia.geometry.transform.warp_perspective(pred_back_code.permute(0,3,1,2), 
                                                                            torch.linalg.inv(Detect_Bbox_tfs_128.to(torch.float32)), 
                                                                            dsize=[img_torch[0].shape[0],img_torch[0].shape[1]], mode='nearest', align_corners=False)
                    vis1, vis2 = vis_rgb_mask_Coord_origin(cam_K, obj_ids_l, self.bop_dataset_item.obj_id_list, self.BBox_3d, Rts_l, conf_s, 
                                              img_torch_hccepose, pred_mask_origin, 
                                              pred_front_code_origin, pred_back_code_origin)
            if megapose_vis:
                panels = []
                for obj_id, entry in results_dict.items():
                    image = entry.get('megapose_vis')
                    if image is not None:
                        panels.append(_megapose_add_title(image, f'obj_{int(obj_id):06d}', height=26))
                if panels:
                    results_dict['show_megapose'] = _megapose_vstack_images(panels)
            stage_times['visualization'] += time.time() - t_stage
            if use_wapr:
                results_dict['wapr_postprocess'] = wapr_report
            results_dict['time'] = t2 - t1
            other_time = max(0.0, results_dict['time'] - sum(stage_times.values()))
            results_dict['time_dict'] = {
                'yolo': stage_times['yolo'],
                'hccepose': stage_times['hccepose'],
                'megapose': stage_times['megapose'],
                'wapr': stage_times['wapr'],
                'visualization': stage_times['visualization'],
                'other': other_time,
            }
            if self.show_op:
                results_dict['show_2D_results'] = draw_image
                results_dict['show_6D_vis0'] = vis0
                results_dict['show_6D_vis1'] = vis1
                results_dict['show_6D_vis2'] = vis2

        return results_dict
