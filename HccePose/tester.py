# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

"""HccePose RGB inference and optional MegaPose refinement."""

import os, torch, kornia, time, cv2, random
from pathlib import Path
import numpy as np 
from ultralytics import YOLO
from HccePose.bop_loader import bop_dataset, IMAGENET_MEAN_BGR, IMAGENET_STD_BGR
from HccePose.network_model import HccePose_BF_Net
from HccePose.network_model import load_checkpoint
from HccePose.image_geometry import inference_image_geometry
import torchvision.transforms as transforms
from HccePose.visualization import vis_rgb_mask_Coord, vis_rgb_mask_Coord_origin
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb

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
    """Detect known objects, estimate their poses with PnP and optionally run MegaPose."""

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
        if acceleration is not None:
            hccepose_acceleration = acceleration
        if acceleration_cache_dir is not None:
            hccepose_acceleration_cache_dir = acceleration_cache_dir
        self.acceleration = 'pytorch' if hccepose_acceleration is None else str(hccepose_acceleration).lower()
        if self.acceleration not in ['pytorch', 'onnx', 'tensorrt']:
            raise ValueError('Unsupported HccePose acceleration backend: %s' % hccepose_acceleration)
        self.acceleration_cache_dir = hccepose_acceleration_cache_dir

        _repo_root = str(Path(__file__).resolve().parents[1])
        from HccePose.ensure_rgbd_megapose_demo import ensure_acceleration_backend_environment

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
                from HccePose.hccepose_acceleration import HccePoseOnnxRunner
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
        from Refinement.refinement_group import Refinement_MP

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
    def predict(self, cam_K, img, obj_ids, confidence_threshold=0.75, conf=0.75, iou=0.50, max_det=200, pnp_op='ransac+comb',
                depth=None, use_megapose=False, megapose_vis=False, megapose_vis_stages=None):
        """Estimate object-to-camera poses from an OpenCV BGR image and original pixel intrinsics.

        Depth, when supplied for MegaPose, is in meters. Each object entry contains
        Rts (translations in millimeters) and a pnp_success flag per detection.
        The flag describes the HccePose solver before optional MegaPose refinement.
        """
        if use_megapose and not self.megapose_registered:
            self.register_megapose()

        pnp_op_l = [['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb'],[0,2,1]]
        # Combined PnP requires the object prior; single-solver modes can use older
        # checkpoints without it. Report the asset before running the detector.
        if pnp_op not in ('epnp', 'ransac', 'ransac+vvs'):
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
                        
                        if pnp_op in ['epnp', 'ransac', 'ransac+vvs']:
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

                        pred_results['Rts'] = np.array(Rt_list)
                        pred_results['pnp_success'] = np.asarray([result['success'] for result in results], dtype=bool)
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
                from Refinement.MegaPose import _megapose_add_title, _megapose_vstack_images

                panels = []
                for obj_id, entry in results_dict.items():
                    image = entry.get('megapose_vis')
                    if image is not None:
                        panels.append(_megapose_add_title(image, f'obj_{int(obj_id):06d}', height=26))
                if panels:
                    results_dict['show_megapose'] = _megapose_vstack_images(panels)
            stage_times['visualization'] += time.time() - t_stage
            results_dict['time'] = t2 - t1
            other_time = max(0.0, results_dict['time'] - sum(stage_times.values()))
            results_dict['time_dict'] = {
                'yolo': stage_times['yolo'],
                'hccepose': stage_times['hccepose'],
                'megapose': stage_times['megapose'],
                'visualization': stage_times['visualization'],
                'other': other_time,
            }
            if self.show_op:
                results_dict['show_2D_results'] = draw_image
                results_dict['show_6D_vis0'] = vis0
                results_dict['show_6D_vis1'] = vis1
                results_dict['show_6D_vis2'] = vis2

        return results_dict
