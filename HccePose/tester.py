# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''Tester utilities for HccePose and optional refinement backends.

---
---
The file keeps HccePose as the primary inference pipeline and attaches
FoundationPose / MegaPose as optional refinement branches. Detection, crop
generation, main-network inference, PnP solving, result packaging and
result collection are all organized here, while refinement backends are only
plugged in as optional post-processing steps.
---
---
本文件封装 HccePose 的主推理入口，并在不改变主流程接口的前提下，
提供 FoundationPose 与 MegaPose 的可选微调接入。整体设计仍以 HccePose
为中心：检测、裁剪、主网络推理、PnP 与结果组织都在这里完成，而微调器
只作为可选后处理分支插入。
'''

import os, torch, kornia, time, cv2, random
from pathlib import Path
import numpy as np 
from ultralytics import YOLO
from HccePose.bop_loader import bop_dataset, IMAGENET_MEAN_BGR, IMAGENET_STD_BGR
from HccePose.network_model import HccePose_BF_Net
from HccePose.network_model import load_checkpoint, get_checkpoint
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

def _foundationpose_add_title(image, title, height=26, bg_value=18):
    '''
    ---
    ---
    Attach a compact title bar above a visualization tile.
    ---
    ---
    This helper adds a small title header to a visualization tile before the
    tile is stacked into a larger refinement canvas. It is layout-only and
    does not participate in pose computation.

    Args:
        - image: BGR uint8 image tile (same layout as ``cv2.imwrite`` / FoundationPose panels).
        - title: Short text rendered on the header.
        - height: Header height in pixels.
        - bg_value: Gray fill value for the header background.

    Returns:
        - Image with header concatenated on top.
    ---
    ---
    给单张可视化图块上方添加简短标题栏。
    ---
    ---
    用于 FoundationPose / MegaPose 阶段图纵向拼接前的排版，不参与位姿计算。

    参数:
        - image: BGR uint8 图块（与 ``cv2.imwrite`` / FoundationPose 拼图一致）。
        - title: 标题栏上绘制的文字。
        - height: 标题栏高度（像素）。
        - bg_value: 标题栏背景灰度值。

    返回:
        - 顶部拼接标题后的图像。
    '''
    header = np.full((height, image.shape[1], 3), bg_value, dtype=np.uint8)
    cv2.putText(header, str(title), (8, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _foundationpose_stack_named_images(named_images, bg_value=18):
    '''
    ---
    ---
    Stack multiple named visualization tiles vertically.
    ---
    ---
    This utility stacks multiple titled image tiles into a single overview
    canvas. It is mainly used to collect FoundationPose or MegaPose per-object
    visualizations into one frame, while safely skipping missing tiles.

    Args:
        - named_images: List of `(title, image)` pairs; `image` may be None.
        - bg_value: Padding/header background constant.

    Returns:
        - Stitched canvas or None if all inputs were skipped.
    ---
    ---
    将若干带标题图块纵向拼接成一张总览图。
    ---
    ---
    用于 FoundationPose / MegaPose 对象级可视化汇总；空图会被跳过。

    参数:
        - named_images: `(标题, 图像)` 列表，图像可为 None。
        - bg_value: 填充与标题背景的灰度常数。

    返回:
        - 拼接后的大图；若全部跳过则返回 None。
    '''
    titled_images = []
    for title, image in named_images:
        if image is None:
            continue
        titled_images.append(_foundationpose_add_title(image, title, bg_value=bg_value))
    if len(titled_images) == 0:
        return None
    max_w = max(image.shape[1] for image in titled_images)
    spacer = np.full((8, max_w, 3), bg_value, dtype=np.uint8)
    canvas = None
    for image in titled_images:
        pad_w = max_w - image.shape[1]
        image_padded = cv2.copyMakeBorder(image, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(bg_value, bg_value, bg_value))
        if canvas is None:
            canvas = image_padded
        else:
            canvas = np.concatenate([canvas, spacer, image_padded], axis=0)
    return canvas

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

class Tester():
    '''
    ---
    ---
    Main HccePose tester with optional refinement backends.
    ---
    ---
    The Tester class orchestrates the full per-frame inference pipeline:
    YOLO detection, HccePose network inference, PnP solving, result packaging,
    and optional FoundationPose / MegaPose refinement. It serves as the common
    entry point for the project test scripts, so interface stability is favored.
    ---
    ---
    串联单帧推理：YOLO、HccePose、PnP、结果整理及可选 FoundationPose /
    MegaPose 微调；作为测试脚本统一入口，强调接口稳定。
    '''
    
    def __init__(self, bop_dataset_item : bop_dataset, show_op = True, hccepose_vis=None, CUDA_DEVICE='0', top_K = None, crop_size = 256, efficientnet_key=None,
                 foundationpose_refine_dir=None, foundationpose_score_dir=None,
                 megapose_model_name_rgb='megapose-1.0-RGB-multi-hypothesis',
                 megapose_model_name_rgbd='megapose-1.0-RGBD',
                 hccepose_acceleration='pytorch', hccepose_acceleration_cache_dir=None,
                 foundationpose_acceleration='pytorch', foundationpose_acceleration_cache_dir=None,
                 acceleration=None, acceleration_cache_dir=None):
        '''
        ---
        ---
        Initialize detector, HccePose networks and optional refinement state.
        ---
        ---
        Selects device, loads YOLO, restores one HccePose checkpoint per object,
        optionally wraps ONNX/TensorRT runners, and registers FoundationPose when
        paths are provided.

        Args:
            - bop_dataset_item: Loaded `bop_dataset` with models and paths.
            - show_op / hccepose_vis: Enable visualization flags.
            - CUDA_DEVICE: GPU index string.
            - top_K, crop_size, efficientnet_key: Inference hyper-parameters.
            - foundationpose_* / megapose_*: Optional refinement backends.
            - hccepose_acceleration / foundationpose_acceleration: Backend keys.
            - acceleration / acceleration_cache_dir: Aliases forwarded to HccePose.

        Returns:
            - None.
        ---
        ---
        初始化检测器、各物体 HccePose 网络及可选微调与加速状态。
        ---
        ---
        完成设备选择、YOLO 加载、逐物体 checkpoint 加载，可选 ONNX/TensorRT，
        以及在给定路径下注册 FoundationPose。

        参数:
            - bop_dataset_item: 已加载的 `bop_dataset`。
            - show_op / hccepose_vis: 是否开启可视化。
            - CUDA_DEVICE: GPU 编号字符串。
            - top_K, crop_size, efficientnet_key: 推理相关超参。
            - foundationpose_* / megapose_*: 可选微调后端配置。
            - hccepose_acceleration / foundationpose_acceleration: 加速后端名称。
            - acceleration / acceleration_cache_dir: 传给 HccePose 的别名参数。

        返回:
            - 无。
        '''
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

        self.foundationpose_registered = False
        self.foundationpose_pose_unit_scale = 1e-3
        self.FoundationPose_Item = None
        self.FoundationPose_obj_id_to_index = {}
        self.megapose_registered = False
        self.MegaPose_Item = None
        if acceleration is not None:
            hccepose_acceleration = acceleration
        if acceleration_cache_dir is not None:
            hccepose_acceleration_cache_dir = acceleration_cache_dir
        self.acceleration = 'pytorch' if hccepose_acceleration is None else str(hccepose_acceleration).lower()
        self.foundationpose_acceleration = 'pytorch' if foundationpose_acceleration is None else str(foundationpose_acceleration).lower()
        if self.acceleration not in ['pytorch', 'onnx', 'tensorrt']:
            raise ValueError('Unsupported HccePose acceleration backend: %s' % hccepose_acceleration)
        if self.foundationpose_acceleration not in ['pytorch', 'onnx', 'tensorrt']:
            raise ValueError('Unsupported FoundationPose acceleration backend: %s' % foundationpose_acceleration)
        self.acceleration_cache_dir = hccepose_acceleration_cache_dir
        self.foundationpose_acceleration_cache_dir = foundationpose_acceleration_cache_dir

        _repo_root = str(Path(__file__).resolve().parents[1])
        from HccePose.ensure_rgbd_megapose_demo import ensure_acceleration_backend_environment

        ensure_acceleration_backend_environment(
            _repo_root,
            self.acceleration,
            self.foundationpose_acceleration,
        )

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
            checkpoint_path_i = get_checkpoint(best_save_path)
            info_i = load_checkpoint(best_save_path, HccePose_BF_Net_i, CUDA_DEVICE=CUDA_DEVICE)
            self.HccePose_Item[obj_id] = HccePose_BF_Net_i
            self.HccePose_Item_info[obj_id] = info_i
            if self.acceleration in ['onnx', 'tensorrt']:
                from HccePose.hccepose_acceleration import HccePoseOnnxRunner
                cache_dir_i = self.acceleration_cache_dir
                if cache_dir_i is None:
                    cache_dir_i = os.path.join(bop_dataset_item.dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'), 'onnx_cache')
                self.HccePose_Runtime_Item[obj_id] = HccePoseOnnxRunner(
                    HccePose_BF_Net_i,
                    checkpoint_path_i,
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

        if foundationpose_refine_dir is not None and foundationpose_score_dir is not None:
            self.register_foundationpose(foundationpose_refine_dir, foundationpose_score_dir)

        pass

    def _resolve_foundationpose_paths(self, foundationpose_path):
        '''
        ---
        ---
        Resolve FoundationPose config/checkpoint paths from a user input path.
        ---
        ---
        Accepts either a run directory or a direct checkpoint file path, validates
        existence, and returns `(config_path, checkpoint_path)`.

        Args:
            - foundationpose_path: Directory or `.pth` file path.

        Returns:
            - config_path, checkpoint_path: Absolute resolved paths.
        ---
        ---
        兼容目录或单文件 checkpoint 输入，统一返回 `(config_path, checkpoint_path)`。

        参数:
            - foundationpose_path: 运行目录或 `.pth` 权重路径。

        返回:
            - config_path, checkpoint_path: 校验存在后的绝对路径对。
        '''
        if os.path.isdir(foundationpose_path):
            config_path = os.path.join(foundationpose_path, 'config.yml')
            checkpoint_path = os.path.join(foundationpose_path, 'model_best.pth')
        else:
            checkpoint_path = foundationpose_path
            config_path = os.path.join(os.path.dirname(foundationpose_path), 'config.yml')
        if not os.path.exists(config_path):
            raise FileNotFoundError('FoundationPose config is not found: %s' % config_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError('FoundationPose checkpoint is not found: %s' % checkpoint_path)
        return config_path, checkpoint_path

    def _load_foundationpose_checkpoint(self, model, checkpoint_path):
        '''
        ---
        ---
        Load a FoundationPose checkpoint into an already-created model.
        ---
        ---
        Unwraps optional `model` / `model_state_dict` keys, loads weights, moves
        the module to `self.device`, and sets `eval()`.

        Args:
            - model: Target torch module.
            - checkpoint_path: Path to the saved weights.

        Returns:
            - None.
        ---
        ---
        兼容裸 `state_dict` 或 `model` / `model_state_dict` 包装；加载后
        将模型移到设备并 `eval()`。

        参数:
            - model: 目标 PyTorch 模块。
            - checkpoint_path: 权重文件路径。

        返回:
            - 无。
        '''
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint)
        model.to(self.device).eval()

    def _prepare_foundationpose_objects(self, foundationpose_item):
        '''
        ---
        ---
        Prepare per-object meshes and mesh tensors for FoundationPose.
        ---
        ---
        Loads BOP meshes, converts millimeters to meters, centers vertices,
        builds mesh tensors, records diameters, and fills `FoundationPose_obj_id_to_index`.

        Args:
            - foundationpose_item: `Refinement_FP` container lists to populate.

        Returns:
            - None.
        ---
        ---
        为每个物体准备居中 mesh、mesh tensor、中心、直径及 id 索引；BOP 毫米
        转米以匹配 FoundationPose。

        参数:
            - foundationpose_item: 待填充的 `Refinement_FP` 容器。

        返回:
            - 无。
        '''
        import trimesh
        from Refinement.foundationpose import make_mesh_tensors

        foundationpose_item.model_center = []
        foundationpose_item.mesh = []
        foundationpose_item.mesh_tensors = []
        foundationpose_item.diameter = []

        for obj_index, obj_id in enumerate(self.bop_dataset_item.obj_id_list):
            model_path = self.bop_dataset_item.obj_model_list[obj_index]
            mesh = trimesh.load(model_path, force='mesh', process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            mesh = mesh.copy()
            mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * self.foundationpose_pose_unit_scale
            _ = mesh.vertex_normals

            max_xyz = mesh.vertices.max(axis=0)
            min_xyz = mesh.vertices.min(axis=0)
            model_center = (min_xyz + max_xyz) / 2.0
            mesh_centered = mesh.copy()
            mesh_centered.vertices = mesh_centered.vertices - model_center.reshape(1, 3)

            foundationpose_item.model_center.append(model_center.astype(np.float32))
            foundationpose_item.mesh.append(mesh_centered)
            foundationpose_item.mesh_tensors.append(make_mesh_tensors(mesh_centered, device=str(self.device)))
            foundationpose_item.diameter.append(float(self.bop_dataset_item.model_info[str(obj_id)]['diameter']) * self.foundationpose_pose_unit_scale)
            self.FoundationPose_obj_id_to_index[obj_id] = obj_index

    def register_foundationpose(self, foundationpose_refine_dir, foundationpose_score_dir):
        '''
        ---
        ---
        Register FoundationPose refinement and score networks.
        ---
        ---
        Loads YAML configs, checkpoints, constructs `Refinement_FP`, applies
        crop ratios / normalizers / acceleration, and prepares meshes.

        Args:
            - foundationpose_refine_dir: Path to refine model assets.
            - foundationpose_score_dir: Path to score model assets.

        Returns:
            - None.
        ---
        ---
        读取 refine/score 配置与权重，创建 `Refinement_FP`，设置裁剪比、
        归一化与加速，并准备全部物体 mesh。

        参数:
            - foundationpose_refine_dir: refine 模型资源路径。
            - foundationpose_score_dir: score 模型资源路径。

        返回:
            - 无。
        '''
        import yaml
        from Refinement.refinement_group import Refinement_FP

        if not torch.cuda.is_available():
            raise RuntimeError('FoundationPose registration requires CUDA in the current tester integration.')

        refine_config_path, refine_checkpoint_path = self._resolve_foundationpose_paths(foundationpose_refine_dir)
        score_config_path, score_checkpoint_path = self._resolve_foundationpose_paths(foundationpose_score_dir)

        with open(refine_config_path, 'r', encoding='utf-8') as f:
            refine_config = yaml.safe_load(f)
        with open(score_config_path, 'r', encoding='utf-8') as f:
            score_config = yaml.safe_load(f)

        foundationpose_item = Refinement_FP(device=str(self.device))

        trans_normalizer = torch.as_tensor([1.0, 1.0, 1.0],
                                           device=self.device, dtype=torch.float32).reshape(1, 3)
        rot_normalizer = float(refine_config.get('rot_normalizer', foundationpose_item.refiner.rot_normalizer))
        refine_crop_ratio = float(refine_config.get('crop_ratio', foundationpose_item.refiner.crop_ratio))
        score_crop_ratio = float(score_config.get('crop_ratio', foundationpose_item.scorer.crop_ratio))

        foundationpose_item.refiner.trans_normalizer = trans_normalizer.clone()
        foundationpose_item.refiner.rot_normalizer = rot_normalizer
        foundationpose_item.refiner.crop_ratio = refine_crop_ratio
        foundationpose_item.scorer.crop_ratio = score_crop_ratio

        self._load_foundationpose_checkpoint(foundationpose_item.refiner.model, refine_checkpoint_path)
        self._load_foundationpose_checkpoint(foundationpose_item.scorer.model, score_checkpoint_path)
        foundationpose_item.configure_acceleration(
            acceleration=self.foundationpose_acceleration,
            refine_checkpoint_path=refine_checkpoint_path,
            score_checkpoint_path=score_checkpoint_path,
            cache_dir=self.foundationpose_acceleration_cache_dir,
        )
        self._prepare_foundationpose_objects(foundationpose_item)

        self.FoundationPose_Item = foundationpose_item
        self.foundationpose_registered = True
        print('FoundationPose is registered.')

    def _run_foundationpose_refinement(self, obj_id, cam_K, img, depth, pred_mask_origin, pred_Rts,
                                        foundationpose_vis=False, foundationpose_vis_stages=None):
        '''
        ---
        ---
        Run FoundationPose refinement on HccePose pose hypotheses.
        ---
        ---
        Filters poses with insufficient depth overlap, scales translations to
        meters for FoundationPose, runs `inference_batch`, then scales back.

        Args:
            - obj_id: Target object id.
            - cam_K: Camera intrinsics.
            - img, depth: OpenCV BGR uint8 image and depth in meters.
            - pred_mask_origin: Projected masks for each hypothesis.
            - pred_Rts: Initial 4x4 poses (millimeters in translation).
            - foundationpose_vis, foundationpose_vis_stages: Debug visualization flags.

        Returns:
            - Tuple of refined poses, scores, valid pose indices, vis image (or placeholders).
        ---
        ---
        在 HccePose 假设与掩码上运行 FoundationPose；先按深度支持筛选候选，
        单位换算后微调，再回填位姿与可选可视化。

        参数:
            - obj_id: 物体 id。
            - cam_K: 相机内参。
            - img, depth: OpenCV BGR uint8 图像与深度（米）。
            - pred_mask_origin: 各假设的回投掩码。
            - pred_Rts: 初始 4x4 位姿（平移为毫米）。
            - foundationpose_vis, foundationpose_vis_stages: 可视化选项。

        返回:
            - 微调后位姿、分数、有效索引、可视化图等元组（失败时含占位）。
        '''
        pred_mask_binary = (pred_mask_origin > 0.5).astype(np.float32)
        valid_pose_ids = []
        for pose_id in range(pred_mask_binary.shape[0]):
            valid_support = (pred_mask_binary[pose_id] > 0.5) & (depth > 0)
            if np.count_nonzero(valid_support) > 16:
                valid_pose_ids.append(pose_id)

        if len(valid_pose_ids) == 0:
            return pred_Rts, None, np.array([], dtype=np.int64), None

        valid_pose_ids = np.array(valid_pose_ids, dtype=np.int64)
        obj_index = self.FoundationPose_obj_id_to_index[obj_id]
        initial_poses_m = pred_Rts[valid_pose_ids].astype(np.float32).copy()
        initial_poses_m[:, :3, 3] *= self.foundationpose_pose_unit_scale

        self.FoundationPose_Item.last_visualization = None
        refined_poses_m, foundationpose_scores, _ = self.FoundationPose_Item.inference_batch(
            cam_K.astype(np.float32),
            initial_poses_m,
            img[None].astype(np.float32),
            depth[None].astype(np.float32),
            pred_mask_binary[valid_pose_ids].astype(np.float32),
            ob_id=obj_index,
            foundationpose_vis=foundationpose_vis,
            foundationpose_vis_stages=foundationpose_vis_stages,
        )
        foundationpose_vis_image = self.FoundationPose_Item.last_visualization

        if refined_poses_m is None or foundationpose_scores is None:
            return pred_Rts, None, valid_pose_ids, foundationpose_vis_image
        if refined_poses_m.shape[0] != valid_pose_ids.shape[0]:
            return pred_Rts, None, valid_pose_ids, foundationpose_vis_image

        refined_Rts = pred_Rts.astype(np.float32).copy()
        refined_subset = refined_poses_m.astype(np.float32).copy()
        refined_subset[:, :3, 3] /= self.foundationpose_pose_unit_scale
        refined_Rts[valid_pose_ids] = refined_subset
        return refined_Rts, foundationpose_scores.astype(np.float32), valid_pose_ids, foundationpose_vis_image

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
                depth=None, use_foundationpose=False, foundationpose_vis=False, foundationpose_vis_stages=None,
                use_megapose=False, megapose_vis=False, megapose_vis_stages=None):
        '''
        ---
        ---
        Run the full HccePose pipeline on one image.
        ---
        ---
        Resizes/pads the input, runs YOLO, per-object HccePose inference, PnP,
        optional FoundationPose (needs depth) or MegaPose refinement, and builds
        `results_dict` for downstream evaluation or visualization.

        Args:
            - cam_K: 3x3 intrinsics for the original (pre-resize) conventions after internal scaling.
            - img: OpenCV BGR uint8; HccePose uses BGR + BGR-permuted ImageNet norm. FoundationPose: ``Refinement_FP.inference_batch`` converts NHWC BGR→RGB before the NVLabs models. MegaPose: the subprocess packs RGB for ``ObservationTensor``; MegaPose refinement vis panels are assembled in BGR for ``cv2.imwrite``.
            - obj_ids: List of object ids to evaluate.
            - confidence_threshold, conf, iou, max_det: YOLO filtering knobs.
            - pnp_op: String selector mapped to internal PnP backend id.
            - depth: Depth in meters when refinement backends need it.
            - use_foundationpose / use_megapose: Enable optional refiners.
            - *_vis* flags: Pass-through visualization toggles.

        Returns:
            - results_dict: Structured detections, poses, masks, timing, etc.
        ---
        ---
        单张图像的完整 HccePose 流程主入口。
        ---
        ---
        包含缩放与 padding、YOLO、HccePose 裁剪推理、PnP、可选 FoundationPose /
        MegaPose 微调，并打包为 `results_dict`。

        参数:
            - cam_K: 相机内参（随内部缩放同步调整）。
            - img: OpenCV BGR uint8；HccePose 主干为 BGR + 按 BGR 重排的 ImageNet 归一化。FoundationPose：``Refinement_FP.inference_batch`` 将 NHWC BGR 转为 RGB 再送模型。MegaPose：子进程内打包 RGB 供 ``ObservationTensor``；MegaPose 微调可视化拼图为 BGR，便于 ``cv2.imwrite``。
            - obj_ids: 待评测物体 id 列表。
            - confidence_threshold, conf, iou, max_det: YOLO 相关阈值与上限。
            - pnp_op: PnP 后端选择字符串。
            - depth: 深度图（米），部分微调路径必需。
            - use_foundationpose / use_megapose: 是否启用对应微调。
            - *_vis* flags: 可视化开关。

        返回:
            - results_dict: 检测、位姿、掩码、耗时等结构化结果。
        '''
        if use_foundationpose and depth is None:
            raise ValueError('FoundationPose requires a depth map in meters.')
        if use_foundationpose and not self.foundationpose_registered:
            raise RuntimeError('FoundationPose is requested but not registered in Tester.')
        if use_megapose and not self.megapose_registered:
            self.register_megapose()

        pnp_op_l = [['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb'],[0,2,1]]
        depth_for_foundationpose = None
        if depth is not None:
            depth_for_foundationpose = depth.astype(np.float32).copy()
        height, width = img.shape[:2]
        ratio_ = 1
        ratio_ = max(height / (640*1), width / (640*1))
        pad_s = 32
        if ratio_ > 1:
            height_new = int(height / ratio_)
            width_new = int(width / ratio_)
            img = cv2.resize(img, (width_new, height_new), cv2.INTER_LINEAR)
            if depth_for_foundationpose is not None:
                depth_for_foundationpose = cv2.resize(depth_for_foundationpose, (width_new, height_new), interpolation=cv2.INTER_NEAREST)
            cam_K = cam_K.copy()
            cam_K[:2, :] /= ratio_
        height, width = img.shape[:2]
        if height % pad_s == 0:
            height_pad = height
            h_move = 0
        else:
            height_pad = (int(height / pad_s) + 1) * pad_s
            h_move = int((height_pad - height) / 2)
        if width % pad_s == 0:
            width_pad = width
            w_move = 0
        else:
            width_pad = (int(width / pad_s) + 1) * pad_s
            w_move = int((width_pad - width) / 2)
        if w_move > 0 or h_move > 0:
            img_new = np.zeros((height_pad, width_pad, 3), dtype=img.dtype)
            img_new[h_move:h_move + height, w_move:w_move + width, :] = img
            img = img_new
            if depth_for_foundationpose is not None:
                depth_new = np.zeros((height_pad, width_pad), dtype=depth_for_foundationpose.dtype)
                depth_new[h_move:h_move + height, w_move:w_move + width] = depth_for_foundationpose
                depth_for_foundationpose = depth_new
        cam_K = cam_K.copy()
        cam_K[0, 2] += w_move
        cam_K[1, 2] += h_move
        
        img_torch = torch.from_numpy(img.astype(np.float32)).to(self.device)
        
        img_torch = img_torch[None]

        with torch.amp.autocast('cuda'):
            t1 = time.time()
            stage_times = {
                'yolo': 0.0,
                'hccepose': 0.0,
                'foundationpose': 0.0,
                'megapose': 0.0,
                'visualization': 0.0,
            }
            
            det_results = {}
            results_dict = {}
            
            t_stage = time.time()
            scaled_img_torch = kornia.geometry.transform.resize(img_torch.permute(0,3,1,2), (int(img_torch.shape[1] * 1.0), int(img_torch.shape[2] * 1.0)), interpolation='bilinear')
            det_yolo = self.model_yolo(scaled_img_torch.clamp(0,255)/255, 
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
                        stage_times['hccepose'] += time.time() - t_stage

                        hccepose_Rts = pred_results['Rts'].astype(np.float32).copy()

                        if use_foundationpose and depth_for_foundationpose is not None and pred_results['Rts'].shape[0] > 0:
                            t_stage = time.time()
                            pred_mask_origin = kornia.geometry.transform.warp_perspective(
                                pred_mask[:,None,...],
                                torch.linalg.inv(Detect_Bbox_tfs_128.to(torch.float32)),
                                dsize=[img_torch[0].shape[0], img_torch[0].shape[1]], mode='nearest', align_corners=False,
                            )[:, 0]
                            refined_Rts, foundationpose_scores, valid_pose_ids, foundationpose_vis_image = self._run_foundationpose_refinement(
                                obj_id,
                                cam_K,
                                img,
                                depth_for_foundationpose,
                                pred_mask_origin.detach().cpu().numpy(),
                                hccepose_Rts,
                                foundationpose_vis=foundationpose_vis,
                                foundationpose_vis_stages=foundationpose_vis_stages,
                            )
                            if foundationpose_vis_image is not None:
                                pred_results['foundationpose_vis'] = foundationpose_vis_image
                            if foundationpose_scores is not None and foundationpose_scores.shape[0] == valid_pose_ids.shape[0]:
                                pred_results['Rts'] = refined_Rts
                                pred_results['foundationpose_scores'] = foundationpose_scores
                                pred_results['foundationpose_valid_pose_ids'] = valid_pose_ids
                                if use_megapose:
                                    pred_results['Rts_foundationpose'] = refined_Rts
                                # Use (200 - foundationpose_scores) / 200 as updated detection confidence for refined instances only.
                                conf_base = pred_results['conf']
                                device, dtype = conf_base.device, conf_base.dtype
                                vi = torch.as_tensor(valid_pose_ids, device=device, dtype=torch.long)
                                fp_conf = torch.as_tensor(
                                    (200.0 - foundationpose_scores) / 200.0,
                                    device=device,
                                    dtype=dtype,
                                )
                                conf_new = conf_base.clone()
                                conf_new[vi] = fp_conf
                                pred_results['conf'] = conf_new
                            stage_times['foundationpose'] += time.time() - t_stage

                        if use_megapose and hccepose_Rts.shape[0] > 0:
                            t_stage = time.time()
                            megapose_Rts, megapose_scores, megapose_valid_pose_ids, megapose_vis_image = self._run_megapose_refinement(
                                obj_id,
                                cam_K,
                                img,
                                depth_for_foundationpose,
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
            if foundationpose_vis:
                foundationpose_vis_items = []
                for obj_id in results_dict:
                    pred_results = results_dict[obj_id]
                    if isinstance(pred_results, dict) and 'foundationpose_vis' in pred_results:
                        foundationpose_vis_items.append((f'obj_{int(obj_id):06d}', pred_results['foundationpose_vis']))
                foundationpose_frame_vis = _foundationpose_stack_named_images(foundationpose_vis_items)
                if foundationpose_frame_vis is not None:
                    results_dict['show_foundationpose'] = foundationpose_frame_vis
            if megapose_vis:
                megapose_vis_items = []
                for obj_id in results_dict:
                    pred_results = results_dict[obj_id]
                    if isinstance(pred_results, dict) and 'megapose_vis' in pred_results:
                        megapose_vis_items.append((f'obj_{int(obj_id):06d}', pred_results['megapose_vis']))
                megapose_frame_vis = _foundationpose_stack_named_images(megapose_vis_items)
                if megapose_frame_vis is not None:
                    results_dict['show_megapose'] = megapose_frame_vis
            stage_times['visualization'] += time.time() - t_stage
            results_dict['time'] = t2 - t1
            other_time = max(0.0, results_dict['time'] - sum(stage_times.values()))
            results_dict['time_dict'] = {
                'yolo': stage_times['yolo'],
                'hccepose': stage_times['hccepose'],
                'foundationpose': stage_times['foundationpose'],
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
