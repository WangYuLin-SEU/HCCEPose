# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import json
import os

import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

from Refinement.foundationpose import make_mesh_tensors, nvdiffrast_render


_MESH_RENDER_CACHE = {}
_GLCTX_CACHE = {}


def convert_depth_to_meter(depth_raw):
    '''
    ---
    ---
    Convert a raw saved depth map to meter units.
    ---
    ---
    The helper infers a practical scale from the depth magnitude so the same
    utility can handle depth saved in meters, millimeters, or millimeters times
    ten.

    Args:
        - depth_raw: Raw depth image read from disk.

    Returns:
        - depth_m: Depth image converted to meters.
    ---
    ---
    将原始保存的深度图转换到米单位。
    ---
    ---
    该函数会根据深度数值量级推断一个实用缩放比例，因此同一套工具既可处理
    以米保存的深度图，也可处理以毫米或毫米乘十保存的深度图。

    参数:
        - depth_raw: 从磁盘读取的原始深度图。

    返回:
        - depth_m: 已转换为米单位的深度图。
    '''
    depth_m = depth_raw.astype(np.float32).copy()
    valid_mask = depth_m > 0
    if not np.any(valid_mask):
        return depth_m
    valid_depth = depth_m[valid_mask]
    depth_median = float(np.median(valid_depth))
    if depth_median > 2000.0:
        depth_scale_to_meter = 1e-4
    elif depth_median > 10.0:
        depth_scale_to_meter = 1e-3
    else:
        depth_scale_to_meter = 1.0
    depth_m *= depth_scale_to_meter
    return depth_m


def list_capture_frame_names(capture_dir):
    '''
    ---
    ---
    List capture frame stems from one RGBD capture directory.
    ---
    ---
    The function scans `*_rgb.png` files and returns their shared frame names in
    sorted order so downstream test scripts can iterate frames consistently.
    Example inputs are published under the Hugging Face dataset
    ``SEU-WYL/HccePose`` in ``test_imgs_RGBD/`` (e.g. stems ``000000``–``000003``).

    Args:
        - capture_dir: Directory containing captured RGBD frames.

    Returns:
        - frame_names: Sorted frame stem list.
    ---
    ---
    从一个 RGBD 采集目录中列出所有帧名。
    ---
    ---
    该函数会扫描 `*_rgb.png` 文件，并按排序后的顺序返回它们共享的帧名前缀，
    方便下游测试脚本稳定遍历各帧。
    示例数据见 Hugging Face 数据集 ``SEU-WYL/HccePose`` 的 ``test_imgs_RGBD/``
    目录（如 ``000000``–``000003``）。

    参数:
        - capture_dir: 保存 RGBD 采集帧的目录。

    返回:
        - frame_names: 排序后的帧名前缀列表。
    '''
    return sorted([
        file_name.replace('_rgb.png', '')
        for file_name in os.listdir(capture_dir)
        if file_name.endswith('_rgb.png')
    ])


def load_capture_frame(capture_dir, name):
    '''
    ---
    ---
    Load one color frame, depth map, and camera intrinsics from disk.
    ---
    ---
    The helper reads the color PNG (typically named ``{stem}_rgb.png``), raw depth,
    and camera intrinsics json, then returns both the original depth and the
    meter-converted depth used by refinement and comparison scripts. The color
    file is loaded with ``cv2.imread`` → **BGR** layout (OpenCV), despite ``rgb``
    in the filename.

    Args:
        - capture_dir: Capture directory path.
        - name: Shared frame stem.

    Returns:
        - image: BGR uint8 image (``cv2.imread`` layout; pass unchanged to ``Tester.predict``).
        - depth: Raw depth image.
        - depth_m: Depth image converted to meters.
        - cam_K: Camera intrinsic matrix.
    ---
    ---
    从磁盘加载一帧彩色图、深度与相机内参。
    ---
    ---
    读取 ``{stem}_rgb.png``、深度图与 ``camK.json``；彩色图经 ``cv2.imread`` 加载为 **BGR**
    uint8（文件名中的 ``rgb`` 仅表示三通道彩色，非内存 RGB 顺序）。同时返回原始深度与米制深度。

    参数:
        - capture_dir: 采集目录路径。
        - name: 共享帧名前缀。

    返回:
        - image: BGR uint8（与 ``cv2.imread`` 一致，可直接传入 ``Tester.predict``）。
        - depth: 原始深度图。
        - depth_m: 转换为米单位后的深度图。
        - cam_K: 相机内参矩阵。
    '''
    rgb_path = os.path.join(capture_dir, '%s_rgb.png' % name)
    depth_path = os.path.join(capture_dir, '%s_depth.png' % name)
    camera_path = os.path.join(capture_dir, '%s_camK.json' % name)

    image = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    with open(camera_path, 'r', encoding='utf-8') as f:
        camera_info = json.load(f)
    cam_K = np.array([
        [camera_info['fx'], 0.0, camera_info['cx']],
        [0.0, camera_info['fy'], camera_info['cy']],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    if image is None:
        raise FileNotFoundError(f'Failed to load rgb image: {rgb_path}')
    if depth is None:
        raise FileNotFoundError(f'Failed to load depth image: {depth_path}')

    depth_m = convert_depth_to_meter(depth)
    return image, depth, depth_m, cam_K


def _get_glctx(device):
    '''
    ---
    ---
    Get or create a cached nvdiffrast rasterization context.
    ---
    ---
    获取或创建缓存的 nvdiffrast 光栅化上下文。
    '''
    if device not in _GLCTX_CACHE:
        _GLCTX_CACHE[device] = dr.RasterizeCudaContext(device)
    return _GLCTX_CACHE[device]


def _get_mesh_bundle(model_path, device):
    '''
    ---
    ---
    Load one mesh once and cache its render tensors.
    ---
    ---
    The mesh is converted from millimeters to meters and packed into the tensor
    representation used by the shared renderer.
    ---
    ---
    只加载一次 mesh，并缓存其渲染张量。
    ---
    ---
    该函数会把 mesh 从毫米转换到米，并打包成共享 renderer 所使用的张量表示。
    '''
    cache_key = (model_path, device)
    if cache_key not in _MESH_RENDER_CACHE:
        mesh = trimesh.load(model_path, force='mesh', process=False)
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices.astype(np.float32) * 1e-3
        mesh_tensors = make_mesh_tensors(mesh, device=device)
        _MESH_RENDER_CACHE[cache_key] = (mesh, mesh_tensors)
    return _MESH_RENDER_CACHE[cache_key]


def _depth_to_color(depth_mm, valid_mask, vmin, vmax):
    '''
    ---
    ---
    Convert a depth patch in millimeters to a pseudo-color image.
    ---
    ---
    将毫米单位的深度块转换为伪彩色图像。
    '''
    canvas = np.zeros(depth_mm.shape + (3,), dtype=np.uint8)
    if not np.any(valid_mask):
        return canvas
    denom = max(float(vmax - vmin), 1e-6)
    depth_norm = ((depth_mm - vmin) / denom).clip(0.0, 1.0)
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    canvas[valid_mask] = color[valid_mask]
    return canvas


def _diff_to_color(diff_mm, valid_mask, vmax):
    '''
    ---
    ---
    Convert absolute depth differences to a pseudo-color visualization.
    ---
    ---
    将绝对深度差转换为伪彩色可视化结果。
    '''
    canvas = np.zeros(diff_mm.shape + (3,), dtype=np.uint8)
    if not np.any(valid_mask):
        return canvas
    vmax = max(float(vmax), 1e-6)
    diff_norm = (diff_mm / vmax).clip(0.0, 1.0)
    diff_u8 = (diff_norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)
    canvas[valid_mask] = color[valid_mask]
    return canvas


def _robust_diff_mode_mm(valid_diff, bin_size_mm=0.5):
    '''
    ---
    ---
    Estimate a robust mode of valid depth differences in millimeters.
    ---
    ---
    The mode is approximated with a histogram so the summary is less sensitive
    to long-tail outliers than a plain mean.
    ---
    ---
    估计有效深度差在毫米尺度下的稳健众数。
    ---
    ---
    该函数通过直方图近似众数，使统计结果比直接均值更不容易受到长尾异常值影响。
    '''
    if valid_diff.size == 0:
        return None
    valid_diff = np.asarray(valid_diff, dtype=np.float32)
    max_edge = float(np.max(valid_diff)) + float(bin_size_mm)
    if max_edge <= 0:
        return 0.0
    bins = np.arange(0.0, max_edge + float(bin_size_mm), float(bin_size_mm), dtype=np.float32)
    if bins.size < 2:
        bins = np.array([0.0, float(bin_size_mm)], dtype=np.float32)
    hist, edges = np.histogram(valid_diff, bins=bins)
    mode_id = int(np.argmax(hist))
    return float((edges[mode_id] + edges[mode_id + 1]) * 0.5)


def _mean_best_fraction_mm(valid_diff, fraction):
    '''
    ---
    ---
    Compute the mean depth error over the best aligned fraction.
    ---
    ---
    The helper sorts valid depth differences and averages the smallest portion,
    such as the best 70 percent used in the current visualization title.
    ---
    ---
    计算最佳对齐部分的平均深度误差。
    ---
    ---
    该函数会先对有效深度差排序，再对其中最小的一部分取均值，
    例如当前标题里展示的 best70%。
    '''
    if valid_diff.size == 0:
        return None
    valid_diff = np.sort(np.asarray(valid_diff, dtype=np.float32))
    keep = max(1, int(np.ceil(valid_diff.size * float(fraction))))
    return float(np.mean(valid_diff[:keep]))


def _wrap_title_lines(title, max_width, font, font_scale, thickness):
    '''
    ---
    ---
    Wrap one title string into multiple lines under a width limit.
    ---
    ---
    将一条标题字符串按宽度限制自动换成多行。
    '''
    words = str(title).split()
    if len(words) == 0:
        return ['']
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + ' ' + word
        width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _add_title(image, title, height=None):
    '''
    ---
    ---
    Add a title bar above one visualization panel.
    ---
    ---
    为单个可视化面板添加标题栏。
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.54
    thickness = 1
    pad_x = 8
    pad_top = 8
    line_step = 18
    max_width = max(40, image.shape[1] - pad_x * 2)
    lines = _wrap_title_lines(title, max_width, font, font_scale, thickness)
    header_h = pad_top * 2 + line_step * len(lines)
    if height is not None:
        header_h = max(int(height), header_h)
    header = np.full((header_h, image.shape[1], 3), 18, dtype=np.uint8)
    y = pad_top + 12
    for line in lines:
        cv2.putText(header, line, (pad_x, y), font, font_scale, (230, 230, 230), thickness, cv2.LINE_AA)
        y += line_step
    return np.concatenate([header, image], axis=0)


def _add_panel_border(image, color=(245, 245, 245), thickness=2):
    '''
    ---
    ---
    Add a visible border around the image area of one panel.
    ---
    ---
    The border is used to reduce ambiguity when cropped depth panels are not
    visually aligned. Only the image area is framed, not the title bar.
    ---
    ---
    为单个面板的图像区域添加明显边框。
    ---
    ---
    该边框用于在不同裁剪深度图视觉上不对齐时减少误解。
    这里只框图像区域，不框标题栏。
    '''
    return cv2.copyMakeBorder(
        image,
        int(thickness),
        int(thickness),
        int(thickness),
        int(thickness),
        cv2.BORDER_CONSTANT,
        value=tuple(int(v) for v in color),
    )


def _stack_cols(images, pad=4):
    '''
    ---
    ---
    Stack multiple panels horizontally with a dark separator.
    ---
    ---
    使用深色分隔条将多个面板横向拼接。
    '''
    if len(images) == 0:
        return None
    max_h = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        pad_h = max_h - image.shape[0]
        padded.append(cv2.copyMakeBorder(image, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18)))
    if len(padded) == 1:
        return padded[0]
    separator = np.full((max_h, pad, 3), 18, dtype=np.uint8)
    stacked = [padded[0]]
    for image in padded[1:]:
        stacked.extend([separator, image])
    return np.concatenate(stacked, axis=1)


def _stack_rows(images, pad=6):
    '''
    ---
    ---
    Stack multiple panels vertically with a dark separator.
    ---
    ---
    使用深色分隔条将多个面板纵向拼接。
    '''
    if len(images) == 0:
        return None
    max_w = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        pad_w = max_w - image.shape[1]
        padded.append(cv2.copyMakeBorder(image, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(18, 18, 18)))
    if len(padded) == 1:
        return padded[0]
    separator = np.full((pad, max_w, 3), 18, dtype=np.uint8)
    stacked = [padded[0]]
    for image in padded[1:]:
        stacked.extend([separator, image])
    return np.concatenate(stacked, axis=0)


def _crop_from_masks(masks, image_shape, margin=20):
    '''
    ---
    ---
    Compute one crop window that covers a set of boolean masks.
    ---
    ---
    根据一组布尔掩膜计算一个统一裁剪窗口。
    '''
    valid = np.zeros(image_shape[:2], dtype=bool)
    for mask in masks:
        if mask is not None:
            valid |= mask
    if not np.any(valid):
        return 0, 0, image_shape[1], image_shape[0]
    ys, xs = np.where(valid)
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, image_shape[1])
    y1 = min(int(ys.max()) + margin + 1, image_shape[0])
    return x0, y0, x1, y1


def render_pose_depth_mm(model_path, pose_mm, cam_K, image_shape, device='cuda:0'):
    '''
    ---
    ---
    Render the depth map of one pose and return millimeter depth.
    ---
    ---
    The input pose is provided in the current project convention with
    millimeter translation. Rendering is done in meters internally and then
    converted back to millimeters for direct comparison with captured depth.

    Args:
        - model_path: Mesh path.
        - pose_mm: Pose matrix with translation in millimeters.
        - cam_K: Camera intrinsic matrix.
        - image_shape: Target output image shape.
        - device: Rendering device.

    Returns:
        - depth_mm: Rendered depth map in millimeters.
    ---
    ---
    渲染单个位姿对应的深度图，并返回毫米单位深度。
    ---
    ---
    输入位姿遵循当前项目的约定，即平移量为毫米。内部渲染时会先转换到米，
    最后再转回毫米，以便和实拍深度图直接比较。

    参数:
        - model_path: mesh 路径。
        - pose_mm: 平移为毫米的位姿矩阵。
        - cam_K: 相机内参矩阵。
        - image_shape: 目标输出图像尺寸。
        - device: 渲染设备。

    返回:
        - depth_mm: 毫米单位的渲染深度图。
    '''
    mesh, mesh_tensors = _get_mesh_bundle(model_path, device)
    glctx = _get_glctx(device)
    pose_m = np.asarray(pose_mm, dtype=np.float32).copy()
    pose_m[:3, 3] *= 1e-3
    pose_t = torch.as_tensor(pose_m[None], dtype=torch.float32, device=device)
    _, depth_render, _ = nvdiffrast_render(
        K=cam_K.astype(np.float32),
        H=int(image_shape[0]),
        W=int(image_shape[1]),
        ob_in_cams=pose_t,
        glctx=glctx,
        context='cuda',
        mesh_tensors=mesh_tensors,
        mesh=mesh,
        output_size=[int(image_shape[0]), int(image_shape[1])],
        use_light=False,
        device=device,
    )
    return depth_render[0].detach().cpu().numpy() * 1000.0


def build_depth_comparison_visual(observed_depth_mm, cam_K, model_path, pose_sets_mm, device='cuda:0', max_items=4):
    '''
    ---
    ---
    Build a multi-method depth comparison visualization and summary.
    ---
    ---
    The function renders pose hypotheses from different methods, crops them to a
    shared region, computes depth error statistics, and composes the final depth
    comparison image used by the FP-vs-MP script.

    Args:
        - observed_depth_mm: Captured depth image in millimeters.
        - cam_K: Camera intrinsic matrix.
        - model_path: Mesh path for rendering.
        - pose_sets_mm: Dict of method name to pose arrays in millimeters.
        - device: Rendering device.
        - max_items: Maximum number of pose rows to visualize.

    Returns:
        - visual: Combined depth comparison image.
        - summary: Statistics dictionary for each compared method.
    ---
    ---
    构建多方法深度对比可视化图与统计结果。
    ---
    ---
    该函数会渲染不同方法给出的位姿，裁剪到统一区域，计算深度误差统计，
    并拼出 FP-vs-MP 脚本所使用的最终深度对比图片。

    参数:
        - observed_depth_mm: 毫米单位的实拍深度图。
        - cam_K: 相机内参矩阵。
        - model_path: 用于渲染的 mesh 路径。
        - pose_sets_mm: 方法名到毫米位姿数组的字典。
        - device: 渲染设备。
        - max_items: 最多可视化多少行位姿。

    返回:
        - visual: 拼接后的深度对比图。
        - summary: 每种方法对应的统计结果字典。
    '''
    pose_names = list(pose_sets_mm.keys())
    pose_arrays = {
        name: np.asarray(pose_sets_mm[name], dtype=np.float32)
        for name in pose_names
    }
    if len(pose_arrays) == 0:
        return None, {}

    reference_names = [name for name in ['HccePose', 'FoundationPose'] if name in pose_arrays]
    if len(reference_names) == 0:
        reference_names = [pose_names[0]]
    pose_count = min(len(pose_arrays[name]) for name in reference_names)
    pose_count = min(int(max_items), int(pose_count))
    if pose_count <= 0:
        return None, {}

    observed_depth_mm = np.asarray(observed_depth_mm, dtype=np.float32)
    image_shape = observed_depth_mm.shape[:2]
    row_images = []
    summary = {}

    for pose_id in range(pose_count):
        render_depths = {}
        for name in pose_names:
            if pose_id >= len(pose_arrays[name]):
                continue
            render_depths[name] = render_pose_depth_mm(model_path, pose_arrays[name][pose_id], cam_K, image_shape, device=device)

        reference_masks = [render_depths[name] > 0 for name in reference_names if name in render_depths]
        if len(reference_masks) == 0:
            reference_masks = [render_depth > 0 for render_depth in render_depths.values()]
        x0, y0, x1, y1 = _crop_from_masks(reference_masks, image_shape)
        observed_crop = observed_depth_mm[y0:y1, x0:x1]
        observed_valid = observed_crop > 0

        ref_depth_values = []
        if np.any(observed_valid):
            ref_depth_values.append(observed_crop[observed_valid])
        for name in reference_names:
            if name not in render_depths:
                continue
            render_crop = render_depths[name][y0:y1, x0:x1]
            render_valid = render_crop > 0
            if np.any(render_valid):
                ref_depth_values.append(render_crop[render_valid])
        if len(ref_depth_values) == 0:
            observed_depth_min, observed_depth_max = 0.0, 1.0
        else:
            merged_ref = np.concatenate(ref_depth_values, axis=0)
            observed_depth_min = float(np.percentile(merged_ref, 2))
            observed_depth_max = float(np.percentile(merged_ref, 98))
            if observed_depth_max <= observed_depth_min:
                observed_depth_max = observed_depth_min + 1.0

        panels = [
            _add_title(
                _add_panel_border(_depth_to_color(observed_crop, observed_valid, observed_depth_min, observed_depth_max)),
                'Captured depth',
            )
        ]
        method_stats = {}
        for name in pose_names:
            if name not in render_depths:
                method_stats[name] = {
                    'valid_pixel_count': 0,
                    'mean_abs_diff_mm': None,
                    'median_abs_diff_mm': None,
                    'max_abs_diff_mm': None,
                    'depth_range_mm': None,
                    'diff_clip_mm': None,
                }
                blank = np.zeros(observed_crop.shape + (3,), dtype=np.uint8)
                panels.append(_add_title(_add_panel_border(blank), f'{name} render'))
                panels.append(_add_title(_add_panel_border(blank), f'{name} |diff| n/a'))
                continue

            render_crop = render_depths[name][y0:y1, x0:x1]
            render_valid = render_crop > 0
            depth_values = []
            if np.any(observed_valid):
                depth_values.append(observed_crop[observed_valid])
            if np.any(render_valid):
                depth_values.append(render_crop[render_valid])
            if len(depth_values) == 0:
                depth_min, depth_max = 0.0, 1.0
            else:
                merged = np.concatenate(depth_values, axis=0)
                depth_min = float(np.percentile(merged, 2))
                depth_max = float(np.percentile(merged, 98))
                if depth_max <= depth_min:
                    depth_max = depth_min + 1.0

            render_panel = _depth_to_color(render_crop, render_valid, depth_min, depth_max)
            panels.append(_add_title(_add_panel_border(render_panel), f'{name} render'))

            valid_mask = observed_valid & render_valid
            abs_diff = np.abs(render_crop - observed_crop)
            if np.any(valid_mask):
                valid_diff = abs_diff[valid_mask]
                diff_max = float(np.percentile(valid_diff, 95))
                diff_max = max(diff_max, 1.0)
                diff_panel = _diff_to_color(abs_diff, valid_mask, diff_max)
                mean_abs = float(np.mean(valid_diff))
                median_abs = float(np.median(valid_diff))
                mode_abs = _robust_diff_mode_mm(valid_diff)
                mean_best50 = _mean_best_fraction_mm(valid_diff, 0.50)
                mean_best70 = _mean_best_fraction_mm(valid_diff, 0.70)
                mean_best80 = _mean_best_fraction_mm(valid_diff, 0.80)
                method_stats[name] = {
                    'valid_pixel_count': int(np.count_nonzero(valid_mask)),
                    'mean_abs_diff_mm': mean_abs,
                    'median_abs_diff_mm': median_abs,
                    'mode_abs_diff_mm': mode_abs,
                    'mean_best50_abs_diff_mm': mean_best50,
                    'mean_best70_abs_diff_mm': mean_best70,
                    'mean_best80_abs_diff_mm': mean_best80,
                    'max_abs_diff_mm': float(np.max(valid_diff)),
                    'depth_range_mm': [float(depth_min), float(depth_max)],
                    'diff_clip_mm': float(diff_max),
                }
                diff_title = f'{name} |diff| med {median_abs:.1f} mm | best70% {mean_best70:.1f} mm'
            else:
                diff_panel = np.zeros(render_crop.shape + (3,), dtype=np.uint8)
                method_stats[name] = {
                    'valid_pixel_count': 0,
                    'mean_abs_diff_mm': None,
                    'median_abs_diff_mm': None,
                    'mode_abs_diff_mm': None,
                    'mean_best50_abs_diff_mm': None,
                    'mean_best70_abs_diff_mm': None,
                    'mean_best80_abs_diff_mm': None,
                    'max_abs_diff_mm': None,
                    'depth_range_mm': [float(depth_min), float(depth_max)],
                    'diff_clip_mm': None,
                }
                diff_title = f'{name} |diff| n/a'
            panels.append(_add_title(_add_panel_border(diff_panel), diff_title))

        row_canvas = _stack_cols(panels, pad=4)
        row_images.append(_add_title(row_canvas, f'Pose {pose_id}', height=32))
        summary[str(pose_id)] = {
            'crop_box_xyxy': [int(x0), int(y0), int(x1), int(y1)],
            'reference_methods': list(reference_names),
            'captured_depth_range_mm': [float(observed_depth_min), float(observed_depth_max)],
            'methods': method_stats,
        }

    return _stack_rows(row_images, pad=8), summary
