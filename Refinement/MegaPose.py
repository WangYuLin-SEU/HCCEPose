# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import cv2
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


MEGAPOSE_PIP_PACKAGES = [
    'numpy==1.26.4',
    'opencv-python==4.9.0.80',
    'pandas==2.2.3',
    'pyyaml==6.0.2',
    'scipy==1.13.1',
    'trimesh==4.11.5',
    'webdataset==0.2.100',
    'roma==1.5.4',
    'omegaconf==2.3.0',
    'simplejson==3.20.2',
    'colorama==0.4.6',
    'structlog==25.5.0',
    'panda3d==1.10.16',
    'panda3d-gltf==1.3.0',
    'transforms3d==0.4.2',
    'tqdm==4.67.1',
    'bokeh==3.4.3',
    'matplotlib==3.9.4',
    'pillow==9.4.0',
    'plyfile==1.1.3',
    'h5py==3.14.0',
    'seaborn==0.13.2',
    'pyarrow==21.0.0',
    'imageio==2.37.0',
    'psutil==7.0.0',
    'pypng',
    'joblib',
]


def _megapose_resolved_local_data(project_root):
    megapose_repo = (Path(project_root) / 'third_party_megapose6d').resolve()
    data = megapose_repo / 'local_data'
    data.mkdir(parents=True, exist_ok=True)
    return data


def _ensure_megapose_model_artifacts(project_root, coarse_run_id, refiner_run_id):
    '''
    MegaPose inference needs `config.yaml` next to each `checkpoint.pth.tar`. Partial
    mirrors (checkpoint-only) or skipped rclone leave VS Code / subprocess debugging
    failing with FileNotFoundError; try the official downloader once before load.
    Paths are resolved from `MEGAPOSE_DATA_DIR` (set by Refinement_MP) or repo default;
    checks do not depend on cwd.
    '''
    megapose_repo = (Path(project_root) / 'third_party_megapose6d').resolve()
    local_data = Path(os.environ['MEGAPOSE_DATA_DIR']).resolve()
    models_root = (local_data / 'megapose-models').resolve()

    run_ids = list(dict.fromkeys((coarse_run_id, refiner_run_id)))
    paths_needed = []
    for rid in run_ids:
        paths_needed.append(models_root / rid / 'checkpoint.pth.tar')
        paths_needed.append(models_root / rid / 'config.yaml')
    if all(p.is_file() for p in paths_needed):
        return

    env = os.environ.copy()
    bin_dir = str(Path(sys.executable).resolve().parent)
    env['PATH'] = bin_dir + os.pathsep + env.get('PATH', '')
    env['MEGAPOSE_DATA_DIR'] = os.fspath(local_data)
    env['MEGAPOSE_DIR'] = os.fspath(megapose_repo)

    cmd = [sys.executable, '-m', 'megapose.scripts.download', '--megapose_models']
    proc = subprocess.run(cmd, cwd=os.fspath(megapose_repo), env=env, capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(
            'megapose.scripts.download --megapose_models failed (exit %s). '
            'Install `rclone` in this env (conda package `rclone`) and ensure PATH '
            'includes its bin when debugging.\n'
            '官方模型下载失败（退出码 %s）：请在 megapose 环境中安装 rclone，'
            '调试时保证 PATH 含该环境的 bin。'
            % (proc.returncode, proc.returncode)
        )

    still = [p for p in paths_needed if not p.is_file()]
    if still:
        # Upstream download uses copyto + epoch excludes; often skips config.yaml.
        rclone = Path(sys.executable).resolve().parent / 'rclone'
        cfg = megapose_repo / 'rclone.conf'
        if rclone.is_file() and cfg.is_file():
            rc_env = os.environ.copy()
            rc_env['PATH'] = str(rclone.parent) + os.pathsep + rc_env.get('PATH', '')
            for rid in run_ids:
                dstdir = models_root / rid
                dstdir.mkdir(parents=True, exist_ok=True)
                dst = os.fspath(dstdir) + '/'
                for name in ('config.yaml', 'checkpoint.pth.tar'):
                    src = 'inria_data:megapose-models/%s/%s' % (rid, name)
                    subprocess.run(
                        [
                            os.fspath(rclone), 'copy', src, dst,
                            '--config', os.fspath(cfg),
                            '--retries', '8', '--low-level-retries', '32',
                        ],
                        env=rc_env,
                        capture_output=False,
                    )
        still = [p for p in paths_needed if not p.is_file()]
    if still:
        raise FileNotFoundError(
            'MegaPose model files still missing after download and per-file rclone:\n  %s\n'
            'Example: %s copy inria_data:megapose-models/<run_id>/config.yaml <dest>/ --config %s'
            % (
                '\n  '.join(os.fspath(p) for p in still),
                os.fspath(Path(sys.executable).resolve().parent / 'rclone'),
                os.fspath(megapose_repo / 'rclone.conf'),
            )
        )


def _megapose_parse_vis_stages(vis_stages, refine_stage_count):
    '''
    ---
    ---
    Parse requested MegaPose visualization stages into valid iteration ids.
    ---
    ---
    The helper accepts integers or strings such as `all` and normalizes them to
    a clean list of refinement iteration indices.
    ---
    ---
    将 MegaPose 可视化阶段请求解析为合法的迭代编号。
    ---
    ---
    该函数可接受整数或 `all` 等字符串，并将其规范化为一组有效的 refinement
    迭代序号。
    '''
    if refine_stage_count <= 0:
        return []
    if vis_stages is None:
        return list(range(1, refine_stage_count + 1))
    stages = []
    for stage in vis_stages:
        if isinstance(stage, str):
            stage_str = stage.strip().lower()
            if stage_str == 'all':
                return list(range(1, refine_stage_count + 1))
            if stage_str.startswith('iter'):
                stage_str = stage_str.replace('iter', '').replace('iteration', '')
            try:
                stage = int(stage_str)
            except ValueError:
                continue
        try:
            stage_i = int(stage)
        except (TypeError, ValueError):
            continue
        if 1 <= stage_i <= refine_stage_count and stage_i not in stages:
            stages.append(stage_i)
    if len(stages) == 0:
        return list(range(1, refine_stage_count + 1))
    return stages


def _megapose_tensor_rgb_to_uint8(image):
    '''
    ---
    ---
    Convert a MegaPose image-like tensor or array to uint8 RGB visualization.
    ---
    ---
    将 MegaPose 风格的图像张量或数组转换为 uint8 RGB 可视化格式。
    '''
    if hasattr(image, 'detach'):
        image = image.detach().float().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in [3, 4, 6, 9, 12] and image.shape[-1] not in [3, 4]:
        image = image[:3].transpose(1, 2, 0)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        image = image[:, :, :3]
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.dtype != np.uint8:
        if image.max() <= 1.5:
            image = np.clip(image, 0.0, 1.0) * 255.0
        else:
            image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)
    return image


def _megapose_rgb_uint8_hwc_to_bgr(image):
    '''
    MegaPose tensors are RGB; OpenCV drawing and ``cv2.imwrite`` expect BGR.
    Convert uint8 HxWx3 panels once before ``cv2.line`` / ``rectangle`` so a later
    full-frame RGB2BGR does not swap R/B on overlays (legend, boxes, contours).
    '''
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)


def _megapose_add_title(image, title, height=34, bg_value=18):
    '''
    ---
    ---
    Add a title bar above one MegaPose visualization panel.
    ---
    ---
    为一个 MegaPose 可视化面板添加标题栏。
    '''
    header = np.full((height, image.shape[1], 3), bg_value, dtype=np.uint8)
    cv2.putText(header, str(title), (8, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def _megapose_add_legend(image, entries, height=44, bg_value=18):
    '''
    ---
    ---
    Add a compact legend panel below one MegaPose visualization block.
    ---
    ---
    在 MegaPose 可视化块下方添加紧凑的图例面板。
    '''
    if entries is None or len(entries) == 0:
        return image
    legend = np.full((height, image.shape[1], 3), bg_value, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    item_h = 14
    left_pad = 8
    right_pad = 12
    row_y = [14, 32]
    x = left_pad
    row_id = 0
    for color, label in entries:
        label = str(label)
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        item_w = 14 + 8 + text_size[0] + 18
        if x + item_w > image.shape[1] - right_pad and row_id == 0:
            row_id = 1
            x = left_pad
        if x + item_w > image.shape[1] - right_pad:
            break
        y = row_y[row_id]
        cv2.rectangle(legend, (x, y - item_h + 2), (x + 14, y + 2), color, -1)
        x += 22
        cv2.putText(legend, label, (x, y), font, font_scale, (235, 235, 235), thickness, cv2.LINE_AA)
        x += text_size[0] + 18
    return np.concatenate([image, legend], axis=0)


def _megapose_hstack_images(images, bg_value=18):
    '''
    ---
    ---
    Stack visualization panels horizontally on a dark background.
    ---
    ---
    在深色背景上将多个可视化面板横向拼接。
    '''
    images = [image for image in images if image is not None]
    if len(images) == 0:
        return None
    max_h = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        pad_h = max_h - image.shape[0]
        padded.append(cv2.copyMakeBorder(image, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(bg_value, bg_value, bg_value)))
    return np.concatenate(padded, axis=1)


def _megapose_vstack_images(images, bg_value=18):
    '''
    ---
    ---
    Stack visualization panels vertically with row spacing.
    ---
    ---
    带行间隔地将多个可视化面板纵向拼接。
    '''
    images = [image for image in images if image is not None]
    if len(images) == 0:
        return None
    max_w = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        pad_w = max_w - image.shape[1]
        padded.append(cv2.copyMakeBorder(image, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(bg_value, bg_value, bg_value)))
    spacer = np.full((8, max_w, 3), bg_value, dtype=np.uint8)
    canvas = padded[0]
    for image in padded[1:]:
        canvas = np.concatenate([canvas, spacer, image], axis=0)
    return canvas


def _megapose_make_overlay(observed_rgb, rendered_rgb, render_mask=None, contour_color=(255, 0, 255)):
    '''
    ---
    ---
    Blend observed and rendered images into one overlay panel.
    ---
    ---
    Inputs follow MegaPose RGB layout internally; the returned array is **BGR** uint8
    for stacking with other OpenCV-built panels and ``cv2.imwrite``.
    The overlay highlights rendered support regions and also draws the render
    contour so pose differences can be judged visually.
    ---
    ---
    将观测图与渲染图混合为一张 overlay 面板。
    ---
    ---
    该函数会突出渲染支撑区域，并额外绘制渲染轮廓，
    以便直观判断位姿差异。
    '''
    observed_rgb = _megapose_tensor_rgb_to_uint8(observed_rgb)
    rendered_rgb = _megapose_tensor_rgb_to_uint8(rendered_rgb)
    overlay = observed_rgb.copy()
    if render_mask is None:
        render_mask = np.max(rendered_rgb, axis=2) > 0
    if np.any(render_mask):
        overlay[render_mask] = np.clip(
            observed_rgb[render_mask].astype(np.float32) * 0.55 + rendered_rgb[render_mask].astype(np.float32) * 0.45,
            0.0,
            255.0,
        ).astype(np.uint8)
        edges = cv2.Canny((render_mask.astype(np.uint8) * 255), 30, 100)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
        overlay[edges > 0] = contour_color
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


def _megapose_extract_render_visual(render_tensor):
    '''
    ---
    ---
    Extract a stable visualization image from a MegaPose render tensor.
    ---
    ---
    Depending on the available channels, the helper prefers RGB render and then
    falls back to normal-based visualization when the RGB render is nearly dark.
    ---
    ---
    从 MegaPose 渲染张量中提取稳定的可视化图像。
    ---
    ---
    该函数会优先使用 RGB 渲染；若 RGB 渲染几乎全黑，则退回到法线可视化。
    '''
    render_np = render_tensor.detach().float().cpu().numpy() if hasattr(render_tensor, 'detach') else np.asarray(render_tensor)
    render_np = np.asarray(render_np)
    if render_np.ndim != 3:
        return _megapose_tensor_rgb_to_uint8(render_np)

    rgb_vis = _megapose_tensor_rgb_to_uint8(render_np[:3])
    if np.max(rgb_vis) > 8:
        return rgb_vis

    if render_np.shape[0] >= 6:
        normals = render_np[3:6].transpose(1, 2, 0)
        mask = np.linalg.norm(normals, axis=2) > 1e-6
        normals_vis = normals.copy()
        if normals_vis.min() >= -1.1 and normals_vis.max() <= 1.1:
            normals_vis = (normals_vis + 1.0) * 0.5
        normals_vis = np.clip(normals_vis, 0.0, 1.0)
        normals_vis = (normals_vis * 255.0).astype(np.uint8)
        normals_vis[~mask] = 0
        return normals_vis

    return rgb_vis


def _megapose_unit_scale(mesh_units):
    '''
    ---
    ---
    Convert a mesh unit string to a metric scale factor.
    ---
    ---
    将 mesh 单位字符串转换为米制缩放系数。
    '''
    mesh_units = str(mesh_units).lower()
    if mesh_units in ['m', 'meter', 'meters']:
        return 1.0
    if mesh_units in ['mm', 'millimeter', 'millimeters']:
        return 1e-3
    if mesh_units in ['cm', 'centimeter', 'centimeters']:
        return 1e-2
    return 1.0


def _megapose_project_points(points_obj, K, TCO):
    '''
    ---
    ---
    Project 3D object points to image coordinates under one pose.
    ---
    ---
    在给定位姿下，将三维物体点投影到图像坐标。
    '''
    points_obj = np.asarray(points_obj, dtype=np.float32)
    K_np = K.detach().float().cpu().numpy() if hasattr(K, 'detach') else np.asarray(K, dtype=np.float32)
    TCO_np = TCO.detach().float().cpu().numpy() if hasattr(TCO, 'detach') else np.asarray(TCO, dtype=np.float32)
    points_cam = (TCO_np[:3, :3] @ points_obj.T).T + TCO_np[:3, 3][None]
    valid = points_cam[:, 2] > 1e-6
    uv = np.full((points_obj.shape[0], 2), -1.0, dtype=np.float32)
    if np.any(valid):
        pts = points_cam[valid]
        uv_valid = np.empty((pts.shape[0], 2), dtype=np.float32)
        uv_valid[:, 0] = K_np[0, 0] * pts[:, 0] / pts[:, 2] + K_np[0, 2]
        uv_valid[:, 1] = K_np[1, 1] * pts[:, 1] / pts[:, 2] + K_np[1, 2]
        uv[valid] = uv_valid
    return uv, valid


def _megapose_draw_projected_geometry(image, K, TCO, sample_points=None, bbox_corners=None, box_color=(0, 255, 0), point_color=(0, 220, 255)):
    '''
    ---
    ---
    Draw projected 3D box edges and sparse vertices on one image.
    ---
    ---
    在一张图像上绘制投影后的三维包围盒边和稀疏顶点。
    '''
    canvas = image.copy()
    h, w = canvas.shape[:2]
    if bbox_corners is not None:
        bbox_edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.int32)
        uv_bbox, valid_bbox = _megapose_project_points(bbox_corners, K, TCO)
        for start_id, end_id in bbox_edges:
            if not (valid_bbox[start_id] and valid_bbox[end_id]):
                continue
            pt0 = tuple(np.round(uv_bbox[start_id]).astype(np.int32))
            pt1 = tuple(np.round(uv_bbox[end_id]).astype(np.int32))
            if (pt0[0] < -w or pt0[0] > 2 * w or pt0[1] < -h or pt0[1] > 2 * h or
                    pt1[0] < -w or pt1[0] > 2 * w or pt1[1] < -h or pt1[1] > 2 * h):
                continue
            cv2.line(canvas, pt0, pt1, box_color, 2, cv2.LINE_AA)
    if sample_points is not None and len(sample_points) > 0:
        uv_pts, valid_pts = _megapose_project_points(sample_points, K, TCO)
        for uv in uv_pts[valid_pts]:
            x_i, y_i = np.round(uv).astype(np.int32)
            if 0 <= x_i < w and 0 <= y_i < h:
                cv2.circle(canvas, (x_i, y_i), 2, point_color, -1, cv2.LINE_AA)
    return canvas


def _megapose_map_box_to_crop(box_xyxy, crop_xyxy, out_shape):
    '''
    ---
    ---
    Map a box from image coordinates into crop coordinates.
    ---
    ---
    将原图坐标系中的检测框映射到裁剪图坐标系。
    '''
    if box_xyxy is None or crop_xyxy is None:
        return None
    box = box_xyxy.detach().float().cpu().numpy() if hasattr(box_xyxy, 'detach') else np.asarray(box_xyxy, dtype=np.float32)
    crop = crop_xyxy.detach().float().cpu().numpy() if hasattr(crop_xyxy, 'detach') else np.asarray(crop_xyxy, dtype=np.float32)
    if box.shape[0] != 4 or crop.shape[0] != 4:
        return None
    crop_w = max(float(crop[2] - crop[0]), 1e-6)
    crop_h = max(float(crop[3] - crop[1]), 1e-6)
    out_h, out_w = out_shape[:2]
    mapped = np.empty(4, dtype=np.float32)
    mapped[0] = (box[0] - crop[0]) * out_w / crop_w
    mapped[1] = (box[1] - crop[1]) * out_h / crop_h
    mapped[2] = (box[2] - crop[0]) * out_w / crop_w
    mapped[3] = (box[3] - crop[1]) * out_h / crop_h
    return mapped


def _megapose_draw_2d_box(image, box_xyxy, color=(0, 165, 255), thickness=2):
    '''
    ---
    ---
    Draw one clipped 2D box on the visualization image.
    ---
    ---
    在可视化图像上绘制一个裁剪后的二维框。
    '''
    if box_xyxy is None:
        return image
    canvas = image.copy()
    h, w = canvas.shape[:2]
    x0, y0, x1, y1 = np.round(box_xyxy).astype(np.int32)
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return canvas
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    return canvas


def _megapose_build_refinement_visualization(all_outputs, vis_geometry=None, vis_stages=None, vis_max_items=4):
    '''
    ---
    ---
    Build the grouped MegaPose refinement visualization panel.
    ---
    ---
    The visualization organizes poses by rows and refinement stages by columns,
    and shows observed, render and overlay panels with shared geometry hints.

    Args:
        - all_outputs: MegaPose refinement outputs across iterations.
        - vis_geometry: Optional projected box and point templates per object label.
        - vis_stages: Requested refinement stages to visualize.
        - vis_max_items: Maximum number of pose rows to include.

    Returns:
        - visual: Combined MegaPose refinement visualization image.
    ---
    ---
    构建分组式的 MegaPose refinement 可视化面板。
    ---
    ---
    该可视化会按“位姿为行、微调阶段为列”的方式组织布局，
    并展示 observed、render、overlay 三类面板以及共享几何提示。

    参数:
        - all_outputs: 各轮迭代的 MegaPose refinement 输出。
        - vis_geometry: 按物体标签保存的投影框和点模板，可选。
        - vis_stages: 需要显示的微调阶段。
        - vis_max_items: 最多显示多少行位姿。

    返回:
        - visual: 拼接后的 MegaPose refinement 可视化图像。
    '''
    if all_outputs is None or len(all_outputs) == 0:
        return None
    rows = []
    pose_counter = 0
    vis_max_items = max(1, int(vis_max_items))
    for batch_outputs in all_outputs:
        iteration_ids = sorted([
            int(key.split('=')[1])
            for key in batch_outputs.keys()
            if key.startswith('iteration=')
        ])
        if len(iteration_ids) == 0:
            continue
        selected_stage_ids = _megapose_parse_vis_stages(vis_stages, max(iteration_ids))
        first_output = batch_outputs[f'iteration={iteration_ids[0]}']
        batch_size = int(first_output.images_crop.shape[0])
        for local_idx in range(batch_size):
            if pose_counter >= vis_max_items:
                return _megapose_vstack_images(rows)
            stage_tiles = []
            for stage_id in selected_stage_ids:
                stage_key = f'iteration={stage_id}'
                if stage_key not in batch_outputs:
                    continue
                stage_output = batch_outputs[stage_key]
                observed_rgb = _megapose_tensor_rgb_to_uint8(stage_output.images_crop[local_idx])
                rendered_rgb = _megapose_extract_render_visual(stage_output.renders[local_idx])
                observed_base = observed_rgb.copy()
                render_mask = np.max(rendered_rgb, axis=2) > 0
                observed_bgr = _megapose_rgb_uint8_hwc_to_bgr(observed_rgb)
                rendered_bgr = _megapose_rgb_uint8_hwc_to_bgr(rendered_rgb)
                label_i = stage_output.labels[local_idx]
                geometry_i = None if vis_geometry is None else vis_geometry.get(label_i)
                input_pose = getattr(stage_output, 'TCO_input', None)
                input_pose_i = None if input_pose is None else input_pose[local_idx]
                output_pose_i = stage_output.TCO_output[local_idx]
                render_panel = rendered_bgr.copy()
                if geometry_i is not None:
                    render_panel = _megapose_draw_projected_geometry(
                        render_panel,
                        stage_output.K_crop[local_idx],
                        output_pose_i,
                        sample_points=geometry_i.get('sample_points'),
                        bbox_corners=geometry_i.get('bbox_corners'),
                        box_color=(0, 255, 0),
                        point_color=(0, 220, 255),
                    )
                input_box_crop = None
                if hasattr(stage_output, 'boxes_rend') and hasattr(stage_output, 'boxes_crop'):
                    input_box_crop = _megapose_map_box_to_crop(
                        stage_output.boxes_rend[local_idx],
                        stage_output.boxes_crop[local_idx],
                        observed_bgr.shape,
                    )
                if input_box_crop is not None:
                    observed_bgr = _megapose_draw_2d_box(observed_bgr, input_box_crop, color=(0, 165, 255), thickness=2)
                overlay = _megapose_make_overlay(observed_base, rendered_rgb, render_mask=render_mask, contour_color=(255, 0, 255))
                if geometry_i is not None and input_pose_i is not None:
                    overlay = _megapose_draw_projected_geometry(
                        overlay,
                        stage_output.K_crop[local_idx],
                        input_pose_i,
                        sample_points=None,
                        bbox_corners=geometry_i.get('bbox_corners'),
                        box_color=(0, 165, 255),
                        point_color=(0, 165, 255),
                    )
                if geometry_i is not None:
                    overlay = _megapose_draw_projected_geometry(
                        overlay,
                        stage_output.K_crop[local_idx],
                        output_pose_i,
                        sample_points=geometry_i.get('sample_points'),
                        bbox_corners=geometry_i.get('bbox_corners'),
                        box_color=(0, 255, 0),
                        point_color=(0, 220, 255),
                    )
                stage_panel = _megapose_hstack_images([
                    _megapose_add_title(observed_bgr, 'Observed'),
                    _megapose_add_title(render_panel, 'Render'),
                    _megapose_add_title(overlay, 'Overlay'),
                ])
                stage_panel = _megapose_add_legend(stage_panel, [
                    ((0, 165, 255), 'Input box'),
                    ((0, 255, 0), 'Refined box'),
                    ((255, 0, 255), 'Render contour'),
                    ((0, 220, 255), 'Vertices'),
                ])
                stage_tiles.append(_megapose_add_title(stage_panel, f'Pose {pose_counter} | Refine {stage_id}'))
            if len(stage_tiles) > 0:
                rows.append(_megapose_hstack_images(stage_tiles))
                pose_counter += 1
    return _megapose_vstack_images(rows)


def _run_megapose_job(input_json_path, input_npz_path, output_npz_path):
    '''
    ---
    ---
    Execute one standalone MegaPose subprocess job.
    ---
    ---
    This function is the worker-side entry used by the refinement group. It
    loads packed job inputs, constructs the official MegaPose runtime objects,
    runs coarse/refiner or refiner-only inference, and stores compact outputs
    back to disk.

    Args:
        - input_json_path: Metadata json path for the job.
        - input_npz_path: Packed numeric inputs path for the job.
        - output_npz_path: Output npz path written by the worker.

    Returns:
        - None
    ---
    ---
    执行一个独立的 MegaPose 子进程 job。
    ---
    ---
    该函数是 refinement group 使用的 worker 侧入口。它会加载打包好的输入，
    构造官方 MegaPose 运行对象，执行 coarse/refiner 或仅 refiner 推理，
    并把紧凑输出结果重新写回磁盘。

    参数:
        - input_json_path: job 对应的元信息 json 路径。
        - input_npz_path: job 对应的数值输入 npz 路径。
        - output_npz_path: worker 写出的输出 npz 路径。

    返回:
        - None
    '''
    current_file = Path(__file__).resolve()
    current_dir = str(current_file.parent)
    if sys.path and sys.path[0] == current_dir:
        sys.path.pop(0)
    elif current_dir in sys.path:
        sys.path.remove(current_dir)
    project_root = current_file.parents[1]
    megapose_src = str((project_root / 'third_party_megapose6d' / 'src').resolve())
    bop_toolkit_root = project_root / 'bop_toolkit'
    bop_lib = bop_toolkit_root / 'bop_toolkit_lib'
    if not bop_toolkit_root.is_dir() or not bop_lib.is_dir():
        raise FileNotFoundError(
            'Missing bop_toolkit at %s (expected bop_toolkit_lib inside). '
            'Place the toolkit at the HCCEPose project root; automatic clone is disabled.\n'
            '未找到 bop_toolkit：请在项目根自备含 bop_toolkit_lib 的目录。' % (bop_toolkit_root,)
        )
    bop_toolkit_dir = str(bop_toolkit_root.resolve())
    if megapose_src not in sys.path:
        sys.path.insert(0, megapose_src)
    if bop_toolkit_dir not in sys.path:
        sys.path.insert(0, bop_toolkit_dir)

    # megapose.config reads MEGAPOSE_DATA_DIR at import time; parent usually sets it.
    if not os.environ.get('MEGAPOSE_DATA_DIR'):
        os.environ['MEGAPOSE_DATA_DIR'] = os.fspath(_megapose_resolved_local_data(project_root))

    import pandas as pd
    import torch
    import trimesh
    from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
    from megapose.inference.pose_estimator import PoseEstimator
    from megapose.inference.types import ObservationTensor
    from megapose.inference.utils import load_pose_models
    from megapose.utils.tensor_collection import PandasTensorCollection

    named_models = {
        'megapose-1.0-RGB': {
            'coarse_run_id': 'coarse-rgb-906902141',
            'refiner_run_id': 'refiner-rgb-653307694',
            'requires_depth': False,
            'inference_parameters': {'n_refiner_iterations': 5, 'n_pose_hypotheses': 1},
        },
        'megapose-1.0-RGBD': {
            'coarse_run_id': 'coarse-rgb-906902141',
            'refiner_run_id': 'refiner-rgbd-288182519',
            'requires_depth': True,
            'inference_parameters': {'n_refiner_iterations': 5, 'n_pose_hypotheses': 1},
        },
        'megapose-1.0-RGB-multi-hypothesis': {
            'coarse_run_id': 'coarse-rgb-906902141',
            'refiner_run_id': 'refiner-rgb-653307694',
            'requires_depth': False,
            'inference_parameters': {'n_refiner_iterations': 5, 'n_pose_hypotheses': 5},
        },
    }

    meta = json.loads(Path(input_json_path).read_text())
    data = np.load(input_npz_path, allow_pickle=False)

    rgb = data['rgb'].astype(np.uint8)
    K = data['K'].astype(np.float32)
    bboxes_xyxy = data['bboxes_xyxy'].astype(np.float32)
    has_depth = bool(meta['has_depth'])
    has_initial_poses = bool(meta.get('has_initial_poses', False))
    depth = data['depth'].astype(np.float32) if has_depth else None
    initial_poses = data['initial_poses'].astype(np.float32) if has_initial_poses else None
    vis_enabled = bool(meta.get('vis_enabled', False))
    vis_stages = meta.get('vis_stages', None)
    vis_max_items = int(meta.get('vis_max_items', 4))

    mesh_cache_dir = Path(output_npz_path).parent / 'meshes'
    mesh_cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_mesh_path(mesh_path):
        mesh_path = Path(mesh_path)
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if len(mesh.vertices) >= 2000 or len(mesh.faces) == 0:
            return mesh_path

        mesh_dense = mesh.copy()
        max_subdivide = 6
        while len(mesh_dense.vertices) < 2000 and max_subdivide > 0:
            mesh_dense = mesh_dense.subdivide()
            max_subdivide -= 1
        dense_path = mesh_cache_dir / f'{mesh_path.stem}_megapose_dense.ply'
        mesh_dense.export(dense_path)
        return dense_path

    rigid_objects = []
    for record in meta['objects']:
        rigid_objects.append(
            RigidObject(
                label=record['label'],
                mesh_path=prepare_mesh_path(record['mesh_path']),
                mesh_units=record.get('mesh_units', 'mm'),
            )
        )
    object_dataset = RigidObjectDataset(rigid_objects)

    vis_geometry = {}
    for record in meta['objects']:
        mesh_vis = trimesh.load(record['mesh_path'], force='mesh', process=False)
        if isinstance(mesh_vis, trimesh.Scene):
            mesh_vis = mesh_vis.dump(concatenate=True)
        vertices = np.asarray(mesh_vis.vertices, dtype=np.float32) * _megapose_unit_scale(record.get('mesh_units', 'mm'))
        if vertices.shape[0] == 0:
            continue
        min_xyz = vertices.min(axis=0)
        max_xyz = vertices.max(axis=0)
        bbox_corners = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
        ], dtype=np.float32)
        sample_count = min(48, vertices.shape[0])
        sample_ids = np.linspace(0, vertices.shape[0] - 1, sample_count).astype(np.int64)
        sample_points = vertices[sample_ids]
        vis_geometry[record['label']] = {
            'bbox_corners': bbox_corners,
            'sample_points': sample_points,
        }

    model_name = meta['model_name']
    if model_name not in named_models:
        raise KeyError('Unsupported MegaPose model name: %s' % model_name)
    model_info = named_models[model_name]
    if model_info['requires_depth'] and depth is None:
        raise ValueError('Selected MegaPose model requires depth input.')

    renderer_kwargs = {'preload_cache': False, 'split_objects': False, 'n_workers': 1}
    _ensure_megapose_model_artifacts(
        project_root,
        model_info['coarse_run_id'],
        model_info['refiner_run_id'],
    )
    models_root = Path(os.environ['MEGAPOSE_DATA_DIR']).resolve() / 'megapose-models'
    coarse_model, refiner_model, _ = load_pose_models(
        coarse_run_id=model_info['coarse_run_id'],
        refiner_run_id=model_info['refiner_run_id'],
        object_dataset=object_dataset,
        force_panda3d_renderer=True,
        renderer_kwargs=renderer_kwargs,
        models_root=models_root,
    )
    pose_estimator = PoseEstimator(
        refiner_model=refiner_model,
        coarse_model=coarse_model,
        detector_model=None,
        depth_refiner=None,
        bsz_objects=8,
        bsz_images=128,
    ).cuda()

    observation = ObservationTensor.from_numpy(rgb, depth if model_info['requires_depth'] else None, K).cuda()
    infos = pd.DataFrame(dict(
        label=list(meta['labels']),
        batch_im_id=0,
        instance_id=np.arange(len(meta['labels'])),
    ))

    vis_image = None
    if has_initial_poses:
        coarse_estimates = PandasTensorCollection(
            infos=infos.copy(),
            poses=torch.as_tensor(initial_poses).float(),
        ).cuda()
        preds, refiner_extra = pose_estimator.forward_refiner(
            observation,
            coarse_estimates,
            n_iterations=model_info['inference_parameters']['n_refiner_iterations'],
            keep_all_outputs=vis_enabled,
        )
        predictions = preds[f"iteration={model_info['inference_parameters']['n_refiner_iterations']}"]
        if vis_enabled:
            vis_image = _megapose_build_refinement_visualization(
                refiner_extra.get('outputs', []),
                vis_geometry=vis_geometry,
                vis_stages=vis_stages,
                vis_max_items=vis_max_items,
            )
    else:
        detections = PandasTensorCollection(infos=infos, bboxes=torch.as_tensor(bboxes_xyxy).float()).cuda()
        predictions, _ = pose_estimator.run_inference_pipeline(
            observation,
            detections=detections,
            **model_info['inference_parameters']
        )

    infos = predictions.infos.sort_values('instance_id').reset_index(drop=True)
    poses = predictions.poses.cpu().numpy().astype(np.float32)
    order = infos['instance_id'].to_numpy(dtype=np.int64)
    if has_initial_poses:
        scores = np.zeros((poses.shape[0],), dtype=np.float32)
    elif 'pose_score' in infos:
        scores = infos['pose_score'].to_numpy(dtype=np.float32)
    elif 'pose_logit' in infos:
        scores = infos['pose_logit'].to_numpy(dtype=np.float32)
    else:
        scores = np.zeros((poses.shape[0],), dtype=np.float32)
    pose_order = np.argsort(order)
    poses = poses[pose_order]
    scores = scores[pose_order]
    valid_ids = np.sort(order).astype(np.int64)
    save_dict = {
        'poses': poses,
        'scores': scores,
        'valid_ids': valid_ids,
    }
    if vis_image is not None:
        # Visualization is assembled in BGR (tensor panels converted before cv2 drawing).
        save_dict['vis_image'] = vis_image.astype(np.uint8)
    np.savez_compressed(output_npz_path, **save_dict)


if __name__ == '__main__':
    if '--megapose-job' in sys.argv:
        input_json = sys.argv[sys.argv.index('--input-json') + 1]
        input_npz = sys.argv[sys.argv.index('--input-npz') + 1]
        output_npz = sys.argv[sys.argv.index('--output-npz') + 1]
        _run_megapose_job(input_json, input_npz, output_npz)
