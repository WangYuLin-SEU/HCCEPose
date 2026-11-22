# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China


"""Read public RGB-D frames for the optional MegaPose example.
为可选 MegaPose 示例读取公开 RGB-D 帧。
"""
import json
import os

import cv2
import numpy as np


def convert_depth_to_meter(depth_raw, depth_scale_to_m=1e-3):
    '''
    ---
    ---
    Convert a raw saved depth map to meter units.
    ---
    ---
    Set the scale explicitly: 1 for meters, 1e-3 for millimeters (the public
    capture default), or 1e-4 for tenths of a millimeter. Distance is not a unit.

    Args:
        - depth_raw: Raw depth image read from disk.
        - depth_scale_to_m: Positive conversion factor from raw values to meters.

    Returns:
        - depth_m: Depth image converted to meters.
    ---
    ---
    将原始保存的深度图转换到米单位。
    ---
    ---
    显式指定换算系数：米为 1，毫米为 1e-3（公开采集样例默认值），
    十分之一毫米为 1e-4。不能根据场景距离推断深度单位。

    参数:
        - depth_raw: 从磁盘读取的原始深度图。
        - depth_scale_to_m: 原始数值换算到米的正系数。

    返回:
        - depth_m: 已转换为米单位的深度图。
    '''
    if not np.isfinite(depth_scale_to_m) or depth_scale_to_m <= 0:
        raise ValueError('depth_scale_to_m must be finite and positive')
    depth_m = np.asarray(depth_raw, dtype=np.float32).copy()
    if not np.isfinite(depth_m).all() or np.any(depth_m < 0):
        raise ValueError('Raw depth must be finite and nonnegative')
    depth_m *= depth_scale_to_m
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

def load_capture_frame(capture_dir, name, depth_scale_to_m=1e-3):
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
        - depth_scale_to_m: Raw-depth-to-meter factor; public frames use millimeters.

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
        - depth_scale_to_m: 原始深度到米的系数；公开帧使用毫米。

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

    depth_m = convert_depth_to_meter(depth, depth_scale_to_m)
    return image, depth, depth_m, cam_K
