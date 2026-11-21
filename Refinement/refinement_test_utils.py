# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China


"""Read public RGB-D frames for the optional MegaPose example.
为可选 MegaPose 示例读取公开 RGB-D 帧。
"""
import json
import os

import cv2
import numpy as np


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
