# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

"""Estimate RGB-D poses with public WAPR, then filter scores and rendered-mask overlaps.
使用公开 WAPR 估计 RGB-D 位姿，随后自动按评分与渲染掩码重叠筛选。
"""

import json
import os
import sys
from numbers import Integral

import cv2
import numpy as np

# Resolve the project from this script, including launches from another directory.
# 根据当前脚本定位项目，支持从其他目录启动。
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from HccePose.bop_loader import bop_dataset, inout
from HccePose.tester import Tester
from Refinement import refinement_group


if __name__ == '__main__':
    # Keep application thresholds explicit; WAPR scores are not probabilities.
    # 应用筛选门槛显式保留；WAPR 分数不解释为概率。
    score_threshold = 0.75
    nms_iou_threshold = 0.5
    # estimate: WAPR candidates only; hybrid: add PnP; refine: refine PnP only.
    # estimate：只用 WAPR 候选；hybrid：加入 PnP；refine：只精修 PnP 初值。
    wapr_mode = 'estimate'
    pnp_op = 'ransac+comb'
    device = '0'
    object_ids = [1]
    dataset_root = os.path.join(current_dir, 'data', 'demo-bin-picking')
    capture_dir = os.path.join(current_dir, 'sample_data', 'rgbd')
    frame_name = '000003'
    image_path = os.path.join(capture_dir, frame_name + '_rgb.png')
    depth_path = os.path.join(capture_dir, frame_name + '_depth.png')
    intrinsics_path = os.path.join(capture_dir, frame_name + '_camK.json')
    output_path = os.path.join(current_dir, 'outputs', 'wapr.json')
    # This example stores depth in millimeters; WAPR expects meters.
    # 本示例原始深度单位为毫米，WAPR 输入使用米。
    depth_scale_to_m = 1e-3

    if not np.isfinite(depth_scale_to_m) or depth_scale_to_m <= 0:
        raise ValueError('depth_scale_to_m must be finite and positive')
    if wapr_mode not in ('estimate', 'hybrid', 'refine'):
        raise ValueError('WAPR mode must be estimate, hybrid or refine')
    if not np.isfinite(score_threshold) or not np.isfinite(nms_iou_threshold) or not 0 <= nms_iou_threshold <= 1:
        raise ValueError('Use a finite score threshold and an IoU threshold in [0,1]')
    for path in (dataset_root, image_path, depth_path, intrinsics_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    filtered_path = os.path.splitext(output_path)[0] + '_filtered.json'
    same_file = os.path.realpath(output_path) == os.path.realpath(filtered_path)
    if os.path.exists(output_path) and os.path.exists(filtered_path):
        same_file = same_file or os.path.samefile(output_path, filtered_path)
    if same_file:
        raise ValueError('Choose separate output files; raw input must not be overwritten')
    # Protect source images and calibration, including symlink and hardlink aliases.
    # 保护原始图像及标定，输出文件的软链接、硬链接也不能指向这些输入。
    for destination in (output_path, filtered_path):
        for input_path in (image_path, depth_path, intrinsics_path):
            same_input = os.path.realpath(destination) == os.path.realpath(input_path)
            if os.path.exists(destination):
                same_input = same_input or os.path.samefile(destination, input_path)
            if same_input:
                raise ValueError(f'Output file must not overwrite input: {destination} -> {input_path}')
    image = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if image is None or depth is None or depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValueError('RGB and scalar depth must decode at identical aligned dimensions')
    depth_m = refinement_group.convert_depth_to_meter(depth, depth_scale_to_m)
    camera = inout.load_json(intrinsics_path)
    cam_K = inout.load_cam_params(intrinsics_path)['K'] if isinstance(camera, dict) else np.asarray(camera, dtype=np.float32)
    if cam_K.shape != (3, 3) or not np.isfinite(cam_K).all() or cam_K[0, 0] <= 0 or cam_K[1, 1] <= 0:
        raise ValueError('Intrinsics must be a finite 3 x 3 matrix with positive focal lengths')
    dataset = bop_dataset(dataset_root)
    requested_ids = []
    for object_id in object_ids:
        if isinstance(object_id, bool) or not isinstance(object_id, Integral):
            raise ValueError(f'Expected an integer object ID, received {object_id!r}')
        if object_id not in dataset.obj_id_list:
            raise ValueError(f'Unknown object ID {object_id} in this dataset')
        if object_id in requested_ids:
            raise ValueError(f'Duplicate object ID {object_id} in the request')
        requested_ids.append(int(object_id))
    tester = Tester(dataset, CUDA_DEVICE=device, hccepose_vis=False)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        predictions = tester.predict(cam_K, image, requested_ids, depth=depth_m,
                                     use_wapr=True, wbps_threshold=score_threshold,
                                     wapr_nms_iou_threshold=nms_iou_threshold,
                                     wapr_mode=wapr_mode, pnp_op=pnp_op)
    finally:
        # Preserve raw estimates even if the subsequent rendering/NMS backend fails.
        # 即使后续渲染或 NMS 后端失败，也单独保存已生成的原始估计。
        if tester.last_wapr_raw_records is not None:
            with open(output_path, 'w', encoding='utf-8') as handle:
                json.dump(tester.last_wapr_raw_records, handle, indent=2, allow_nan=False)
    report = predictions['wapr_postprocess']
    records = report['raw_records']
    with open(filtered_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"WAPR {wapr_mode}: {len(records)} raw, {len(report['retained_records'])} kept; "
          f"score >= {score_threshold}, mask IoU > {nms_iou_threshold}")
    print('Raw:', output_path)
    print('Filtered:', filtered_path)
