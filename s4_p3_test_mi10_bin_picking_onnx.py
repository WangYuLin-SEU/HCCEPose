# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import json
import os
import sys


def add_title(image, title, height=28):
    import cv2
    import numpy as np

    header = np.full((height, image.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, str(title), (8, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def hstack_images(images):
    import cv2
    import numpy as np

    images = [image for image in images if image is not None]
    if len(images) == 0:
        return None
    max_h = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        pad_h = max_h - image.shape[0]
        padded.append(cv2.copyMakeBorder(image, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18)))
    if len(padded) == 1:
        return padded[0]
    spacer = np.full((max_h, 8, 3), 18, dtype=np.uint8)
    canvas = padded[0]
    for image in padded[1:]:
        canvas = np.concatenate([canvas, spacer, image], axis=1)
    return canvas


def rotation_error_deg(R_a, R_b):
    import numpy as np

    rel = R_a @ R_b.T
    trace = float(np.trace(rel))
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def build_comparison(results_pytorch, results_onnx, obj_id):
    import numpy as np
    import torch

    summary = {
        'obj_id': int(obj_id),
        'pytorch_time_s': float(results_pytorch.get('time', 0.0)),
        'onnx_time_s': float(results_onnx.get('time', 0.0)),
    }
    pred_pytorch = results_pytorch.get(obj_id, {})
    pred_onnx = results_onnx.get(obj_id, {})
    if not isinstance(pred_pytorch, dict) or not isinstance(pred_onnx, dict):
        summary['status'] = 'missing_obj_result'
        return summary

    if 'pred_mask_logits' in pred_pytorch and 'pred_mask_logits' in pred_onnx:
        mask_logits_diff = torch.abs(pred_pytorch['pred_mask_logits'] - pred_onnx['pred_mask_logits'])
        summary['mask_logits_mae'] = float(mask_logits_diff.mean().item())
        summary['mask_logits_max'] = float(mask_logits_diff.max().item())
    if 'pred_front_back_code_logits' in pred_pytorch and 'pred_front_back_code_logits' in pred_onnx:
        code_logits_diff = torch.abs(pred_pytorch['pred_front_back_code_logits'] - pred_onnx['pred_front_back_code_logits'])
        summary['code_logits_mae'] = float(code_logits_diff.mean().item())
        summary['code_logits_max'] = float(code_logits_diff.max().item())

    if 'pred_mask' in pred_pytorch and 'pred_mask' in pred_onnx:
        mask_binary_match = (pred_pytorch['pred_mask'] == pred_onnx['pred_mask']).to(torch.float32)
        summary['mask_binary_match_ratio'] = float(mask_binary_match.mean().item())

    Rts_pytorch = np.asarray(pred_pytorch.get('Rts', []), dtype=np.float32)
    Rts_onnx = np.asarray(pred_onnx.get('Rts', []), dtype=np.float32)
    pose_count = int(min(len(Rts_pytorch), len(Rts_onnx)))
    summary['pose_count_compared'] = pose_count
    pose_diffs = []
    for pose_id in range(pose_count):
        Rt_pt = Rts_pytorch[pose_id]
        Rt_ox = Rts_onnx[pose_id]
        trans_mm = float(np.linalg.norm(Rt_pt[:3, 3] - Rt_ox[:3, 3]))
        rot_deg = rotation_error_deg(Rt_pt[:3, :3], Rt_ox[:3, :3])
        pose_diffs.append({
            'pose_id': pose_id,
            'translation_diff_mm': trans_mm,
            'rotation_diff_deg': rot_deg,
        })
    summary['pose_diffs'] = pose_diffs
    if pose_diffs:
        summary['translation_diff_mm_mean'] = float(np.mean([item['translation_diff_mm'] for item in pose_diffs]))
        summary['rotation_diff_deg_mean'] = float(np.mean([item['rotation_diff_deg'] for item in pose_diffs]))
    return summary


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    sys.path.insert(0, current_dir)

    import cv2
    import numpy as np
    import torch
    from HccePose.bop_loader import bop_dataset
    from HccePose.test_script_utils import print_stage_time_breakdown, save_visual_artifacts
    from HccePose.tester import Tester

    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    test_img_path = os.path.join(current_dir, 'test_imgs')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    CUDA_DEVICE = '0'
    hccepose_vis = True
    save_visualizations = hccepose_vis
    print_stage_timing = False

    hccepose_acceleration_pytorch = 'pytorch'
    hccepose_acceleration_onnx = 'onnx'
    pytorch_tester = Tester(bop_dataset_item, hccepose_vis=hccepose_vis, CUDA_DEVICE=CUDA_DEVICE, hccepose_acceleration=hccepose_acceleration_pytorch)
    onnx_tester = Tester(bop_dataset_item, hccepose_vis=hccepose_vis, CUDA_DEVICE=CUDA_DEVICE, hccepose_acceleration=hccepose_acceleration_onnx)

    for name in ['IMG_20251007_165718',
                 'IMG_20251007_165725',
                 'IMG_20251007_165800',
                 'IMG_20251007_170223',
                 'IMG_20251007_170230',
                 'IMG_20251007_170240',
                 'IMG_20251007_170250']:
        file_name = os.path.join(test_img_path, '%s.jpg' % name)
        image = cv2.imread(file_name)
        if image is None:
            print('skip missing image:', file_name)
            continue
        cam_K = np.array([
            [2.83925618e+03, 0.00000000e+00, 2.02288638e+03],
            [0.00000000e+00, 2.84037288e+03, 1.53940473e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])

        results_pytorch = pytorch_tester.predict(cam_K, image, [obj_id], conf=0.85, confidence_threshold=0.85)
        results_onnx = onnx_tester.predict(cam_K, image, [obj_id], conf=0.85, confidence_threshold=0.85)
        print_stage_time_breakdown(results_pytorch, enabled=print_stage_timing, prefix='%s | PyTorch' % name)
        print_stage_time_breakdown(results_onnx, enabled=print_stage_timing, prefix='%s | ONNX' % name)

        save_visual_artifacts([
            (file_name.replace('.jpg', '_pytorch_show_2d.jpg'), results_pytorch.get('show_2D_results')),
            (file_name.replace('.jpg', '_pytorch_show_6d_vis0.jpg'), results_pytorch.get('show_6D_vis0')),
            (file_name.replace('.jpg', '_pytorch_show_6d_vis1.jpg'), results_pytorch.get('show_6D_vis1')),
            (file_name.replace('.jpg', '_pytorch_show_6d_vis2.jpg'), results_pytorch.get('show_6D_vis2')),
            (file_name.replace('.jpg', '_onnx_show_2d.jpg'), results_onnx.get('show_2D_results')),
            (file_name.replace('.jpg', '_onnx_show_6d_vis0.jpg'), results_onnx.get('show_6D_vis0')),
            (file_name.replace('.jpg', '_onnx_show_6d_vis1.jpg'), results_onnx.get('show_6D_vis1')),
            (file_name.replace('.jpg', '_onnx_show_6d_vis2.jpg'), results_onnx.get('show_6D_vis2')),
        ], enabled=save_visualizations)

        if save_visualizations:
            compare_vis = hstack_images([
                add_title(results_pytorch.get('show_6D_vis1'), 'PyTorch'),
                add_title(results_onnx.get('show_6D_vis1'), 'ONNX'),
            ])
            save_visual_artifacts([
                (file_name.replace('.jpg', '_compare_pytorch_onnx_6d.jpg'), compare_vis),
            ], enabled=True)

        comparison = build_comparison(results_pytorch, results_onnx, obj_id)
        with open(file_name.replace('.jpg', '_compare_pytorch_onnx.json'), 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
