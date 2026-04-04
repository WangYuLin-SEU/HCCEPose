# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import cv2, os, sys
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.test_script_utils import print_stage_time_breakdown, save_visual_artifacts
from HccePose.tester import Tester


def get_output_prefix(file_name, hccepose_acceleration):
    if hccepose_acceleration == 'pytorch':
        return file_name.replace('.jpg', '')
    return file_name.replace('.jpg', f'_{hccepose_acceleration}')


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    test_img_path = os.path.join(current_dir, 'test_imgs')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    CUDA_DEVICE = '0'
    hccepose_vis = True
    save_visualizations = hccepose_vis
    print_stage_timing = False

    hccepose_acceleration = 'pytorch'

    Tester_item = Tester(
        bop_dataset_item,
        hccepose_vis=hccepose_vis,
        CUDA_DEVICE=CUDA_DEVICE,
        hccepose_acceleration=hccepose_acceleration,
    )
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
        results_dict = Tester_item.predict(
            cam_K,
            image,
            [obj_id],
            conf=0.85,
            confidence_threshold=0.85,
        )
        print_stage_time_breakdown(results_dict, enabled=print_stage_timing, prefix=name)
        output_prefix = get_output_prefix(file_name, hccepose_acceleration)
        save_visual_artifacts([
            (output_prefix + '_show_2d.jpg', results_dict.get('show_2D_results')),
            (output_prefix + '_show_6d_vis0.jpg', results_dict.get('show_6D_vis0')),
            (output_prefix + '_show_6d_vis1.jpg', results_dict.get('show_6D_vis1')),
            (output_prefix + '_show_6d_vis2.jpg', results_dict.get('show_6D_vis2')),
        ], enabled=save_visualizations)
