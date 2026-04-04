# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import cv2, os, sys
from HccePose.bop_loader import bop_dataset
from HccePose.test_script_utils import print_stage_time_breakdown, save_visual_artifacts
from HccePose.tester import Tester
from Refinement.refinement_test_utils import load_capture_frame, list_capture_frame_names


if __name__ == '__main__':

    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    capture_dir = os.path.join(current_dir, 'test_imgs_RGBD')
    foundationpose_refine_dir = os.path.join(current_dir, '2023-10-28-18-33-37')
    foundationpose_score_dir = os.path.join(current_dir, '2024-01-11-20-02-45')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    CUDA_DEVICE = '0'
    hccepose_vis = True
    foundationpose_vis = True
    foundationpose_vis_stages = [1, 2, 3, 4, 5, 'score']
    hccepose_acceleration = 'pytorch'
    foundationpose_acceleration = 'pytorch'
    save_visualizations = hccepose_vis or foundationpose_vis
    print_stage_timing = False

    tester_item = Tester(
        bop_dataset_item,
        hccepose_vis=hccepose_vis,
        CUDA_DEVICE=CUDA_DEVICE,
        foundationpose_refine_dir=foundationpose_refine_dir,
        foundationpose_score_dir=foundationpose_score_dir,
        hccepose_acceleration=hccepose_acceleration,
        foundationpose_acceleration=foundationpose_acceleration,
    )
    frame_names = list_capture_frame_names(capture_dir)
    for name in frame_names:
        image, depth, depth_m, cam_K = load_capture_frame(capture_dir, name)

        results_dict = tester_item.predict(
            cam_K,
            image,
            [obj_id],
            conf=0.85,
            confidence_threshold=0.85,
            depth=depth_m,
            use_foundationpose=True,
            foundationpose_vis=foundationpose_vis,
            foundationpose_vis_stages=foundationpose_vis_stages,
        )
        print_stage_time_breakdown(results_dict, enabled=print_stage_timing, prefix=name)
        save_visual_artifacts([
            (os.path.join(capture_dir, '%s_show_2d.jpg' % name), results_dict.get('show_2D_results')),
            (os.path.join(capture_dir, '%s_show_6d_vis0.jpg' % name), results_dict.get('show_6D_vis0')),
            (os.path.join(capture_dir, '%s_show_6d_vis1.jpg' % name), results_dict.get('show_6D_vis1')),
            (os.path.join(capture_dir, '%s_show_6d_vis2.jpg' % name), results_dict.get('show_6D_vis2')),
            (os.path.join(capture_dir, '%s_show_foundationpose.jpg' % name), results_dict.get('show_foundationpose')),
        ], enabled=save_visualizations)
