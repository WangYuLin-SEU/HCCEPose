# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os
import sys


if __name__ == '__main__':

    # Repo root: robust first run (MegaPose setup, demo assets) regardless of shell cwd.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    sys.path.insert(0, current_dir)

    CUDA_DEVICE = '0'

    from HccePose.ensure_rgbd_megapose_demo import prepare_rgbd_megapose_demo_environment

    prepare_rgbd_megapose_demo_environment(cuda_device=CUDA_DEVICE, project_root=current_dir)

    import cv2
    from HccePose.bop_loader import bop_dataset
    from HccePose.test_script_utils import print_stage_time_breakdown, save_visual_artifacts
    from HccePose.tester import Tester
    from Refinement.refinement_test_utils import load_capture_frame, list_capture_frame_names

    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    capture_dir = os.path.join(current_dir, 'test_imgs_RGBD')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    hccepose_vis = True
    hccepose_acceleration = 'pytorch'
    megapose_use_depth = True
    megapose_vis = True
    megapose_vis_stages = [1, 2, 3, 4, 5]
    megapose_variant_name = 'rgbd' if megapose_use_depth else 'rgb'
    save_visualizations = hccepose_vis or megapose_vis
    print_stage_timing = False

    tester_item = Tester(
        bop_dataset_item,
        hccepose_vis=hccepose_vis,
        CUDA_DEVICE=CUDA_DEVICE,
        hccepose_acceleration=hccepose_acceleration,
    )

    frame_names = list_capture_frame_names(capture_dir)
    for name in frame_names:
        image, depth, depth_m, cam_K = load_capture_frame(capture_dir, name)
        megapose_depth = depth_m if megapose_use_depth else None

        results_mp = tester_item.predict(
            cam_K,
            image,
            [obj_id],
            conf=0.85,
            confidence_threshold=0.85,
            depth=megapose_depth,
            use_megapose=True,
            megapose_vis=megapose_vis,
            megapose_vis_stages=megapose_vis_stages,
        )
        print_stage_time_breakdown(results_mp, enabled=print_stage_timing, prefix='%s | MegaPose %s' % (name, megapose_variant_name.upper()))

        save_visual_artifacts([
            (os.path.join(capture_dir, '%s_show_megapose.jpg' % name), results_mp.get('show_megapose')),
            (os.path.join(capture_dir, '%s_megapose_%s_show_2d.jpg' % (name, megapose_variant_name)), results_mp.get('show_2D_results')),
            (os.path.join(capture_dir, '%s_megapose_%s_show_6d_vis0.jpg' % (name, megapose_variant_name)), results_mp.get('show_6D_vis0')),
            (os.path.join(capture_dir, '%s_megapose_%s_show_6d_vis1.jpg' % (name, megapose_variant_name)), results_mp.get('show_6D_vis1')),
            (os.path.join(capture_dir, '%s_megapose_%s_show_6d_vis2.jpg' % (name, megapose_variant_name)), results_mp.get('show_6D_vis2')),
        ], enabled=save_visualizations)
