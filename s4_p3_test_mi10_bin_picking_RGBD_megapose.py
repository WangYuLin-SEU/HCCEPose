# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import os
import sys

# Resolve project-local modules from this script before importing them.
# 导入前从当前脚本定位项目内模块，不依赖启动目录。
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from HccePose.ensure_rgbd_megapose_demo import prepare_rgbd_megapose_demo_environment

CUDA_DEVICE = '0'

# Prepare optional dependencies only when executing this recipe, before loading them.
# 仅执行本示例时先准备可选依赖，再进行顶部模块导入；普通导入不会安装环境。
if __name__ == '__main__':
    os.chdir(current_dir)
    prepare_rgbd_megapose_demo_environment(cuda_device=CUDA_DEVICE, project_root=current_dir)

import cv2

from HccePose.bop_loader import bop_dataset
from HccePose.test_script_utils import (
    print_stage_time_breakdown,
    save_visual_artifacts,
)
from HccePose.tester import Tester
from Refinement.refinement_group import list_capture_frame_names, load_capture_frame


if __name__ == '__main__':
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    # Keep each RGB-D frame and its calibration together under sample_data.
    # 每帧 RGB-D 与对应标定一起存放在 sample_data 下。
    capture_dir = os.path.join(current_dir, 'sample_data', 'rgbd')
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
