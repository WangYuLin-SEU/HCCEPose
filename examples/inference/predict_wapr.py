"""Estimate poses and scores from HccePose masks predicted for the current image.
使用 HccePose 从当前图像预测的掩码，通过 WAPR 估计位姿并评分。
"""
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
# Keep imports stable when launched outside the repository.
# 从仓库外部启动时也从当前脚本定位项目库。
sys.path.insert(0, project_dir)


def main(dataset_root, image_path, depth_path, intrinsics_path, object_ids,
         output_path, depth_scale_to_m, device='0'):
    """Depth must already be registered to RGB; its unit conversion is explicit.
    深度须预先与彩色图配准；单位换算由调用方显式指定。
    """
    import cv2
    import numpy as np
    import trimesh
    from HccePose.bop_loader import bop_dataset
    from HccePose.tester import Tester
    from Refinement.wapr_public import estimate_from_masks, restore_full_frame_masks
    from examples.inference._inputs import checked_object_ids, read_intrinsics

    if not np.isfinite(depth_scale_to_m) or depth_scale_to_m <= 0:
        raise ValueError('depth_scale_to_m must be finite and positive')
    for path in (dataset_root, image_path, depth_path, intrinsics_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    image = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if image is None or depth is None or depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValueError('RGB and scalar depth must decode at identical aligned dimensions')
    depth_m = depth.astype(np.float32) * depth_scale_to_m
    cam_K = read_intrinsics(intrinsics_path)
    dataset = bop_dataset(dataset_root)
    object_ids = checked_object_ids(object_ids, dataset.obj_id_list)
    tester = Tester(dataset, CUDA_DEVICE=device, hccepose_vis=False)
    predictions = tester.predict(cam_K, image, object_ids)
    instances = []
    identifiers = []
    for obj_id in object_ids:
        if obj_id not in predictions:
            continue
        index = dataset.obj_id_list.index(obj_id)
        mesh = trimesh.load(dataset.obj_model_list[index], force='mesh', process=False)
        # Published BOP models use millimeters; WAPR requires meter vertices/diameter.
        # 公开 BOP 模型使用毫米；WAPR 接口需要米制顶点和直径。
        mesh.apply_scale(1e-3)
        diameter_m = float(dataset.model_info[str(obj_id)]['diameter']) * 1e-3
        masks = restore_full_frame_masks(predictions[obj_id], image.shape)
        for instance_index, mask in enumerate(masks):
            instances.append({'mesh': mesh, 'diameter_m': diameter_m, 'mask': mask})
            identifiers.append({'object_id': int(obj_id), 'instance_index': instance_index})
    estimator = None
    if instances:
        from wapr.estimator import WAPREstimator
        estimator = WAPREstimator(device='cuda:' + str(device))
    results = estimate_from_masks(estimator, image, depth_m, cam_K, instances)
    records = []
    for identity, result in zip(identifiers, results):
        records.append({**identity, 'pose_4x4_m': np.asarray(result['pose_4x4']).tolist(),
                        'score_6d': float(result['score_6d'])})
    # Save poses (object to camera, translation in meters) and scores for this input.
    # 保存本次输入的位姿（物体到相机，平移单位为米）与评分。
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(records, handle, indent=2, allow_nan=False)
    return records


if __name__ == '__main__':
    # Example conversion for an integer millimeter depth image; check your sensor units.
    # 示例使用整数毫米深度图；请核对自己传感器的实际单位。
    data_root = os.path.join(project_dir, 'data')
    main(os.path.join(data_root, 'demo-bin-picking'), os.path.join(data_root, 'test_imgs_RGBD', '000003_rgb.png'),
         os.path.join(data_root, 'test_imgs_RGBD', '000003_depth.png'), os.path.join(data_root, 'test_imgs_RGBD', '000003_camera_matrix.json'),
         [1], os.path.join(project_dir, 'outputs', 'wapr.json'), depth_scale_to_m=1e-3)
