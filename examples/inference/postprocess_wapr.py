"""Filter saved WAPR scores and run public rendered-mask NMS per object category.
按分数筛选已保存的 WAPR 结果，再调用公开的同类渲染掩码 NMS。
"""
import json
import os
import sys
from numbers import Integral, Real

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
# Resolve project imports from this file, including launches outside the repository.
# 从当前脚本定位项目内库，也支持从仓库外启动。
sys.path.insert(0, project_dir)


def _checked_records(records):
    """Validate raw predict_wapr records without modifying them.
    检查 predict_wapr 的原始记录，不修改调用方输入。
    """
    if not isinstance(records, list):
        raise ValueError('Expected the raw list written by predict_wapr.py')
    checked = []
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'Record {index} must be a dictionary')
        object_id = record.get('object_id')
        instance_index = record.get('instance_index')
        score = record.get('score_6d')
        if isinstance(object_id, bool) or not isinstance(object_id, Integral) or object_id <= 0:
            raise ValueError(f'Record {index} needs a positive integer object_id')
        if isinstance(instance_index, bool) or not isinstance(instance_index, Integral) or instance_index < 0:
            raise ValueError(f'Record {index} needs a nonnegative integer instance_index')
        if isinstance(score, bool) or not isinstance(score, Real) or not np.isfinite(score):
            raise ValueError(f'Record {index} needs a finite numeric score_6d')
        identity = (int(object_id), int(instance_index))
        if identity in seen:
            raise ValueError(f'Duplicate instance identity {identity}')
        seen.add(identity)
        pose = np.asarray(record.get('pose_4x4_m'), dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f'Record {index} needs a finite pose_4x4_m of shape [4,4]')
        if not np.array_equal(pose[3], [0, 0, 0, 1]):
            raise ValueError(f'Record {index} needs an object-to-camera homogeneous pose')
        row = dict(record)
        row.update(object_id=identity[0], instance_index=identity[1],
                   score_6d=float(score), pose_4x4_m=pose.tolist())
        checked.append(row)
    return checked


def filter_records(records, meshes_by_id, cam_K, image_shape, score_threshold,
                   nms_iou_threshold, device='cuda:0'):
    """Keep score >= threshold, then use public CAD-silhouette NMS within each ID.
    先保留分数不低于门槛的结果，再调用公开 CAD 剪影 NMS，仅在同物体编号内去重。

    Mesh vertices and object-to-camera translations are meters; K uses original
    image pixels. Kept records preserve input order. Scores are not probabilities.
    网格顶点及物体到相机平移均为米；K 使用原图像素。保留结果沿用输入顺序，分数不是概率。
    """
    if not np.isfinite(score_threshold):
        raise ValueError('score_threshold must be finite; scores need not lie in [0,1]')
    if not np.isfinite(nms_iou_threshold) or not 0 <= nms_iou_threshold <= 1:
        raise ValueError('nms_iou_threshold must be between zero and one')
    if len(image_shape) < 2 or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in image_shape[:2]):
        raise ValueError('image_shape must start with positive integer height and width')
    height, width = map(int, image_shape[:2])
    cam_K = np.asarray(cam_K, dtype=np.float32)
    if cam_K.shape != (3, 3) or not np.isfinite(cam_K).all() or cam_K[0, 0] <= 0 or cam_K[1, 1] <= 0:
        raise ValueError('cam_K must contain finite 3 x 3 original-image pixel intrinsics')
    rows = _checked_records(records)
    groups = {}
    statuses = ['below_score_threshold'] * len(rows)
    for index, row in enumerate(rows):
        if row['score_6d'] >= score_threshold:
            groups.setdefault(row['object_id'], []).append(index)
            statuses[index] = 'not_retained_by_mask_nms'
    for object_id in groups:
        if object_id not in meshes_by_id:
            raise ValueError(f'Missing meter mesh for object {object_id}')

    # Reuse the fixed public backend; do not implement a second NMS algorithm here.
    # 复用固定公开后端，本示例不另写一套 NMS 算法。
    # Import only when a score-qualified pose actually needs GPU rendering.
    # 只有结果通过分数门槛并需要 GPU 渲染时才导入后端。
    if groups:
        from wapr.suppression import rendered_mask_iou_nms_gpu
        for object_id, indices in groups.items():
            poses = [{'obj_id': object_id, 'score_6d': rows[index]['score_6d'],
                      'pose_4x4': rows[index]['pose_4x4_m'], 'input_index': index}
                     for index in indices]
            kept = rendered_mask_iou_nms_gpu(poses, meshes_by_id[object_id], cam_K,
                                             height, width, nms_iou_threshold, device)
            for pose in kept:
                statuses[pose['input_index']] = 'kept'
    kept_indices = [index for index, status in enumerate(statuses) if status == 'kept']
    decisions = [{'input_index': index, 'object_id': row['object_id'],
                  'instance_index': row['instance_index'], 'score_6d': row['score_6d'],
                  'status': statuses[index]} for index, row in enumerate(rows)]
    return {'score_threshold': float(score_threshold), 'nms_iou_threshold': float(nms_iou_threshold),
            'nms_method': 'same_object_id_rendered_mask_iou', 'raw_records': rows,
            'decisions': decisions, 'kept_input_indices': kept_indices,
            'retained_records': [rows[index] for index in kept_indices]}


def main(dataset_root, image_path, intrinsics_path, input_path, output_path,
         score_threshold, nms_iou_threshold, device='cuda:0', mesh_scale_to_m=1e-3):
    """Postprocess one saved frame using its image, intrinsics and BOP meshes.
    使用同一帧的图像、内参和 BOP 网格，对保存结果做后处理。

    BOP meshes use millimeters by convention; set mesh_scale_to_m=1 for meter meshes.
    BOP 网格按约定使用毫米；若输入网格已是米，将 mesh_scale_to_m 设为 1。
    """
    import cv2
    from examples.inference._inputs import read_intrinsics

    same_file = os.path.realpath(input_path) == os.path.realpath(output_path)
    if os.path.exists(input_path) and os.path.exists(output_path):
        same_file = same_file or os.path.samefile(input_path, output_path)
    if same_file:
        raise ValueError('Choose a separate output file; raw input must not be overwritten')
    if not np.isfinite(score_threshold) or not np.isfinite(nms_iou_threshold) or not 0 <= nms_iou_threshold <= 1:
        raise ValueError('Use a finite score threshold and an IoU threshold in [0,1]')
    if not np.isfinite(mesh_scale_to_m) or mesh_scale_to_m <= 0:
        raise ValueError('mesh_scale_to_m must be finite and positive')
    for path in (dataset_root, image_path, intrinsics_path, input_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Cannot decode the original image')
    cam_K = read_intrinsics(intrinsics_path)
    with open(input_path, encoding='utf-8') as handle:
        records = _checked_records(json.load(handle))
    meshes = {}
    for row in records:
        object_id = row['object_id']
        if row['score_6d'] < score_threshold or object_id in meshes:
            continue
        mesh_path = os.path.join(dataset_root, 'models', f'obj_{object_id:06d}.ply')
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(mesh_path)
        import trimesh
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        mesh.apply_scale(mesh_scale_to_m)
        if mesh.is_empty or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f'Expected a finite triangle mesh at {mesh_path}')
        meshes[object_id] = mesh
    report = filter_records(records, meshes, cam_K, image.shape, score_threshold,
                            nms_iou_threshold, device)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    qualified = sum(row['score_6d'] >= score_threshold for row in records)
    print(f"WAPR postprocessing: {len(records)} raw, {qualified} score-qualified, "
          f"{len(report['retained_records'])} kept; score >= {score_threshold}, mask IoU > {nms_iou_threshold}")
    return report


if __name__ == '__main__':
    # Application choices: WBPS follows this example's acceptance rule, not a probability.
    # 应用参数：WBPS 门槛用于本示例的接收规则，不解释为概率。
    score_threshold = 0.75
    # Same 0.5 IoU as the fixed public WAPR example; tune for your camera and objects.
    # IoU 沿用固定公开 WAPR 示例的 0.5，需按相机与物体调整。
    nms_iou_threshold = 0.5
    device = 'cuda:0'
    data_root = os.path.join(project_dir, 'data')
    main(os.path.join(data_root, 'demo-bin-picking'),
         os.path.join(data_root, 'test_imgs_RGBD', '000003_rgb.png'),
         os.path.join(data_root, 'test_imgs_RGBD', '000003_camera_matrix.json'),
         os.path.join(project_dir, 'outputs', 'wapr.json'),
         os.path.join(project_dir, 'outputs', 'wapr_filtered.json'),
         score_threshold, nms_iou_threshold, device=device, mesh_scale_to_m=1e-3)
