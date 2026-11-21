"""Run RGB inference with caller-supplied assets. / 使用调用方提供的资源进行 RGB 推理。"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
# Resolve the project package from this file, independent of the launching directory.
# 从当前脚本定位项目内库，不依赖启动目录。
sys.path.insert(0, project_dir)


def main(dataset_root, image_path, intrinsics_path, object_ids, output_dir, device='0'):
    """Save standard visualization artifacts; the camera file is a 3 x 3 JSON array.
    保存标准可视化结果；相机文件为 3 x 3 JSON 数组。
    """
    import cv2
    from examples.inference._inputs import checked_object_ids, read_intrinsics
    from HccePose.bop_loader import bop_dataset
    from HccePose.tester import Tester

    for path in (dataset_root, image_path, intrinsics_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Cannot decode input image')
    cam_K = read_intrinsics(intrinsics_path)
    dataset = bop_dataset(dataset_root)
    object_ids = checked_object_ids(object_ids, dataset.obj_id_list)
    tester = Tester(dataset, CUDA_DEVICE=device, hccepose_vis=True)
    results = tester.predict(cam_K, image, object_ids)
    os.makedirs(output_dir, exist_ok=True)
    for key in ('show_2D_results', 'show_6D_vis0', 'show_6D_vis1', 'show_6D_vis2'):
        artifact = results.get(key)
        if artifact is not None and not cv2.imwrite(os.path.join(output_dir, key + '.jpg'), artifact):
            raise OSError('Cannot save ' + key)
    return results


if __name__ == '__main__':
    # Replace these paths and IDs with your own matched assets.
    # 将这些路径和编号改为自己的配套资源。
    data_root = os.path.join(project_dir, 'data', 'demo-bin-picking')
    image_path = os.path.join(project_dir, 'data', 'test_imgs_RGBD', '000003_rgb.png')
    intrinsics_path = os.path.join(project_dir, 'data', 'test_imgs_RGBD', '000003_camera_matrix.json')
    main(data_root, image_path, intrinsics_path, [1], os.path.join(project_dir, 'outputs', 'rgb'))
