"""Check public API boundaries using synthetic fixtures, without model weights.
使用合成输入检查公开接口边界，不需要模型权重。
"""
import os
import sys
import numpy as np
import pytest

# Import the checkout directly so tests work from an external directory.
# 直接定位当前副本，支持从外部目录执行测试。
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
from Refinement.wapr_public import estimate_from_masks, restore_full_frame_masks


class RecordingEstimator:
    def estimate_many_categories_many_instances(self, rgb, depth_m, K, instances):
        self.received = rgb, depth_m, K, instances
        return [{'pose_4x4': np.eye(4), 'score_6d': 0.2 + i} for i in range(len(instances))]


def inputs():
    image = np.zeros((16, 24, 3), np.uint8)
    image[:] = [3, 7, 11]
    depth = np.ones((16, 24), np.float32) * 2
    camera = np.array([[20, 0, 12], [0, 20, 8], [0, 0, 1]], np.float32)
    instances = [{'mesh': object(), 'diameter_m': .1, 'mask': np.ones((16, 24), bool)}]
    return image, depth, camera, instances


def test_rgb_units_order_and_no_mutation():
    image, depth, camera, instances = inputs()
    instances.append({**instances[0], 'diameter_m': .3})
    estimator = RecordingEstimator()
    results = estimate_from_masks(estimator, image, depth, camera, instances)
    rgb, observed_depth, K, forwarded = estimator.received
    assert rgb[0, 0].tolist() == [11, 7, 3]
    assert rgb.flags.c_contiguous
    np.testing.assert_array_equal(image[0, 0], [3, 7, 11])
    np.testing.assert_array_equal(observed_depth, depth)
    np.testing.assert_array_equal(K, camera)
    assert forwarded[0]['mesh'] is instances[0]['mesh']
    assert [item['diameter_m'] for item in forwarded] == [.1, .3]
    assert [item['score_6d'] for item in results] == [.2, 1.2]
    assert all(set(item) == {'mesh', 'diameter_m', 'mask'} for item in forwarded)


@pytest.mark.parametrize('case', ['image_dtype','depth_shape','depth_nan','depth_negative','camera','mask_shape','mask_nan','empty_mask','no_depth','diameter'])
def test_invalid_input_rejected(case):
    image, depth, camera, instances = inputs()
    if case == 'image_dtype': image = image.astype(np.float32)
    if case == 'depth_shape': depth = depth[:2]
    if case == 'depth_nan': depth[0, 0] = np.nan
    if case == 'depth_negative': depth[0, 0] = -1
    if case == 'camera': camera[0, 0] = 0
    if case == 'mask_shape': instances[0]['mask'] = np.ones((2, 2))
    if case == 'mask_nan': instances[0]['mask'] = np.full(depth.shape, np.nan)
    if case == 'empty_mask': instances[0]['mask'][:] = False
    if case == 'no_depth': depth[:] = 0
    if case == 'diameter': instances[0]['diameter_m'] = -1
    with pytest.raises(ValueError): estimate_from_masks(RecordingEstimator(), image, depth, camera, instances)


def test_empty_inputs_do_not_call_backend():
    image, depth, camera, _ = inputs()
    assert estimate_from_masks(object(), image, depth, camera, []) == []


def test_negative_stride_views_reach_backend_as_contiguous_tensors():
    import torch
    image, depth, camera, instances = inputs()
    depth[:] = np.arange(1, depth.shape[0] + 1)[:, None]
    depth = depth[::-1]
    camera = camera[:, ::-1].copy()[:, ::-1]
    estimator = RecordingEstimator()
    estimate_from_masks(estimator, image, depth, camera, instances)
    _, actual_depth, actual_camera, _ = estimator.received
    torch.testing.assert_close(torch.as_tensor(actual_depth), torch.as_tensor(depth.copy()))
    torch.testing.assert_close(torch.as_tensor(actual_camera), torch.as_tensor(camera.copy()))


def test_backend_contract_errors():
    image, depth, camera, instances = inputs()
    class BrokenEstimator:
        def estimate_many_categories_many_instances(self, *args): return []
    with pytest.raises(RuntimeError, match='count'):
        estimate_from_masks(BrokenEstimator(), image, depth, camera, instances)


@pytest.mark.parametrize('shape', [(32, 32), (65, 97), (31, 31), (480, 1280), (800, 601)])
def test_crop_masks_restore_to_original_frame(shape):
    import torch
    height, width = shape
    # A constant crop with an identity transform makes pixel alignment observable.
    # 恒定裁剪及单位变换用于观察原图像素对齐，而非模拟实验数据。
    mask = torch.zeros((1, 128, 128), dtype=torch.float32)
    mask[:, 12:40, 8:30] = 1
    prediction = {'pred_mask': mask, 'Detect_Bbox_tfs_128': torch.eye(3)[None]}
    actual = restore_full_frame_masks(prediction, (height, width, 3))
    assert actual.shape == (1, height, width)
    assert actual.dtype == np.bool_
    assert actual.any()
    ys, xs = np.where(actual[0])
    ratio = max(height / 640, width / 640)
    scaled_h = int(height / ratio) if ratio > 1 else height
    scaled_w = int(width / ratio) if ratio > 1 else width
    top = ((((scaled_h + 31) // 32) * 32) - scaled_h) // 2
    left = ((((scaled_w + 31) // 32) * 32) - scaled_w) // 2
    expected_x = max(0, 8 - left) * width / scaled_w
    expected_y = max(0, 12 - top) * height / scaled_h
    assert abs(xs.min() - expected_x) <= 3
    assert abs(ys.min() - expected_y) <= 3


@pytest.mark.parametrize('mask_shape,transform_shape', [
    ((2, 16, 16), (1, 3, 3)),
    ((0, 16, 16), (1, 3, 3)),
    ((1, 1, 16, 16), (1, 3, 3)),
    ((1, 16, 16), (1, 2, 3)),
])
def test_misaligned_crop_batch_is_rejected(mask_shape, transform_shape):
    import torch
    prediction = {'pred_mask': torch.zeros(mask_shape), 'Detect_Bbox_tfs_128': torch.zeros(transform_shape)}
    with pytest.raises(ValueError, match='crop|mask'):
        restore_full_frame_masks(prediction, (32, 32))


def test_empty_crop_batch_restores_to_empty_original_frame():
    import torch
    prediction = {'pred_mask': torch.empty(0, 16, 16), 'Detect_Bbox_tfs_128': torch.empty(0, 3, 3)}
    masks = restore_full_frame_masks(prediction, (31, 63))
    assert masks.shape == (0, 31, 63)
    assert masks.dtype == np.bool_


def test_entry_scripts_import_from_external_cwd(tmp_path, monkeypatch):
    import importlib.util
    monkeypatch.chdir(tmp_path)
    for filename in ('predict_image.py', 'predict_wapr.py'):
        path = os.path.join(project_dir, 'examples', 'inference', filename)
        spec = importlib.util.spec_from_file_location('entry_' + filename[:-3], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)


@pytest.fixture
def example_inputs(tmp_path, monkeypatch):
    import cv2
    import json
    from types import SimpleNamespace

    image_path = str(tmp_path / 'image.png')
    depth_path = str(tmp_path / 'depth.png')
    camera_path = tmp_path / 'camera.json'
    camera = [[80, 0, 64], [0, 80, 32], [0, 0, 1]]
    camera_path.write_text(json.dumps(camera), encoding='utf-8')
    # Both dimensions need a one-pixel trailing border at detector stride 32.
    assert cv2.imwrite(image_path, np.zeros((63, 127, 3), np.uint8))
    assert cv2.imwrite(depth_path, np.full((63, 127), 2000, np.uint16))

    class Dataset:
        obj_id_list = [1, 2, 3]
        obj_model_list = ['model1.ply', 'model2.ply', 'model3.ply']
        model_info = {'1': {'diameter': 120}, '2': {'diameter': 270}, '3': {'diameter': 150}}

        def __init__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, 'HccePose.bop_loader', SimpleNamespace(bop_dataset=Dataset))
    return SimpleNamespace(root=str(tmp_path), image=image_path, depth=depth_path,
                           camera=camera_path, output=str(tmp_path / 'outputs' / 'result.json'))


@pytest.mark.parametrize('entry', ['rgb', 'wapr'])
@pytest.mark.parametrize('ids,message', [([1, 1], 'Duplicate'), ([99], 'Unknown')])
def test_invalid_requested_ids_fail_before_tester(example_inputs, monkeypatch, entry, ids, message):
    from types import SimpleNamespace
    from examples.inference import predict_image, predict_wapr

    def unexpected_model(*args, **kwargs):
        pytest.fail('Invalid requests must not load model weights')

    monkeypatch.setitem(sys.modules, 'HccePose.tester', SimpleNamespace(Tester=unexpected_model))
    files = example_inputs
    with pytest.raises(ValueError, match=message):
        if entry == 'rgb':
            predict_image.main(files.root, files.image, files.camera, ids, files.root)
        else:
            predict_wapr.main(files.root, files.image, files.depth, files.camera, ids, files.output, .001)
    assert not os.path.exists(files.output)


def test_wapr_recipe_unit_conversion_and_saved_identity(example_inputs, monkeypatch):
    import json
    from types import SimpleNamespace
    import torch
    from examples.inference import predict_wapr

    class Mesh:
        def __init__(self, path):
            self.path = path

        def apply_scale(self, scale):
            self.scale = scale

    def crops(left_offsets):
        masks = torch.zeros((len(left_offsets), 32, 32))
        masks[:, 8:24, 8:24] = 1
        transforms = torch.eye(3).repeat(len(left_offsets), 1, 1)
        transforms[:, 0, 2] = -torch.tensor(left_offsets)
        return {'pred_mask': masks, 'Detect_Bbox_tfs_128': transforms}

    class Tester:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, camera, image, ids):
            assert ids == [2, 1]
            return {1: crops([0, 40]), 2: crops([80])}

    class Estimator(RecordingEstimator):
        def __init__(self, device):
            assert device == 'cuda:0'

        def estimate_many_categories_many_instances(self, rgb, depth_m, K, instances):
            assert np.all(depth_m == 2)
            assert [i['mesh'].path for i in instances] == ['model2.ply', 'model1.ply', 'model1.ply']
            assert all(i['mesh'].scale == .001 for i in instances)
            assert [i['diameter_m'] for i in instances] == pytest.approx([.27, .12, .12])
            for item, center_x in zip(instances, [96, 16, 56]):
                mask = item['mask']
                assert mask.shape == (63, 127)
                assert mask[16, center_x]
                ys, xs = np.where(mask)
                assert abs(xs.mean() - center_x) < 2
                assert abs(ys.mean() - 16) < 2
            return super().estimate_many_categories_many_instances(rgb, depth_m, K, instances)

    files = example_inputs
    monkeypatch.setitem(sys.modules, 'HccePose.tester', SimpleNamespace(Tester=Tester))
    monkeypatch.setitem(sys.modules, 'wapr.estimator', SimpleNamespace(WAPREstimator=Estimator))
    monkeypatch.setitem(sys.modules, 'trimesh', SimpleNamespace(load=lambda path, **kwargs: Mesh(path)))
    records = predict_wapr.main(files.root, files.image, files.depth, files.camera, iter([2, 1]),
                                files.output, .001)
    assert [(r['object_id'], r['instance_index']) for r in records] == [(2, 0), (1, 0), (1, 1)]
    assert [r['score_6d'] for r in records] == pytest.approx([.2, 1.2, 2.2])
    with open(files.output, encoding='utf-8') as handle:
        assert json.load(handle) == records


def test_known_category_without_detections_returns_no_records(example_inputs, monkeypatch):
    import json
    from types import SimpleNamespace
    from examples.inference import predict_wapr

    class EmptyTester:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args):
            return {}

    files = example_inputs
    monkeypatch.setitem(sys.modules, 'HccePose.tester', SimpleNamespace(Tester=EmptyTester))
    monkeypatch.setitem(sys.modules, 'trimesh', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'wapr.estimator', None)
    result = predict_wapr.main(files.root, files.image, files.depth, files.camera, [3], files.output, .001)
    assert result == []
    with open(files.output, encoding='utf-8') as handle:
        assert json.load(handle) == []
