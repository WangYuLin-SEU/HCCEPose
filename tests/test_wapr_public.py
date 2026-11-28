"""Check public API boundaries using synthetic fixtures, without model weights.
使用合成输入检查公开接口边界，不需要模型权重。
"""
import importlib.util
import json
import os
import runpy
import shutil
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

# Import the checkout directly so tests work from an external directory.
# 直接定位当前副本，支持从外部目录执行测试。
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
from Refinement.refinement_group import estimate_from_masks, restore_full_frame_masks
from Refinement import refinement_group
from HccePose import bop_loader, tester as tester_module
import s4_p3_test_mi10_bin_picking_RGBD_wapr as wapr_example


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
    prediction = {'pred_mask': torch.zeros(mask_shape), 'Detect_Bbox_tfs_128': torch.zeros(transform_shape)}
    with pytest.raises(ValueError, match='crop|mask'):
        restore_full_frame_masks(prediction, (32, 32))


def test_empty_crop_batch_restores_to_empty_original_frame():
    prediction = {'pred_mask': torch.empty(0, 16, 16), 'Detect_Bbox_tfs_128': torch.empty(0, 3, 3)}
    masks = restore_full_frame_masks(prediction, (31, 63))
    assert masks.shape == (0, 31, 63)
    assert masks.dtype == np.bool_


def test_entry_scripts_import_from_external_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = os.path.join(project_dir, 's4_p3_test_mi10_bin_picking_RGBD_wapr.py')
    spec = importlib.util.spec_from_file_location('entry_wapr', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.Tester is tester_module.Tester
    assert not hasattr(module, 'main')
    assert not (tmp_path / 'outputs').exists()


@pytest.fixture
def example_inputs(tmp_path, monkeypatch):
    # Run an unchanged copy of the recipe against an isolated project layout.
    # 使用未改写的示例副本和隔离目录运行入口，不替换或动态生成 Python 源码。
    script_path = tmp_path / 's4_p3_test_mi10_bin_picking_RGBD_wapr.py'
    shutil.copyfile(os.path.join(project_dir, script_path.name), script_path)
    dataset_root = tmp_path / 'data' / 'demo-bin-picking'
    dataset_root.mkdir(parents=True)
    capture_dir = tmp_path / 'sample_data' / 'rgbd'
    capture_dir.mkdir(parents=True)
    image_path = str(capture_dir / '000003_rgb.png')
    depth_path = str(capture_dir / '000003_depth.png')
    camera_path = capture_dir / '000003_camK.json'
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

    monkeypatch.setattr(bop_loader, 'bop_dataset', Dataset)
    return SimpleNamespace(root=str(dataset_root), image=image_path, depth=depth_path,
                           camera=camera_path, output=str(tmp_path / 'outputs' / 'wapr.json'),
                           script=str(script_path))


def test_unknown_requested_id_fails_before_tester(example_inputs, monkeypatch):
    def unexpected_model(*args, **kwargs):
        pytest.fail('Invalid requests must not load model weights')

    monkeypatch.setattr(tester_module, 'Tester', unexpected_model)
    monkeypatch.setattr(bop_loader.bop_dataset, 'obj_id_list', [2, 3])
    files = example_inputs
    with pytest.raises(ValueError, match='Unknown'):
        runpy.run_path(files.script, run_name='__main__')
    assert not os.path.exists(files.output)


@pytest.mark.parametrize('case', [
    'missing_image', 'missing_depth', 'missing_camera', 'corrupt_image',
    'wrong_depth_shape', 'multichannel_depth', 'bad_camera_shape', 'zero_focal',
])
def test_file_recipe_rejects_invalid_files_before_model_loading(example_inputs, monkeypatch, case):
    files = example_inputs
    if case.startswith('missing_'):
        os.remove(getattr(files, case.removeprefix('missing_')))
    elif case == 'corrupt_image':
        with open(files.image, 'wb') as handle:
            handle.write(b'not an image')
    elif case == 'wrong_depth_shape':
        assert cv2.imwrite(files.depth, np.ones((4, 6), np.uint16))
    elif case == 'multichannel_depth':
        assert cv2.imwrite(files.depth, np.ones((63, 127, 3), np.uint16))
    elif case == 'bad_camera_shape':
        files.camera.write_text(json.dumps(np.eye(2).tolist()))
    else:
        files.camera.write_text(json.dumps([[0, 0, 64], [0, 80, 32], [0, 0, 1]]))

    def unexpected_model(*args, **kwargs):
        pytest.fail('Invalid input files must fail before model loading')

    monkeypatch.setattr(tester_module, 'Tester', unexpected_model)
    error = FileNotFoundError if case.startswith('missing_') else ValueError
    with pytest.raises(error):
        runpy.run_path(files.script, run_name='__main__')
    assert not os.path.exists(files.output)


def test_file_recipe_reads_bundled_rgbd_frame_in_original_units(example_inputs, monkeypatch):
    files = example_inputs
    source = os.path.join(project_dir, 'sample_data', 'rgbd')
    for name, target in [('000003_rgb.png', files.image), ('000003_depth.png', files.depth),
                         ('000003_camK.json', files.camera)]:
        shutil.copyfile(os.path.join(source, name), target)
    image = cv2.imread(files.image)
    depth = cv2.imread(files.depth, cv2.IMREAD_UNCHANGED)
    calibration_text = files.camera.read_text()
    calibration = json.loads(calibration_text)
    camera = np.array([[calibration['fx'], 0, calibration['cx']],
                       [0, calibration['fy'], calibration['cy']], [0, 0, 1]])

    class RecordingTester:
        last_wapr_raw_records = []

        def __init__(self, *args, **kwargs):
            assert kwargs == {'CUDA_DEVICE': '0', 'hccepose_vis': False}

        def predict(self, K, actual_image, ids, **kwargs):
            np.testing.assert_array_equal(K, camera)
            np.testing.assert_array_equal(actual_image, image)
            np.testing.assert_allclose(kwargs['depth'], depth.astype(np.float32) * 1e-3)
            assert kwargs['depth'].dtype == np.float32
            assert ids == [1] and kwargs['use_wapr'] is True
            return {'wapr_postprocess': {'raw_records': [], 'retained_records': []}}

    monkeypatch.setattr(tester_module, 'Tester', RecordingTester)
    runpy.run_path(files.script, run_name='__main__')
    np.testing.assert_array_equal(cv2.imread(files.depth, cv2.IMREAD_UNCHANGED), depth)
    assert files.camera.read_text() == calibration_text


@pytest.mark.parametrize('camera_format', ['matrix', 'bop'])
def test_wapr_recipe_unit_conversion_and_saved_identity(example_inputs, monkeypatch, camera_format):
    class Mesh:
        is_empty = False
        faces = [[0, 1, 2]]
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]

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

    class Tester(wapr_example.Tester):
        def __init__(self, dataset, **kwargs):
            self.bop_dataset_item = dataset
            self.device = torch.device('cpu')
            self.wapr_registered = False
            self.last_wapr_raw_records = None

        def predict(self, camera, image, ids, **kwargs):
            assert ids == [1]
            assert kwargs['use_wapr'] is True
            predictions = {1: crops([0, 40])}
            report = self._run_wapr_estimation(predictions, ids, camera, image, kwargs['depth'],
                                               kwargs['wbps_threshold'], kwargs['wapr_nms_iou_threshold'])
            return {**predictions, 'wapr_postprocess': report}

    class Estimator(RecordingEstimator):
        def __init__(self, device):
            assert device == 'cpu'

        def estimate_many_categories_many_instances(self, rgb, depth_m, K, instances):
            assert np.all(depth_m == 2)
            assert [i['mesh'].path for i in instances] == ['model1.ply', 'model1.ply']
            assert all(i['mesh'].scale == .001 for i in instances)
            assert [i['diameter_m'] for i in instances] == pytest.approx([.12, .12])
            for item, center_x in zip(instances, [16, 56]):
                mask = item['mask']
                assert mask.shape == (63, 127)
                assert mask[16, center_x]
                ys, xs = np.where(mask)
                assert abs(xs.mean() - center_x) < 2
                assert abs(ys.mean() - 16) < 2
            return super().estimate_many_categories_many_instances(rgb, depth_m, K, instances)

    files = example_inputs
    if camera_format == 'bop':
        files.camera.write_text(json.dumps({'width': 127, 'height': 63,
                                           'fx': 80, 'fy': 80, 'cx': 64, 'cy': 32}))
    monkeypatch.setattr(tester_module, 'Tester', Tester)
    monkeypatch.setattr(refinement_group, 'WAPREstimator', Estimator)
    monkeypatch.setattr(refinement_group.trimesh, 'load', lambda path, **kwargs: Mesh(path))
    calls = []
    def nms(poses, mesh, K, height, width, iou, device):
        calls.append((poses, mesh, iou, device))
        return poses
    monkeypatch.setattr(tester_module, 'rendered_mask_iou_nms_gpu', nms)
    report = runpy.run_path(files.script, run_name='__main__')['report']
    records = report['raw_records']
    assert [(r['object_id'], r['instance_index']) for r in records] == [(1, 0), (1, 1)]
    assert [r['score_6d'] for r in records] == pytest.approx([.2, 1.2])
    with open(files.output, encoding='utf-8') as handle:
        assert json.load(handle) == records
    filtered_path = os.path.splitext(files.output)[0] + '_filtered.json'
    with open(filtered_path, encoding='utf-8') as handle:
        assert json.load(handle) == report
    assert report['kept_input_indices'] == [1]
    assert len(calls) == 1 and calls[0][2:] == (.5, 'cpu')


def test_known_category_without_detections_returns_no_records(example_inputs, monkeypatch):
    class EmptyTester(wapr_example.Tester):
        def __init__(self, *args, **kwargs):
            self.last_wapr_raw_records = []

        def predict(self, *args, **kwargs):
            report = tester_module.Tester.postprocess_wapr([], {}, args[0], args[1].shape, .75, .5)
            return {'wapr_postprocess': report}

    files = example_inputs
    monkeypatch.setattr(tester_module, 'Tester', EmptyTester)
    monkeypatch.setattr(refinement_group, 'WAPREstimator', None)
    result = runpy.run_path(files.script, run_name='__main__')['report']
    assert result['raw_records'] == result['retained_records'] == []
    with open(files.output, encoding='utf-8') as handle:
        assert json.load(handle) == []


def test_automatic_nms_failure_keeps_raw_output(example_inputs, monkeypatch):
    files = example_inputs
    record = {'object_id': 1, 'instance_index': 0, 'score_6d': .9,
              'pose_4x4_m': np.eye(4).tolist()}
    def fail(*args, **kwargs):
        tester.last_wapr_raw_records = [record]
        raise RuntimeError('renderer unavailable')
    tester = SimpleNamespace(predict=fail, last_wapr_raw_records=None)
    monkeypatch.setattr(tester_module, 'Tester', lambda *args, **kwargs: tester)
    with pytest.raises(RuntimeError, match='renderer unavailable'):
        runpy.run_path(files.script, run_name='__main__')
    with open(files.output, encoding='utf-8') as handle:
        raw = json.load(handle)
    assert len(raw) == 1 and raw[0]['score_6d'] == .9
    assert not os.path.exists(os.path.splitext(files.output)[0] + '_filtered.json')


@pytest.mark.parametrize('link_type', ['symlink', 'hardlink'])
def test_automatic_outputs_cannot_alias(example_inputs, monkeypatch, link_type):
    files = example_inputs
    os.makedirs(os.path.dirname(files.output))
    with open(files.output, 'w', encoding='utf-8') as handle:
        handle.write('original raw result')
    filtered = os.path.splitext(files.output)[0] + '_filtered.json'
    if link_type == 'symlink':
        os.symlink(files.output, filtered)
    else:
        os.link(files.output, filtered)
    def unexpected_model(*args, **kwargs):
        pytest.fail('Aliased output paths must fail before model loading')
    monkeypatch.setattr(tester_module, 'Tester', unexpected_model)
    with pytest.raises(ValueError, match='separate output'):
        runpy.run_path(files.script, run_name='__main__')
    with open(files.output, encoding='utf-8') as handle:
        assert handle.read() == 'original raw result'


@pytest.mark.parametrize('output_kind', ['raw', 'filtered'])
@pytest.mark.parametrize('input_kind', ['image', 'depth', 'camera'])
@pytest.mark.parametrize('link_type', ['symlink', 'hardlink'])
def test_file_recipe_cannot_overwrite_input_files(example_inputs, monkeypatch,
                                                 output_kind, input_kind, link_type):
    files = example_inputs
    input_path = str(getattr(files, input_kind))
    with open(input_path, 'rb') as handle:
        original = handle.read()
    output = files.output if output_kind == 'raw' else os.path.splitext(files.output)[0] + '_filtered.json'
    os.makedirs(os.path.dirname(output))
    if link_type == 'symlink':
        os.symlink(input_path, output)
    else:
        os.link(input_path, output)

    def unexpected_model(*args, **kwargs):
        pytest.fail('Output paths pointing to inputs must fail before model loading')

    monkeypatch.setattr(tester_module, 'Tester', unexpected_model)
    with pytest.raises(ValueError, match='input'):
        runpy.run_path(files.script, run_name='__main__')
    with open(input_path, 'rb') as handle:
        assert handle.read() == original


@pytest.fixture
def seeded_backend(monkeypatch):
    """Expose the fixed public component contracts with observable synthetic poses.
    以可观测的合成位姿检查固定公开组件的调用约定，不载入权重或渲染器。
    """
    state = SimpleNamespace(refine_calls=[], score_calls=[], global_calls=[], best=-1)
    recipe = SimpleNamespace(wapr_iters=3, sapr_iters=2, wapr_max_rot_rad=.70,
                             sapr_max_rot_rad=.35, refine_crop_ratio=1.2)
    monkeypatch.setattr(refinement_group, 'wapr_recipe', recipe)
    monkeypatch.setattr(refinement_group, 'wbps_group_sizes', frozenset(range(2, 26)))
    monkeypatch.setattr(refinement_group, 'resolve_view_counts', lambda *args: (4, 3))
    monkeypatch.setattr(refinement_group, 'center_from_mesh', lambda mesh: np.array([.1, -.2, .03]))

    def rotations(n_view, n_inplane):
        state.global_calls.append((n_view, n_inplane))
        return np.stack([cv2.Rodrigues(np.array([0., 0., i / 12.]))[0] for i in range(12)])

    monkeypatch.setattr(refinement_group, 'make_view_rots', rotations)
    monkeypatch.setattr(refinement_group, 'guess_translation', lambda *args: np.array([.03, .04, .5]))

    class Estimator:
        device = 'cpu'
        nets = {'wapr_w_mask': 'masked', 'wapr_wo_mask': 'unmasked', 'sapr': 'sapr'}

        def _pose_meshes(self, meshes):
            return meshes

        def _refine(self, net, poses, rgb, depth, mask, K, mesh, center, diameter, **kwargs):
            np.testing.assert_array_equal(rgb[0, 0], [11, 7, 3])
            np.testing.assert_allclose(center, [.1, -.2, .03])
            assert diameter == .1
            state.refine_calls.append((net, poses.clone(), dict(kwargs)))
            result = poses.clone()
            result[:, 0, 3] += .01 if net != 'sapr' else .02
            return result

        def _wbps(self, poses, rgb, depth, mask, K, mesh, center, diameter, group_size):
            state.score_calls.append(poses.clone())
            assert len(poses) == group_size
            within = torch.zeros(1, group_size)
            within[0, state.best] = 10
            between = torch.full((1, group_size), -100.)
            between[0, 0] = -50
            return within, between

        def estimate_many_categories_many_instances(self, rgb, depth, K, instances):
            state.global_calls.append('fallback')
            pose = np.eye(4)
            pose[2, 3] = .8
            return [{'pose_4x4': pose, 'score_6d': .9} for _ in instances]

    backend = refinement_group.Refinement_WAPR(device='cpu')
    backend.estimator = Estimator()
    return backend, state


@pytest.mark.parametrize('mode', ['hybrid', 'refine'])
def test_seeded_modes_use_one_joint_score_group_and_original_frame(seeded_backend, mode):
    backend, state = seeded_backend
    image, depth, camera, instances = inputs()
    seed = np.eye(4, dtype=np.float32)
    seed[:3, 3] = [.12, -.03, .7]
    instances[0]['pose_4x4_m'] = seed.copy()
    result = backend.inference_batch(camera, image, depth, instances, mode=mode)[0]
    assert result['score_6d'] == .75  # Uses the complete group's maximum, not the selected row.
    assert result['hypothesis_source'] == 'pnp_refined'
    assert result['hypothesis_count'] == (13 if mode == 'hybrid' else 2)
    assert len(state.score_calls) == 1 and len(state.refine_calls) == 2
    first, second = state.refine_calls
    assert first[0] == ('masked' if mode == 'hybrid' else 'unmasked')
    assert first[2] == {'iters': 3, 'max_rot': .70, 'crop_ratio': 1.2, 'use_mask': mode == 'hybrid'}
    assert second[0] == 'sapr' and second[2] == {'iters': 2, 'max_rot': .35, 'crop_ratio': 1.2, 'use_mask': False}
    np.testing.assert_array_equal(first[1][-1].numpy(), seed)
    np.testing.assert_allclose(result['pose_4x4'][:3, 3], [.15, -.03, .7])
    np.testing.assert_array_equal(instances[0]['pose_4x4_m'], seed)
    if mode == 'hybrid':
        assert state.global_calls == [(4, 3)]
        np.testing.assert_allclose(first[1][:-1, :3, 3], np.tile([.03, .04, .5], (12, 1)))
    else:
        assert state.global_calls == []
        np.testing.assert_array_equal(state.score_calls[0][0], seed)
        # Refinement needs no mask foreground; it is driven by the PnP pose.
        # 初值精修由 PnP 位姿驱动，不要求掩码存在前景。
        instances[0]['mask'][:] = False
        backend.inference_batch(camera, image, depth, instances, mode=mode)


@pytest.mark.parametrize('mode,source', [('hybrid', 'wapr'), ('refine', 'pnp')])
def test_seeded_modes_can_keep_other_group_members(seeded_backend, mode, source):
    backend, state = seeded_backend
    state.best = 0
    image, depth, camera, instances = inputs()
    seed = np.eye(4)
    seed[2, 3] = .7
    instances[0]['pose_4x4_m'] = seed
    result = backend.inference_batch(camera, image, depth, instances, mode=mode)[0]
    assert result['hypothesis_source'] == source
    assert result['hypothesis_index'] == 0
    if mode == 'refine':
        np.testing.assert_allclose(result['pose_4x4'], seed, atol=1e-7)


def test_hybrid_without_valid_pnp_uses_public_candidate_estimation(seeded_backend):
    backend, state = seeded_backend
    image, depth, camera, instances = inputs()
    result = backend.inference_batch(camera, image, depth, instances, mode='hybrid')[0]
    assert result['score_6d'] == .9 and result['hypothesis_source'] == 'wapr'
    assert state.global_calls == ['fallback']
    assert state.refine_calls == state.score_calls == []


@pytest.mark.parametrize('mode', ['refine', 'hybrid'])
@pytest.mark.parametrize('bad_seed', ['nan', 'negative_z', 'bad_row', 'wrong_shape'])
def test_invalid_pose_seeds_fail_before_backend_work(seeded_backend, mode, bad_seed):
    backend, state = seeded_backend
    image, depth, camera, instances = inputs()
    seed = np.eye(4)
    seed[2, 3] = .7
    if bad_seed == 'nan': seed[0, 0] = np.nan
    if bad_seed == 'negative_z': seed[2, 3] = -.1
    if bad_seed == 'bad_row': seed[3, 0] = 1
    if bad_seed == 'wrong_shape': seed = seed[:3]
    instances[0]['pose_4x4_m'] = seed
    with pytest.raises(ValueError, match='PnP seeds'):
        backend.inference_batch(camera, image, depth, instances, mode=mode)
    assert state.refine_calls == state.score_calls == state.global_calls == []


def test_refine_requires_seed_and_hybrid_requires_supported_joint_group(seeded_backend, monkeypatch):
    backend, state = seeded_backend
    image, depth, camera, instances = inputs()
    with pytest.raises(ValueError, match='requires a valid PnP'):
        backend.inference_batch(camera, image, depth, instances, mode='refine')
    instances[0]['pose_4x4_m'] = np.eye(4)
    instances[0]['pose_4x4_m'][2, 3] = .7
    monkeypatch.setattr(refinement_group, 'wbps_group_sizes', frozenset([2, 12]))
    with pytest.raises(ValueError, match='group size 13'):
        backend.inference_batch(camera, image, depth, instances, mode='hybrid')
    assert state.refine_calls == state.score_calls == []


@pytest.mark.parametrize('mode', ['refine', 'hybrid'])
def test_seeded_modes_score_instances_separately(seeded_backend, mode):
    backend, state = seeded_backend
    image, depth, camera, instances = inputs()
    first = np.eye(4, dtype=np.float32)
    first[2, 3] = .7
    second = first.copy()
    second[2, 3] = 1.2
    instances[0]['pose_4x4_m'] = first
    instances.append({**instances[0], 'pose_4x4_m': second})
    outputs = backend.inference_batch(camera, image, depth, instances, mode=mode)
    assert len(outputs) == len(state.score_calls) == 2
    assert [row['pose_4x4'][2, 3] for row in outputs] == pytest.approx([.7, 1.2])
    assert [len(poses) for poses in state.score_calls] == ([2, 2] if mode == 'refine' else [13, 13])


@pytest.mark.parametrize('bad_scores', ['nan', 'wrong_shape'])
def test_seeded_modes_reject_invalid_wbps_output(seeded_backend, monkeypatch, bad_scores):
    backend, state = seeded_backend
    image, depth, camera, instances = inputs()
    instances[0]['pose_4x4_m'] = np.eye(4)
    instances[0]['pose_4x4_m'][2, 3] = .7
    shape = (1, 1) if bad_scores == 'wrong_shape' else (1, 2)
    within = torch.zeros(shape)
    between = torch.full(shape, float('nan')) if bad_scores == 'nan' else within.clone()
    monkeypatch.setattr(backend.estimator, '_wbps', lambda *args: (within, between))
    with pytest.raises(RuntimeError, match='WBPS'):
        backend.inference_batch(camera, image, depth, instances, mode='refine')


def test_file_recipe_forwards_defaults_without_second_postprocess(example_inputs, monkeypatch):
    class Tester:
        last_wapr_raw_records = []

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, camera, image, ids, **kwargs):
            assert kwargs['wapr_mode'] == 'estimate' and kwargs['pnp_op'] == 'ransac+comb'
            assert kwargs['use_wapr'] is True
            assert kwargs['wbps_threshold'] == .75 and kwargs['wapr_nms_iou_threshold'] == .5
            return {'wapr_postprocess': {'raw_records': [], 'retained_records': [],
                                        'wapr_mode': 'estimate', 'skipped_instances': []}}

    monkeypatch.setattr(tester_module, 'Tester', Tester)
    files = example_inputs
    report = runpy.run_path(files.script, run_name='__main__')['report']
    assert report['wapr_mode'] == 'estimate'
    with open(files.output, encoding='utf-8') as handle:
        assert json.load(handle) == []
