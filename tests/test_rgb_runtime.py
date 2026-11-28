"""Exercise RGB inference with synthetic detections and no object weights.
使用合成检测检查 RGB 推理，无需物体权重。
"""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from importlib.util import module_from_spec, spec_from_file_location

import cv2
import numpy as np
import pytest
import torch
import trimesh

from HccePose.PnP_solver import solve_PnP
from HccePose.bop_loader import inference_image_geometry, inout
from HccePose.network_model import load_checkpoint
from Refinement import refinement_group
from Refinement.refinement_group import Refinement_MP


@pytest.fixture(scope="module")
def runtime():
    # Exercise the real Tester, transforms, projection and solver. Dataset file IO
    # is separate from this test; the class annotation only needs an import target.
    loader = ModuleType('HccePose.bop_loader')
    loader.bop_dataset = object
    loader.IMAGENET_MEAN_BGR = (0.406, 0.456, 0.485)
    loader.IMAGENET_STD_BGR = (0.225, 0.224, 0.229)
    loader.inference_image_geometry = inference_image_geometry
    loader.inout = inout

    spec = spec_from_file_location('hccepose_test_runtime', Path(__file__).resolve().parents[1] / 'HccePose' / 'tester.py')
    tester = module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, loader.__name__, loader)
        spec.loader.exec_module(tester)
    return tester


def test_stride_padding_and_per_detection_pnp_status(runtime):
    tester = object.__new__(runtime.Tester)
    tester.bop_dataset_item = SimpleNamespace(obj_id_list=[1])
    tester.HccePose_Item_info = {1: {'keypoints_': [1]}}
    tester.device = torch.device('cpu')
    tester.crop_size = 32
    tester.show_op = False
    camera = np.array([[80., 0., 15.], [0., 80., 15.], [0., 0., 1.]], dtype=np.float32)
    image = np.full((31, 31, 3), 255, np.uint8)

    def detect(tensor, **kwargs):
        assert tensor.shape == (1, 3, 32, 32)
        assert torch.all(tensor[:, :, -1] == 0)
        assert torch.all(tensor[:, :, :, -1] == 0)
        assert torch.all(tensor[:, :, :31, :31] == 1)
        boxes = SimpleNamespace(
            xywh=torch.tensor([[14., 14., 10., 10.], [19., 14., 8., 8.]]),
            cls=torch.zeros(2), conf=torch.ones(2),
        )
        return [SimpleNamespace(boxes=boxes)]

    def infer(crops, boxes):
        # A projected non-coplanar cube for detection 0, an empty mask for 1.
        cube = np.array([[x, y, z] for x in (-10, 10) for y in (-10, 10)
                        for z in (-10, 10)], dtype=np.float32)
        translation = np.array([0., 0., 180.])
        pixels, _ = cv2.projectPoints(cube, np.zeros(3), translation, camera, None)
        mask = torch.zeros((2, 2, 4))
        mask[0] = 1
        xyz = torch.from_numpy(cube.reshape(1, 2, 4, 3)).repeat(2, 1, 1, 1)
        uv = torch.from_numpy(pixels.reshape(1, 2, 4, 2)).repeat(2, 1, 1, 1)
        return dict(pred_mask=mask, coord_2d_image=uv, pred_front_code_obj=xyz,
                    pred_back_code_obj=xyz, pred_front_code=xyz, pred_back_code=xyz)

    tester.model_yolo = detect
    tester.HccePose_Runtime_Item = {1: SimpleNamespace(inference_batch=infer)}
    group = tester.predict(camera, image, [1])[1]
    np.testing.assert_array_equal(group['pnp_success'], [True, False])
    np.testing.assert_allclose(group['Rts'][0, :3, 3], [0., 0., 180.], atol=1e-3)
    np.testing.assert_array_equal(group['Rts'][1], np.eye(4))


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_degenerate_correspondences_have_no_valid_pose(mode):
    mask = np.ones((3, 3), dtype=np.uint8)
    xyz = np.zeros((3, 3, 3), dtype=np.float32)
    uv = np.zeros((3, 3, 2), dtype=np.float32)
    result = solve_PnP((mask, xyz, uv, np.eye(3, dtype=np.float32)), pnp_op=mode)
    assert not result['success']
    np.testing.assert_array_equal(result['rot'], np.eye(3))
    np.testing.assert_array_equal(result['tvecs'], np.zeros((3, 1)))


def test_invalid_checkpoint_reports_object_file_and_cause(tmp_path):
    checkpoint = tmp_path / '1step.pth'
    checkpoint.write_bytes(b'not a checkpoint')
    with pytest.raises(RuntimeError, match='object 7') as failure:
        load_checkpoint(tmp_path, torch.nn.Linear(1, 1), inference_object_id=7)
    assert str(checkpoint) in str(failure.value)
    assert failure.value.__cause__ is not None
    with pytest.raises(RuntimeError, match='Cannot restore checkpoint'):
        load_checkpoint(tmp_path, torch.nn.Linear(1, 1))


def test_combined_pnp_rejects_missing_prior_before_detector(runtime):
    tester = object.__new__(runtime.Tester)
    tester.HccePose_Item_info = {7: {'keypoints_': [], 'checkpoint_path': 'example.pth'}}
    with pytest.raises(ValueError, match='object 7.*example.pth'):
        tester.predict(np.eye(3), np.zeros((32, 32, 3), np.uint8), [7])


def test_megapose_job_boundary(tmp_path, monkeypatch):
    backend = Refinement_MP(cache_dir=tmp_path)
    monkeypatch.setattr(backend, 'ensure_setup', lambda: None)
    backend.register_objects([7], [tmp_path / 'object.ply'])
    pose = np.eye(4, dtype=np.float32)[None]
    pose[0, 2, 3] = 0.8

    def fake_job(args):
        options = dict(zip(args[2::2], args[3::2]))
        config = json.loads(Path(options['--input-json']).read_text())
        assert config['labels'] == ['obj_000007']
        with np.load(options['--input-npz']) as data:
            np.testing.assert_array_equal(data['rgb'][0, 0], [30, 20, 10])
            np.testing.assert_array_equal(data['initial_poses'], pose)
        np.savez(options['--output-npz'], poses=pose, scores=[0.3], valid_ids=[0])

    monkeypatch.setattr(backend, '_run_env_python', fake_job)
    image = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)
    poses, scores, ids, _ = backend.inference_batch(
        np.eye(3), image, None, None, [7], initial_poses=pose,
    )
    np.testing.assert_array_equal(poses, pose)
    np.testing.assert_allclose(scores, [0.3])
    np.testing.assert_array_equal(ids, [0])


@pytest.mark.parametrize('shape', [(63, 127), (801, 1281)])
@pytest.mark.parametrize('all_rejected', [False, True])
def test_tester_wapr_masks_units_filtering_and_visualization(runtime, tmp_path, monkeypatch, shape, all_rejected):
    """Exercise the real predict path with synthetic network outputs and public-API doubles.
    使用合成网络输出及公开接口替身，覆盖真实 predict 路径，不运行 GPU 模型。
    """
    mesh_path = tmp_path / 'object.ply'
    trimesh.creation.box(extents=[100, 100, 100]).export(mesh_path)
    tester = object.__new__(runtime.Tester)
    tester.bop_dataset_item = SimpleNamespace(obj_id_list=[1, 2], obj_model_list=[mesh_path, mesh_path],
                                             model_info={'1': {'diameter': 100}, '2': {'diameter': 200}})
    tester.device = torch.device('cpu')
    tester.crop_size = 256
    tester.show_op = True
    tester.wapr_registered = False
    tester.WAPR_Item = None
    tester.HccePose_Item_info = {1: {'keypoints_': None}, 2: {'keypoints_': None}}
    tester.BBox_3d = []
    height, width = shape
    camera = np.array([[80., 0., width / 2], [0., 80., height / 2], [0., 0., 1.]], np.float32)
    image = np.full((height, width, 3), [3, 7, 11], np.uint8)
    depth = np.full((height, width), 2., np.float32)
    original_camera = camera.copy()
    captured = []
    nms_calls = []
    real_montage = runtime.vis_rgb_mask_Coord
    real_full_view = runtime.vis_rgb_mask_Coord_origin

    def detect(tensor, **kwargs):
        boxes = SimpleNamespace(xywh=torch.tensor([[20., 20., 12., 12.], [40., 20., 12., 12.],
                                                   [60., 20., 12., 12.], [90., 20., 12., 12.]]),
                                cls=torch.tensor([0., 0., 0., 1.]),
                                conf=torch.tensor([.91, .92, .93, .94]))
        return [SimpleNamespace(boxes=boxes)]

    def infer(crops, boxes):
        count = len(crops)
        xyz = torch.zeros(count, 128, 128, 3)
        return dict(pred_mask=torch.ones(count, 128, 128), coord_2d_image=xyz[..., :2],
                    pred_front_code_obj=xyz, pred_back_code_obj=xyz,
                    pred_front_code=xyz, pred_back_code=xyz,
                    pred_front_code_raw=torch.zeros(count, 128, 128, 24),
                    pred_back_code_raw=torch.zeros(count, 128, 128, 24),
                    pred_mask_logits=torch.arange(count)[:, None, None, None].expand(count, 1, 128, 128),
                    pred_front_back_code_logits=torch.zeros(count, 48, 128, 128))

    class Estimator:
        def __init__(self, device):
            assert device == 'cpu'

        def estimate_many_categories_many_instances(self, rgb, observed, K, instances):
            captured.append(instances)
            assert rgb.shape == image.shape and rgb[0, 0].tolist() == [11, 7, 3]
            np.testing.assert_array_equal(observed, depth)
            np.testing.assert_array_equal(K, original_camera)
            assert [item['diameter_m'] for item in instances] == pytest.approx([.2, .1, .1, .1])
            for item in instances:
                assert item['mask'].shape == shape and item['mask'].any()
                assert item['mesh'].extents.tolist() == pytest.approx([.1, .1, .1])
            outputs = []
            for index, score in enumerate([.75, .749, 1.1, .9]):
                pose = np.eye(4)
                pose[2, 3] = .5 + index
                outputs.append({'pose_4x4': pose, 'score_6d': .1 if all_rejected else score})
            return outputs

    def nms(poses, mesh, K, h, w, iou, device):
        assert (h, w) == shape and iou == .5 and device == 'cpu'
        np.testing.assert_array_equal(K, original_camera)
        nms_calls.append(poses)
        return poses[:1]

    def unexpected_pnp(*args, **kwargs):
        pytest.fail('Mask-based WAPR must not require or solve PnP')

    def montage(crops, masks, front, back):
        assert len(crops) == len(masks) == 2
        return np.zeros((32, 32, 3), np.uint8)

    def full_view(K, ids, object_ids, boxes, poses, conf, *args):
        np.testing.assert_array_equal(ids, [1, 2])
        np.testing.assert_allclose(conf, [.92, .94])
        np.testing.assert_allclose(poses[:, 2, 3], [2500., 500.])
        return np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32, 3), np.uint8)

    tester.model_yolo = detect
    tester.HccePose_Runtime_Item = {key: SimpleNamespace(inference_batch=infer) for key in (1, 2)}
    monkeypatch.setattr(refinement_group, 'WAPREstimator', Estimator)
    monkeypatch.setattr(runtime, 'rendered_mask_iou_nms_gpu', nms)
    monkeypatch.setattr(runtime, 'solve_PnP', unexpected_pnp)
    monkeypatch.setattr(runtime, 'solve_PnP_comb', unexpected_pnp)
    monkeypatch.setattr(runtime, 'vis_rgb_mask_Coord', montage)
    monkeypatch.setattr(runtime, 'vis_rgb_mask_Coord_origin', full_view)
    results = tester.predict(camera, image, [2, 1], depth=depth, use_wapr=True)
    report = results['wapr_postprocess']
    assert len(report['raw_records']) == 4 and len(captured) == 1
    assert report['kept_input_indices'] == ([] if all_rejected else [0, 2])
    assert [row['status'] for row in report['decisions']] == (
        ['below_score_threshold'] * 4 if all_rejected else
        ['kept', 'below_score_threshold', 'kept', 'not_retained_by_mask_nms'])
    assert len(nms_calls) == (0 if all_rejected else 2)
    for obj_id in (1, 2):
        entry = results[obj_id]
        count = 0 if all_rejected else 1
        assert entry['Rts'].shape == (count, 4, 4)
        assert 'pnp_success' not in entry
        for name, value in entry.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                assert value.shape[0] == count, name
    if not all_rejected:
        np.testing.assert_array_equal(results[1]['wapr_instance_indices'], [1])
        np.testing.assert_allclose(results[1]['wapr_scores'], [1.1])
        assert results[1]['pred_mask_logits'][0, 0, 0, 0] == 1
    np.testing.assert_array_equal(camera, original_camera)
    np.testing.assert_array_equal(image[0, 0], [3, 7, 11])
    assert results['time_dict']['wapr'] > 0
    registered = tester.WAPR_Item
    tester.predict(camera, image, [2, 1], depth=depth, use_wapr=True)
    assert tester.WAPR_Item is registered and len(captured) == 2
    if not all_rejected:
        # Check real OpenCV rendering too, after the field-alignment probes above.
        # 字段对齐探针通过后，再检查真实 OpenCV 绘图路径。
        tester.BBox_3d = np.asarray([[[[-50, -50, -50], [50, -50, -50]]]] * 2)
        monkeypatch.setattr(runtime, 'vis_rgb_mask_Coord', real_montage)
        monkeypatch.setattr(runtime, 'vis_rgb_mask_Coord_origin', real_full_view)
        rendered = tester.predict(camera, image, [2, 1], depth=depth, use_wapr=True)
        for key in ('show_2D_results', 'show_6D_vis0', 'show_6D_vis1', 'show_6D_vis2'):
            assert rendered[key].ndim == 3 and np.isfinite(rendered[key]).all()
        def broken_renderer(*args):
            raise RuntimeError('renderer unavailable')
        monkeypatch.setattr(runtime, 'rendered_mask_iou_nms_gpu', broken_renderer)
        with pytest.raises(RuntimeError, match='renderer unavailable'):
            tester.predict(camera, image, [2, 1], depth=depth, use_wapr=True)
        assert len(tester.last_wapr_raw_records) == 4


def test_wapr_no_detections_needs_no_backend(runtime, monkeypatch):
    tester = object.__new__(runtime.Tester)
    tester.bop_dataset_item = SimpleNamespace(obj_id_list=[1])
    tester.device = torch.device('cpu')
    tester.crop_size = 256
    tester.show_op = True
    boxes = SimpleNamespace(xywh=torch.empty(0, 4), conf=torch.empty(0), cls=torch.empty(0))
    tester.model_yolo = lambda *args, **kwargs: [SimpleNamespace(boxes=boxes)]
    monkeypatch.setattr(refinement_group, 'WAPREstimator', None)
    result = tester.predict(np.eye(3), np.zeros((31, 63, 3), np.uint8), [1],
                            depth=np.ones((31, 63), np.float32), use_wapr=True)
    assert result['wapr_postprocess']['raw_records'] == []
    assert result['wapr_postprocess']['retained_records'] == []
    assert tester.last_wapr_raw_records == []
    assert result['show_6D_vis1'].dtype == np.uint8


@pytest.mark.parametrize('case', ['missing_depth', 'wrong_shape', 'negative_depth', 'both_backends',
                                  'duplicate_id', 'unknown_id', 'boolean_id', 'bad_threshold', 'bad_iou'])
def test_wapr_rejects_invalid_requests_before_detection(runtime, case):
    tester = object.__new__(runtime.Tester)
    tester.bop_dataset_item = SimpleNamespace(obj_id_list=[1])
    options = dict(depth=np.ones((31, 63), np.float32), use_wapr=True)
    ids = [1]
    if case == 'missing_depth': options['depth'] = None
    if case == 'wrong_shape': options['depth'] = np.ones((2, 2))
    if case == 'negative_depth': options['depth'][0, 0] = -1
    if case == 'both_backends': options['use_megapose'] = True
    if case == 'duplicate_id': ids = [1, 1]
    if case == 'unknown_id': ids = [3]
    if case == 'boolean_id': ids = [True]
    if case == 'bad_threshold': options['wbps_threshold'] = np.nan
    if case == 'bad_iou': options['wapr_nms_iou_threshold'] = 2
    with pytest.raises(ValueError):
        tester.predict(np.eye(3), np.zeros((31, 63, 3), np.uint8), ids, **options)


@pytest.mark.parametrize('mode', ['estimate', 'hybrid', 'refine'])
@pytest.mark.parametrize('suppress_first', [False, True])
def test_wapr_modes_real_pnp_units_failures_and_retained_rows(runtime, tmp_path, monkeypatch, mode, suppress_first):
    """Use the real PnP solver and Tester with a recording WAPR boundary.
    使用真实 PnP 求解器和 Tester，记录 WAPR 边界的米制初值及实例对应。
    """
    mesh_path = tmp_path / 'object.ply'
    trimesh.creation.box(extents=[100, 100, 100]).export(mesh_path)
    tester = object.__new__(runtime.Tester)
    tester.bop_dataset_item = SimpleNamespace(obj_id_list=[1], obj_model_list=[mesh_path],
                                             model_info={'1': {'diameter': 100}})
    tester.device = torch.device('cpu')
    tester.crop_size = 32
    tester.show_op = False
    tester.wapr_registered = False
    tester.WAPR_Item = None
    camera = np.array([[80., 0., 15.], [0., 80., 15.], [0., 0., 1.]], np.float32)
    image = np.full((31, 31, 3), [3, 7, 11], np.uint8)
    depth = np.full((31, 31), .7, np.float32)
    boxes = SimpleNamespace(xywh=torch.tensor([[14., 14., 10., 10.], [19., 14., 8., 8.]]),
                            cls=torch.zeros(2), conf=torch.tensor([.8, .9]))
    tester.model_yolo = lambda *args, **kwargs: [SimpleNamespace(boxes=boxes)]
    calls = []

    def infer(crops, boxes):
        cube = np.array([[x, y, z] for x in (-10, 10) for y in (-10, 10)
                         for z in (-10, 10)], np.float32)
        pixels, _ = cv2.projectPoints(cube, np.zeros(3), np.array([0., 0., 180.]), camera, None)
        xyz = torch.from_numpy(cube.reshape(1, 2, 4, 3)).repeat(2, 1, 1, 1)
        uv = torch.from_numpy(pixels.reshape(1, 2, 4, 2)).repeat(2, 1, 1, 1)
        xyz[1] = 0
        uv[1] = 0
        return dict(pred_mask=torch.ones(2, 2, 4), coord_2d_image=uv,
                    pred_front_code_obj=xyz, pred_back_code_obj=xyz,
                    pred_front_code=xyz, pred_back_code=xyz)

    def infer_wapr(backend, K, rgb, observed, instances, mode):
        assert not torch.is_grad_enabled()
        np.testing.assert_array_equal(K, camera)
        np.testing.assert_array_equal(observed, depth)
        assert rgb.shape == image.shape
        assert len(instances) == (1 if mode == 'refine' else 2)
        if mode == 'estimate':
            assert all('pose_4x4_m' not in item for item in instances)
        else:
            np.testing.assert_allclose(instances[0]['pose_4x4_m'][:3, 3], [0, 0, .18], atol=1e-5)
            if mode == 'hybrid':
                assert 'pose_4x4_m' not in instances[1]
        calls.append(instances)
        outputs = []
        for index, item in enumerate(instances):
            pose = np.eye(4)
            pose[:3, 3] = [.04, .05, .5 + index]
            outputs.append({'pose_4x4': pose, 'score_6d': .9 + index,
                            'hypothesis_source': 'pnp_refined' if 'pose_4x4_m' in item else 'wapr'})
        return outputs

    tester.HccePose_Runtime_Item = {1: SimpleNamespace(inference_batch=infer)}
    monkeypatch.setattr(refinement_group.Refinement_WAPR, 'inference_batch', infer_wapr)
    monkeypatch.setattr(runtime, 'rendered_mask_iou_nms_gpu',
                        lambda poses, *args: poses[1:] if suppress_first else poses)
    if mode == 'estimate':
        def unexpected_pnp(*args, **kwargs):
            pytest.fail('estimate must not call PnP')
        monkeypatch.setattr(runtime, 'solve_PnP', unexpected_pnp)
    result = tester.predict(camera, image, [1], depth=depth, use_wapr=True, wapr_mode=mode, pnp_op='epnp')
    entry = result[1]
    report = result['wapr_postprocess']
    raw_count = 1 if mode == 'refine' else 2
    kept = list(range(raw_count))[1:] if suppress_first else list(range(raw_count))
    assert len(report['raw_records']) == raw_count
    assert report['wapr_mode'] == entry['wapr_mode'] == mode
    assert entry['wapr_instance_indices'].tolist() == kept
    np.testing.assert_allclose(entry['conf'], np.asarray([.8, .9])[kept])
    assert len(entry['pred_mask']) == len(kept) == len(entry['Rts'])
    for index, pose in zip(kept, entry['Rts']):
        np.testing.assert_allclose(pose[:3, 3], [40., 50., 500. + 1000. * index])
    if mode == 'estimate':
        assert 'pnp_success' not in entry and 'Rts_hccepose' not in entry
        assert report['skipped_instances'] == []
    else:
        np.testing.assert_array_equal(entry['pnp_success'], np.asarray([True, False])[kept])
        assert len(entry['Rts_hccepose']) == len(kept)
        assert report['skipped_instances'] == ([{'object_id': 1, 'instance_index': 1, 'status': 'pnp_failed'}]
                                                if mode == 'refine' else [])
    assert len(calls) == 1


def test_refine_all_failed_does_not_load_wapr_weights(runtime, monkeypatch):
    tester = object.__new__(runtime.Tester)
    tester.device = torch.device('cpu')
    tester.wapr_registered = False
    prediction = {'pred_mask': torch.ones(2, 128, 128), 'conf': torch.ones(2),
                  'Rts': np.tile(np.eye(4), (2, 1, 1)), 'pnp_success': np.array([False, True])}
    monkeypatch.setattr(refinement_group, 'WAPREstimator', None)
    report = tester._run_wapr_estimation({1: prediction}, [1], np.eye(3),
                                         np.zeros((32, 32, 3), np.uint8), np.ones((32, 32)),
                                         .75, .5, mode='refine')
    assert report['raw_records'] == report['retained_records'] == []
    assert [row['status'] for row in report['skipped_instances']] == ['pnp_failed', 'invalid_pnp_pose']
    assert prediction['Rts'].shape == (0, 4, 4) and not tester.wapr_registered


@pytest.mark.parametrize('options', [{'use_wapr': True, 'wapr_mode': 'unknown'}, {'wapr_mode': 'hybrid'}])
def test_invalid_wapr_mode_fails_before_models(runtime, options):
    tester = object.__new__(runtime.Tester)
    with pytest.raises(ValueError, match='WAPR mode|use_wapr'):
        tester.predict(np.eye(3), np.zeros((32, 32, 3), np.uint8), [1], **options)
