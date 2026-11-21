"""Exercise RGB inference with synthetic detections and no object weights.
使用合成检测检查 RGB 推理，无需物体权重。
"""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def runtime():
    # Exercise the real Tester, transforms, projection and solver. Dataset file IO
    # is separate from this test; the class annotation only needs an import target.
    loader = ModuleType('HccePose.bop_loader')
    loader.bop_dataset = object
    loader.IMAGENET_MEAN_BGR = (0.406, 0.456, 0.485)
    loader.IMAGENET_STD_BGR = (0.225, 0.224, 0.229)
    from importlib.util import module_from_spec, spec_from_file_location

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
    from HccePose.PnP_solver import solve_PnP
    mask = np.ones((3, 3), dtype=np.uint8)
    xyz = np.zeros((3, 3, 3), dtype=np.float32)
    uv = np.zeros((3, 3, 2), dtype=np.float32)
    result = solve_PnP((mask, xyz, uv, np.eye(3, dtype=np.float32)), pnp_op=mode)
    assert not result['success']
    np.testing.assert_array_equal(result['rot'], np.eye(3))
    np.testing.assert_array_equal(result['tvecs'], np.zeros((3, 1)))


def test_invalid_checkpoint_reports_object_file_and_cause(tmp_path):
    from HccePose.network_model import load_checkpoint
    checkpoint = tmp_path / '1step.pth'
    checkpoint.write_bytes(b'not a checkpoint')
    with pytest.raises(RuntimeError, match='object 7') as failure:
        load_checkpoint(tmp_path, torch.nn.Linear(1, 1), inference_object_id=7)
    assert str(checkpoint) in str(failure.value)
    assert failure.value.__cause__ is not None
    assert load_checkpoint(tmp_path, torch.nn.Linear(1, 1))['keypoints_'] == []


def test_combined_pnp_rejects_missing_prior_before_detector(runtime):
    tester = object.__new__(runtime.Tester)
    tester.HccePose_Item_info = {7: {'keypoints_': [], 'checkpoint_path': 'example.pth'}}
    with pytest.raises(ValueError, match='object 7.*example.pth'):
        tester.predict(np.eye(3), np.zeros((32, 32, 3), np.uint8), [7])


def test_megapose_job_boundary(tmp_path, monkeypatch):
    from Refinement.refinement_group import Refinement_MP
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
