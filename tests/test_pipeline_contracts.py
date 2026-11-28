"""Exercise inference, training-data and optional-backend boundaries on CPU.
在 CPU 上检查推理、训练数据与可选后端的数据对应关系。
"""
import importlib
import json
import os
import runpy
import subprocess
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import ultralytics
from torchvision.transforms.functional import normalize

project_dir = Path(__file__).resolve().parents[1]
# Import project code relative to this test file, including external pytest launchers.
# 从当前测试文件定位项目代码，支持从仓库外启动 pytest。
sys.path.insert(0, str(project_dir))

from HccePose import PnP_solver, network_model
from HccePose.metric import add_s
from HccePose.network_model import load_checkpoint
from HccePose.tester import Tester as HccePoseTester
from Refinement import MegaPose
from Refinement.refinement_group import Refinement_MP, convert_depth_to_meter
from yolo_train import label


@pytest.fixture(scope='module')
def loader(tmp_path_factory):
    # Use the bundled public toolkit, not a machine-specific checkout.
    # 使用随仓库提供的公开工具包，不依赖机器上的其他副本。
    toolkit_root = tmp_path_factory.mktemp('bop_toolkit')
    with zipfile.ZipFile(project_dir / 'bop_toolkit.zip') as archive:
        archive.extractall(toolkit_root)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(project_dir))
        patch.syspath_prepend(str(toolkit_root))
        patch.syspath_prepend(str(toolkit_root / 'bop_toolkit'))
        yield importlib.import_module('HccePose.bop_loader')


def test_detector_rgb_does_not_change_hcce_bgr_crops(loader):
    tester = object.__new__(HccePoseTester)
    tester.bop_dataset_item = SimpleNamespace(obj_id_list=[1])
    tester.HccePose_Item_info = {1: {'keypoints_': [1]}}
    tester.device = torch.device('cpu')
    tester.crop_size = 32
    tester.show_op = False
    image = np.full((32, 32, 3), [11, 53, 197], np.uint8)
    observed = {}

    def detect(tensor, **kwargs):
        observed['detector'] = tensor[0, :, 16, 16].clone()
        boxes = SimpleNamespace(xywh=torch.tensor([[16., 16., 8., 8.]]),
                                cls=torch.zeros(1), conf=torch.ones(1))
        return [SimpleNamespace(boxes=boxes)]

    def infer(crops, boxes):
        observed['crop'] = crops[0, :, 16, 16].clone()
        mask = torch.zeros((1, 2, 4))
        xyz = torch.zeros((1, 2, 4, 3))
        return dict(pred_mask=mask, coord_2d_image=torch.zeros((1, 2, 4, 2)),
                    pred_front_code_obj=xyz, pred_back_code_obj=xyz,
                    pred_front_code=xyz, pred_back_code=xyz)

    tester.model_yolo = detect
    tester.HccePose_Runtime_Item = {1: SimpleNamespace(inference_batch=infer)}
    tester.predict(np.array([[80., 0, 16], [0, 80., 16], [0, 0, 1]]), image, [1])
    torch.testing.assert_close(observed['detector'], torch.tensor([197., 53., 11.]) / 255)
    expected = normalize(torch.tensor([11., 53., 197.])[:, None, None] / 255,
                         loader.IMAGENET_MEAN_BGR, loader.IMAGENET_STD_BGR)[:, 0, 0]
    torch.testing.assert_close(observed['crop'], expected)
    np.testing.assert_array_equal(image[0, 0], [11, 53, 197])


def test_continuous_and_flat_symmetries_share_pose_convention(loader):
    dataset = object.__new__(loader.rendering_bop_dataset_back_front)
    dataset.dataset_info = {'obj_info': {'obj_000001': [{}]}}
    dataset.bop_dataset_item = SimpleNamespace(model_info={'1': {
        'symmetries_continuous': [{'axis': [0, 0, 1], 'offset': [0, 0, 0]}],
    }})
    dataset.update_obj_id(1, 'unused.ply')
    rotation, _ = cv2.Rodrigues(np.array([0., 0., 1.]))
    translation = np.array([[2.], [3.], [400.]])
    records = dataset.model_info_obj['symmetries_discrete']
    assert isinstance(records[0], dict)
    matrices = []
    for record in records:
        transform = np.eye(4)
        transform[:3, :3] = record['R']
        transform[:3, 3] = record['t'].reshape(3)
        matrices.append(transform.reshape(-1).tolist())
    from_records = dataset.modified_sym_Rt(rotation, translation, dataset.model_info_obj)
    from_matrices = dataset.modified_sym_Rt(rotation, translation, {'symmetries_discrete': matrices})
    np.testing.assert_allclose(from_records[0], from_matrices[0])
    np.testing.assert_allclose(from_records[1], from_matrices[1])


@pytest.mark.parametrize('damage', ['front_missing', 'back_missing', 'corrupt', 'wrong_shape', 'none'])
def test_training_requires_full_frame_coordinate_labels(loader, tmp_path, damage):
    dataset = object.__new__(loader.train_bop_dataset_back_front)
    image_path, mask_path = tmp_path / 'rgb.png', tmp_path / 'mask.png'
    cv2.imwrite(str(image_path), np.full((128, 128, 3), 120, np.uint8))
    cv2.imwrite(str(mask_path), np.full((128, 128), 255, np.uint8))
    dataset.dataset_info = {'obj_info': {'obj_000001': [{
        'rgb': str(image_path), 'mask_visib_path': str(mask_path), 'mask_path': str(mask_path),
        'scene': '000000', 'bbox_visib': [0, 0, 128, 128],
    }]}}
    dataset.current_obj_id = 1
    dataset.crop_size_img, dataset.crop_size_gt = 256, 128
    dataset.padding_ratio, dataset.aug_op = 1.5, None
    dataset.composed_transforms_img = lambda value: value
    dataset.target_dir_front, dataset.target_dir_back = str(tmp_path / 'front'), str(tmp_path / 'back')
    paths = []
    for folder in (dataset.target_dir_front, dataset.target_dir_back):
        path = Path(folder) / '000000/mask.png'
        path.parent.mkdir(parents=True)
        cv2.imwrite(str(path), np.full((128, 128, 3), 90, np.uint8))
        paths.append(path)
    if damage.endswith('_missing'):
        paths[damage == 'back_missing'].unlink()
    elif damage == 'corrupt':
        paths[0].write_bytes(b'not a PNG')
    elif damage == 'wrong_shape':
        cv2.imwrite(str(paths[0]), np.zeros((64, 64, 3), np.uint8))
    if damage != 'none':
        with pytest.raises((FileNotFoundError, ValueError), match='coordinate label'):
            dataset[0]
    else:
        _, mask, front, back = dataset[0]
        assert mask.count_nonzero() > 0
        assert front.count_nonzero() > 0 and back.count_nonzero() > 0


def test_back_render_uses_complete_canonical_pose(loader, tmp_path):
    rotation = np.diag([-1., -1., 1.])
    translation = np.array([[0.], [0.], [100.]])
    symmetry = np.eye(4)
    symmetry[:3, :3], symmetry[:3, 3] = rotation, [20., 0., 0.]
    dataset = object.__new__(loader.rendering_bop_dataset_back_front)
    dataset.current_obj_id = 1
    dataset.dataset_info = {'obj_info': {'obj_000001': [{
        'scene': '000000', 'mask_path': '000000_000000.png',
        'cam_R_m2c': rotation.reshape(-1).tolist(), 'cam_t_m2c': translation.reshape(-1).tolist(),
        'cam_K': [100, 0, 1, 0, 100, 1, 0, 0, 1],
    }]}}
    for folder in ('front', 'back'):
        (tmp_path / folder / '000000').mkdir(parents=True)
    dataset.target_dir_front, dataset.target_dir_back = str(tmp_path / 'front'), str(tmp_path / 'back')
    dataset.model_info_obj = {'symmetries_discrete': [symmetry.reshape(-1).tolist()]}
    dataset.min_xyz, dataset.div_v = np.full(3, -30.), np.full(3, 60.)
    dataset.pnp_solve_re_4p = lambda pose, *args: pose.copy()
    calls = []

    class Renderer:
        def render_object(self, object_id, R, t, *args, draw_back):
            calls.append((R.copy(), t.copy()))
            self.depth = np.full((2, 2), 110. if draw_back else 100.)

    dataset.renderer_vispy = Renderer()
    expected = dataset.modified_sym_Rt(rotation, translation, dataset.model_info_obj, e_rot=rotation)
    dataset[0]
    np.testing.assert_allclose(calls[0][1], translation)
    np.testing.assert_allclose(calls[1][0], expected[0])
    np.testing.assert_allclose(calls[1][1], expected[1])


@pytest.mark.parametrize('symmetry', [{}, {'symmetries_discrete': []},
    {'symmetries_continuous': []}, {'symmetries_discrete': [np.eye(4).reshape(-1).tolist()]}])
def test_empty_symmetry_metadata_keeps_add(loader, symmetry):
    points = {'pts': np.array([[0., 0., 0.], [10., 0., 0.], [2., 1., 0.]])}
    gt = [[np.eye(3), np.zeros((3, 1))]]
    prediction = [[np.diag([-1., -1., 1.]), np.array([[10.], [0.], [0.]])]]
    result = add_s(points, {'diameter': 10., **symmetry}, gt, prediction)
    assert result[0] == 0
    assert result[2][0] == pytest.approx(8.774851773445587)


@pytest.mark.parametrize('symmetry', [
    {'symmetries_discrete': [np.diag([-1., -1., 1., 1.]).reshape(-1).tolist()]},
    {'symmetries_continuous': [{'axis': [0, 0, 1], 'offset': [0, 0, 0]}]},
])
def test_nontrivial_symmetry_keeps_adds(loader, symmetry):
    points = {'pts': np.array([[0., 0., 0.], [10., 0., 0.], [2., 1., 0.]])}
    gt = [[np.eye(3), np.zeros((3, 1))]]
    prediction = [[np.diag([-1., -1., 1.]), np.array([[10.], [0.], [0.]])]]
    result = add_s(points, {'diameter': 10., **symmetry}, gt, prediction)
    assert result[0] == 1
    assert result[2][0] == pytest.approx(.7453559924999299)


@pytest.mark.parametrize('mode', [1, 2])
def test_pnp_prior_branch_identity_is_solver_independent(monkeypatch, mode):
    markers = {None: 1, 'b': 2, 'bf': 3, 'bfu': 4}

    def solve(*args, bfu=None, **kwargs):
        marker = markers[bfu]
        return {'rot': np.eye(3), 'tvecs': np.array([[marker], [0.], [0.]]),
                'success': True, 'inliers': np.zeros((marker, 1), np.int32)}

    monkeypatch.setattr(PnP_solver, 'solve_PnP', solve)
    inputs = (np.ones((2, 3)), np.zeros((2, 3, 3)), np.zeros((2, 3, 3)),
              np.zeros((2, 3, 2)), np.eye(3))
    prior = np.zeros(15)
    prior[13] = 1
    chosen = PnP_solver.solve_PnP_comb(inputs, keypoints_=prior, pnp_op=mode)
    assert chosen['tvecs'][0, 0] == 3
    candidates = PnP_solver.solve_PnP_comb(inputs, pnp_op=mode, train=True)
    assert [item['tvecs'][0, 0] for item in candidates] == [1, 2, 3, 4]


@pytest.mark.parametrize('damage', ['none', 'optimizer_missing', 'optimizer_invalid', 'step_missing', 'score_missing', 'prior_missing', 'corrupt'])
def test_resume_fails_explicitly_without_silent_restart(tmp_path, monkeypatch, damage):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    with torch.no_grad():
        model.weight.fill_(3.)
    checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                  'iteration_step': 123, 'best_score': .9, 'keypoints_': [1] * 15}
    if damage == 'optimizer_missing':
        del checkpoint['optimizer_state_dict']
    elif damage == 'optimizer_invalid':
        checkpoint['optimizer_state_dict'] = {'invalid': True}
    elif damage == 'step_missing':
        del checkpoint['iteration_step']
    elif damage == 'score_missing':
        del checkpoint['best_score']
    elif damage == 'prior_missing':
        del checkpoint['keypoints_']
    path = tmp_path / '123'
    torch.save(checkpoint, path)
    if damage == 'corrupt':
        path.write_bytes(b'not a checkpoint')
    original_load = torch.load
    monkeypatch.setattr(network_model.torch, 'load', lambda path, **kwargs: original_load(path, map_location='cpu', weights_only=False))
    restored = torch.nn.Linear(1, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters())
    if damage in ('none', 'prior_missing'):
        info = network_model.load_checkpoint(tmp_path, restored, restored_optimizer)
        assert info['iteration_step'] == 123 and info['best_score'] == .9
        assert info['keypoints_'] == ([] if damage == 'prior_missing' else [1] * 15)
        torch.testing.assert_close(restored.weight, model.weight)
    else:
        with pytest.raises(RuntimeError, match='Cannot restore checkpoint') as failure:
            network_model.load_checkpoint(tmp_path, restored, restored_optimizer)
        assert str(path) in str(failure.value)


@pytest.mark.parametrize('exists', [True, False])
def test_absent_checkpoint_starts_new_training_only(tmp_path, exists):
    path = tmp_path / 'checkpoints'
    if exists:
        path.mkdir()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    assert load_checkpoint(path, model, optimizer) == {'best_score': 0, 'iteration_step': 0, 'keypoints_': []}
    with pytest.raises(RuntimeError, match='Cannot restore checkpoint'):
        load_checkpoint(path, model)


@pytest.mark.parametrize('visible,device,expected', [
    (None, 'cuda:2', '2'), ('2,5', 'cuda:1', '5'), ('3', 'cuda:0', '3'),
    ('GPU-first,GPU-second', 'cuda:1', 'GPU-second'), ('MIG-device', 'cuda:0', 'MIG-device'),
    ('', 'cuda:0', None), ('2,5', 'cuda:2', None), ('-1', 'cuda:0', None),
])
def test_megapose_cuda_visibility_mapping(tmp_path, monkeypatch, visible, device, expected):
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    backend = Refinement_MP(cache_dir=tmp_path, device=device)
    if expected is None:
        with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'):
            backend._make_env()
    else:
        assert backend._make_env()['CUDA_VISIBLE_DEVICES'] == expected
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == visible


@pytest.mark.parametrize('raw,scale,expected', [(3000, 1e-3, 3.), (30000, 1e-4, 3.), (.4, 1., .4), (0, 1e-3, 0.)])
def test_explicit_depth_units_do_not_depend_on_distance(raw, scale, expected):
    original = np.full((2, 3), raw, np.float32)
    converted = convert_depth_to_meter(original, scale)
    np.testing.assert_allclose(converted, expected)
    np.testing.assert_array_equal(original, raw)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_depth_scale_is_rejected(scale):
    with pytest.raises(ValueError, match='depth_scale_to_m'):
        convert_depth_to_meter(np.ones((2, 3)), scale)


def test_ply_texture_uses_declared_name_and_file_type(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'pymeshlab', types.ModuleType('pymeshlab'))
    script = runpy.run_path(str(project_dir / 's1_p1_obj_rename_center.py'))
    (tmp_path / 'textures').mkdir()
    source = tmp_path / 'textures/surface.jpg'
    source.write_bytes(b'actual texture bytes')
    ply = tmp_path / 'board.ply'
    ply.write_bytes(b'ply\nformat binary_little_endian 1.0\ncomment TextureFile textures/surface.jpg\nend_header\n\xff\x00')
    output = tmp_path / 'obj_000001.ply'
    texture = script['_copy_ply_texture'](str(ply), str(output))
    assert texture == str(tmp_path / 'obj_000001.jpg')
    assert Path(texture).read_bytes() == source.read_bytes()
    output.write_text('ply\ncomment TextureFile textures/surface.jpg\nend_header\n')
    script['modify_ply_texture_filename'](str(output), str(output), Path(texture).name)
    assert 'comment TextureFile obj_000001.jpg' in output.read_text()
    source.unlink()
    with pytest.raises(FileNotFoundError):
        script['_copy_ply_texture'](str(ply), str(output))


@pytest.mark.parametrize('mode', ['already_complete', 'empty_directory', 'zero_weight', 'failure', 'missing_output'])
def test_yolo_wrapper_requires_weight_and_stops_on_failure(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    weight = tmp_path / 'xxx/xxx/demo-bin-picking/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt'
    weight.parent.mkdir(parents=True)
    if mode == 'already_complete':
        weight.write_bytes(b'complete')
    elif mode == 'zero_weight':
        weight.touch()
    calls = []

    def train(args, check):
        calls.append(args)
        assert args[0] == sys.executable and check is True
        if mode == 'failure':
            raise subprocess.CalledProcessError(1, args)
        if mode != 'missing_output':
            weight.write_bytes(b'trained detector')

    monkeypatch.setattr(subprocess, 'run', train)
    if mode in ('failure', 'missing_output'):
        with pytest.raises((subprocess.CalledProcessError, FileNotFoundError)):
            runpy.run_path(str(project_dir / 's3_p2_train_yolo.py'), run_name='__main__')
        assert len(calls) == 1
    else:
        runpy.run_path(str(project_dir / 's3_p2_train_yolo.py'), run_name='__main__')
        assert len(calls) == (mode != 'already_complete')
        assert weight.stat().st_size > 0


def test_yolo_conversion_entry_keeps_single_relative_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    info = tmp_path / 'xxx/xxx/demo-bin-picking/models/models_info.json'
    info.parent.mkdir(parents=True)
    info.write_text('{"2": {}, "1": {}}')
    calls = []
    monkeypatch.setattr(label, 'convert_train_pbr_2_yolo', lambda *args: calls.append(args))
    monkeypatch.setattr(label, 'generate_yaml', lambda *args: None)
    runpy.run_path(str(project_dir / 's3_p1_prepare_yolo_label.py'), run_name='__main__')
    assert calls == [('xxx/xxx/demo-bin-picking/train_pbr', 'xxx/xxx/demo-bin-picking/yolo11/train_obj_s', ['2', '1'])]


@pytest.mark.parametrize('absolute', [False, True])
def test_yolo_model_save_path_matches_inference_layout(tmp_path, monkeypatch, absolute):
    monkeypatch.chdir(tmp_path)
    config = Path('data/demo/yolo11/train_obj_s/yolo_configs/data_objs.yaml')
    config.parent.mkdir(parents=True)
    config.write_text('names: [part]')
    saved = []

    class Model:
        def __init__(self, *args): pass
        def train(self, **kwargs): pass
        def save(self, path):
            Path(path).write_bytes(b'weights')
            saved.append(path)

    monkeypatch.setattr(ultralytics, 'YOLO', Model)
    script = runpy.run_path(str(project_dir / 'yolo_train/train.py'))
    script['train_yolo11']('detection', str(config.resolve() if absolute else config), 1, 1, 640, 1)
    assert Path(saved[0]).resolve() == (config.parents[1] / 'detection/obj_s/yolo11-detection-obj_s.pt').resolve()


def test_label_entry_resolves_project_data_without_rendering(loader, monkeypatch):
    calls = []
    def dataset(path):
        calls.append(path)
        return SimpleNamespace(obj_id_list=[], obj_model_list=[])
    monkeypatch.setattr(loader, 'bop_dataset', dataset)
    monkeypatch.setattr(loader, 'rendering_bop_dataset_back_front', lambda *args: object())
    runpy.run_path(str(project_dir / 's4_p1_gen_bf_labels.py'), run_name='__main__')
    assert calls == [str(project_dir / 'demo-bin-picking')]


def test_nan_recovery_resets_regular_and_sync_batch_norm(loader):
    script = runpy.run_path(str(project_dir / 's4_p2_train_bf_pbr.py'))
    # DDP exposes its network as a child module; no process group is needed here.
    # DDP 把网络注册为子模块；此处验证模块遍历，不创建分布式进程组。
    wrapped = torch.nn.Module()
    wrapped.module = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.SyncBatchNorm(3))
    for layer in wrapped.module:
        layer.running_mean.fill_(2)
        layer.running_var.fill_(3)
        layer.num_batches_tracked.fill_(4)
    script['_reset_batch_norm_running_stats'](wrapped)
    for layer in wrapped.module:
        torch.testing.assert_close(layer.running_mean, torch.zeros(3))
        torch.testing.assert_close(layer.running_var, torch.ones(3))
        assert layer.num_batches_tracked.item() == 0


@pytest.mark.parametrize('with_seed', [False, True])
def test_megapose_job_reorders_pose_score_and_identity_together(tmp_path, monkeypatch, with_seed):
    pd = pytest.importorskip('pandas')
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, 2, 3] = torch.tensor([300., 100., 200.])
    predictions = SimpleNamespace(infos=pd.DataFrame({'instance_id': [2, 0, 1],
                                                      'pose_score': [.2, .8, .5]}), poses=poses)

    class Estimator:
        def __init__(self, **kwargs): pass
        def cuda(self): return self
        def run_inference_pipeline(self, *args, **kwargs): return predictions, {}
        def forward_refiner(self, *args, **kwargs): return {'iteration=5': predictions}, {}

    class Observation:
        @classmethod
        def from_numpy(cls, *args): return cls()
        def cuda(self): return self

    class Collection:
        def __init__(self, **kwargs): pass
        def cuda(self): return self

    for name, attributes in {
        'megapose.datasets.object_dataset': {'RigidObject': SimpleNamespace, 'RigidObjectDataset': list},
        'megapose.inference.pose_estimator': {'PoseEstimator': Estimator},
        'megapose.inference.types': {'ObservationTensor': Observation},
        'megapose.inference.utils': {'load_pose_models': lambda **kwargs: (None, None, None)},
        'megapose.utils.tensor_collection': {'PandasTensorCollection': Collection},
    }.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    (tmp_path / 'bop_toolkit/bop_toolkit_lib').mkdir(parents=True)
    monkeypatch.setattr(MegaPose, '__file__', str(tmp_path / 'Refinement/MegaPose.py'))
    monkeypatch.setattr(MegaPose, '_ensure_megapose_model_artifacts', lambda *args: None)
    monkeypatch.setenv('MEGAPOSE_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    config, inputs, output = [tmp_path / name for name in ('config.json', 'inputs.npz', 'output.npz')]
    config.write_text(json.dumps({'model_name': 'megapose-1.0-RGB', 'objects': [],
                                  'labels': ['a', 'b', 'c'], 'has_depth': False, 'has_initial_poses': with_seed}))
    np.savez(inputs, rgb=np.zeros((2, 2, 3), np.uint8), K=np.eye(3),
             bboxes_xyxy=np.zeros((3, 4)), initial_poses=np.tile(np.eye(4), (3, 1, 1)))
    MegaPose._run_megapose_job(config, inputs, output)
    with np.load(output) as result:
        np.testing.assert_array_equal(result['valid_ids'], [0, 1, 2])
        np.testing.assert_array_equal(result['poses'][:, 2, 3], [100., 200., 300.])
        np.testing.assert_allclose(result['scores'], [0, 0, 0] if with_seed else [.8, .5, .2])
