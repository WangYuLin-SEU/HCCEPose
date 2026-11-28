"""Check score gates and the public NMS call boundary without pose weights.
检查分数门槛及公开 NMS 调用边界，不需要位姿权重。
"""
import copy
import json
import os
import sys

import cv2
import numpy as np
import pytest
import trimesh

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import this checkout, independent of the directory that starts pytest.
# 从当前副本导入，不依赖 pytest 的启动目录。
sys.path.insert(0, project_dir)
from HccePose import tester as tester_module
from HccePose.tester import Tester as PoseTester


def record(object_id=1, instance_index=0, score=0.9):
    pose = np.eye(4)
    pose[2, 3] = 0.5
    return {'object_id': object_id, 'instance_index': instance_index,
            'score_6d': score, 'pose_4x4_m': pose.tolist()}


@pytest.fixture
def nms_backend(monkeypatch):
    calls = []
    keep = None

    def nms(poses, mesh, K, height, width, iou_max, device):
        calls.append({'poses': copy.deepcopy(poses), 'mesh': mesh, 'K': K.copy(),
                      'shape': (height, width), 'iou': iou_max, 'device': device})
        return [pose for pose in poses if keep is None or pose['input_index'] in keep]

    monkeypatch.setattr(tester_module, 'rendered_mask_iou_nms_gpu', nms)

    def select(indices):
        nonlocal keep
        keep = set(indices)
    return calls, select


def camera():
    return np.array([[30, 0, 12], [0, 30, 8], [0, 0, 1]], np.float32)


def test_threshold_equality_scores_above_one_and_no_mutation(nms_backend):
    records = [record(1, 0, 0.749), record(1, 1, 0.75), record(1, 2, 1.001935)]
    before = copy.deepcopy(records)
    mesh = object()
    report = PoseTester.postprocess_wapr(records, {1: mesh}, camera(), (16, 24, 3), 0.75, 0.5)
    assert records == before
    assert report['raw_records'] == records
    assert report['kept_input_indices'] == [1, 2]
    assert [row['score_6d'] for row in report['retained_records']] == [0.75, 1.001935]
    assert [d['status'] for d in report['decisions']] == ['below_score_threshold', 'kept', 'kept']
    calls, _ = nms_backend
    assert len(calls) == 1 and calls[0]['mesh'] is mesh
    assert calls[0]['shape'] == (16, 24)
    assert calls[0]['iou'] == 0.5
    np.testing.assert_array_equal(calls[0]['K'], camera())
    assert [row['input_index'] for row in calls[0]['poses']] == [1, 2]


def test_per_object_meshes_and_original_order(nms_backend):
    calls, select = nms_backend
    select([1, 2, 4])
    records = [record(1, 0, 0.75), record(2, 0, 0.9), record(1, 1, 1.0),
               record(1, 2, 0.7), record(1, 3, 0.8)]
    meshes = {1: object(), 2: object()}
    report = PoseTester.postprocess_wapr(records, meshes, camera(), (16, 24), 0.75, 0.4, 'cuda:2')
    assert report['kept_input_indices'] == [1, 2, 4]
    assert report['retained_records'] == [records[i] for i in [1, 2, 4]]
    assert [c['mesh'] for c in calls] == [meshes[1], meshes[2]]
    assert all(c['device'] == 'cuda:2' and c['iou'] == 0.4 for c in calls)
    assert [[p['obj_id'] for p in c['poses']] for c in calls] == [[1, 1, 1], [2]]
    assert report['decisions'][0]['status'] == 'not_retained_by_mask_nms'
    assert report['decisions'][3]['status'] == 'below_score_threshold'


@pytest.mark.parametrize('records', [[], [record(score=0.5)]])
def test_empty_or_low_scores_need_no_backend(records, monkeypatch):
    monkeypatch.setattr(tester_module, 'rendered_mask_iou_nms_gpu', None)
    report = PoseTester.postprocess_wapr(records, {}, camera(), (16, 24), 0.75, 0.5)
    assert report['retained_records'] == []
    assert len(report['raw_records']) == len(records)
    assert tester_module.rendered_mask_iou_nms_gpu is None


def test_backend_removal_not_mislabeled_confirmed_duplicate(nms_backend):
    _, select = nms_backend
    select([])
    report = PoseTester.postprocess_wapr([record()], {1: object()}, camera(), (16, 24), 0.75, 0.5)
    assert report['decisions'][0]['status'] == 'not_retained_by_mask_nms'
    assert report['raw_records'][0]['score_6d'] == 0.9


@pytest.mark.parametrize('field,value', [
    ('object_id', True), ('object_id', 0), ('object_id', 1.5),
    ('instance_index', -1), ('instance_index', True),
    ('score_6d', float('nan')), ('score_6d', float('inf')), ('score_6d', True),
    ('pose_4x4_m', [[1, 0], [0, 1]]),
])
def test_bad_records_rejected_before_backend(field, value, nms_backend):
    item = record()
    item[field] = value
    with pytest.raises(ValueError):
        PoseTester.postprocess_wapr([item], {1: object()}, camera(), (16, 24), 0.75, 0.5)
    assert not nms_backend[0]


def test_duplicate_identity_and_nonhomogeneous_pose_rejected(nms_backend):
    with pytest.raises(ValueError, match='Duplicate instance'):
        PoseTester.postprocess_wapr([record(), record()], {1: object()}, camera(), (16, 24), 0.75, 0.5)
    row = record()
    row['pose_4x4_m'][3][0] = 1
    with pytest.raises(ValueError, match='homogeneous'):
        PoseTester.postprocess_wapr([row], {1: object()}, camera(), (16, 24), 0.75, 0.5)
    assert not nms_backend[0]


@pytest.mark.parametrize('score,iou', [(float('nan'), 0.5), (0.75, -0.1), (0.75, 1.1), (0.75, float('nan'))])
def test_invalid_thresholds_rejected(score, iou, nms_backend):
    with pytest.raises(ValueError):
        PoseTester.postprocess_wapr([record()], {1: object()}, camera(), (16, 24), score, iou)
    assert not nms_backend[0]


def test_missing_mesh_is_not_unfiltered_success(nms_backend):
    with pytest.raises(ValueError, match='Missing meter mesh'):
        PoseTester.postprocess_wapr([record()], {}, camera(), (16, 24), 0.75, 0.5)
    assert not nms_backend[0]


def example_files(tmp_path):
    dataset = tmp_path / 'objects'
    (dataset / 'models').mkdir(parents=True)
    trimesh.creation.box(extents=(100, 60, 40)).export(dataset / 'models/obj_000001.ply')
    image = tmp_path / 'image.png'
    assert cv2.imwrite(str(image), np.zeros((16, 24, 3), np.uint8))
    intrinsics = tmp_path / 'camera.json'
    intrinsics.write_text(json.dumps(camera().tolist()))
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps([record(1, 0, 0.9), record(1, 1, 0.5)]))
    return dataset, image, intrinsics, raw


def test_file_recipe_units_output_and_raw_preservation(tmp_path, nms_backend):
    dataset, image, intrinsics, raw = example_files(tmp_path)
    original_bytes = raw.read_bytes()
    output = tmp_path / 'output/filtered.json'
    report = PoseTester.postprocess_saved_wapr(str(dataset), str(image), str(intrinsics), str(raw), str(output), 0.75, 0.5)
    assert raw.read_bytes() == original_bytes
    assert json.loads(output.read_text()) == report
    assert len(report['raw_records']) == 2 and len(report['retained_records']) == 1
    np.testing.assert_allclose(nms_backend[0][0]['mesh'].extents, [0.1, 0.06, 0.04])


@pytest.mark.parametrize('mode', ['same_path', 'symlink', 'hardlink'])
def test_raw_file_cannot_be_overwritten(tmp_path, mode, nms_backend):
    dataset, image, intrinsics, raw = example_files(tmp_path)
    output = raw
    if mode == 'symlink':
        output = tmp_path / 'raw-link.json'
        output.symlink_to(raw)
    elif mode == 'hardlink':
        output = tmp_path / 'raw-hardlink.json'
        os.link(raw, output)
    original = raw.read_bytes()
    with pytest.raises(ValueError, match='separate output'):
        PoseTester.postprocess_saved_wapr(str(dataset), str(image), str(intrinsics), str(raw), str(output), 0.75, 0.5)
    assert raw.read_bytes() == original and not nms_backend[0]


def test_backend_error_is_not_silent_success(nms_backend, monkeypatch):
    def fail(*args):
        raise RuntimeError('renderer unavailable')
    monkeypatch.setattr(tester_module, 'rendered_mask_iou_nms_gpu', fail)
    with pytest.raises(RuntimeError, match='renderer unavailable'):
        PoseTester.postprocess_wapr([record()], {1: object()}, camera(), (16, 24), 0.75, 0.5)
