"""Check public download contracts and retained toolkit/reader functionality.
检查公开下载约定及保留的工具包、读取功能。
"""
import hashlib
import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

project_dir = Path(__file__).resolve().parents[1]
# Load the adjacent command helpers without depending on the working directory.
# 定位项目中的命令辅助模块，不依赖启动目录。
sys.path.insert(0, str(project_dir / 'scripts'))
import download_hf_assets as hf
import wget_hf_demo_assets as wget


@pytest.fixture
def small_demo(tmp_path, monkeypatch):
    """Small cache fixture; production metadata stays unchanged.
    使用小型缓存夹具，不改生产元数据。
    """
    required = {}
    for relative, (_, digest) in hf.REQUIRED_DEMO_FILES.items():
        data = (json.dumps({'fx': 20, 'fy': 21, 'cx': 3, 'cy': 4}).encode()
                if relative.endswith('_camK.json') else b'known demo fixture')
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        required[relative] = (len(data), hashlib.sha256(data).hexdigest() if digest else None)
    monkeypatch.setattr(hf, 'REQUIRED_DEMO_FILES', required)
    return tmp_path, required


def test_demo_preset_and_pinned_sdk_boundary(small_demo, monkeypatch):
    root, required = small_demo
    calls = []
    monkeypatch.setattr(hf, 'snapshot_download', lambda **kwargs: calls.append(kwargs) or str(root))
    monkeypatch.delenv('HF_ENDPOINT', raising=False)
    patterns = hf._patterns_for_presets(['demo'])
    assert set(patterns) == set(required)
    hf._run_download(str(root), patterns)
    assert calls[0]['revision'] == hf.HF_REVISION
    assert calls[0]['token'] is False
    assert calls[0]['repo_type'] == 'dataset'
    assert calls[0]['endpoint'] == 'https://huggingface.co'
    assert calls[0]['local_dir'] == str(root)
    assert json.loads((root / 'test_imgs_RGBD/000003_camera_matrix.json').read_text()) == [[20, 0, 3], [0, 21, 4], [0, 0, 1]]


@pytest.mark.parametrize('damage', ['missing', 'empty', 'truncated', 'wrong_digest', 'extra_checkpoint'])
def test_bad_cache_is_reported_without_camera_write_or_deletion(small_demo, damage):
    root, required = small_demo
    checkpoint = root / hf.CHECKPOINT_PATH
    camera = root / 'test_imgs_RGBD/000003_camK.json'
    camera_before = camera.read_bytes()
    extra = None
    if damage == 'missing':
        checkpoint.unlink()
    elif damage == 'empty':
        checkpoint.write_bytes(b'')
    elif damage == 'truncated':
        checkpoint.write_bytes(b'x')
    elif damage == 'wrong_digest':
        checkpoint.write_bytes(b'x' * required[hf.CHECKPOINT_PATH][0])
    else:
        extra = checkpoint.parent / 'README.txt'
        extra.write_text('unexpected file')
    with pytest.raises((FileNotFoundError, RuntimeError)):
        hf.validate_selected_demo(str(root), list(required))
    assert camera.read_bytes() == camera_before
    assert not (camera.parent / '000003_camera_matrix.json').exists()
    if extra:
        assert extra.read_text() == 'unexpected file'
    if damage != 'missing':
        assert checkpoint.exists()


def test_subset_validation_does_not_require_unselected_weights(small_demo):
    root, required = small_demo
    (root / hf.CHECKPOINT_PATH).unlink()
    selected = hf.validate_selected_demo(str(root), ['test_imgs_RGBD/000003_camK.json'])
    assert selected == ['test_imgs_RGBD/000003_camK.json']


def test_wget_pinned_tree_and_failure_are_not_partial_success(monkeypatch):
    assert wget.dataset_resolve_base('https://example.invalid').endswith('/resolve/' + hf.HF_REVISION)
    urls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b'[]'
    monkeypatch.setattr(wget.urllib.request, 'urlopen', lambda url, **kwargs: urls.append(url) or Response())
    assert wget.tree_api_dataset('test_imgs_RGBD', 'https://example.invalid') == []
    assert '/tree/' + hf.HF_REVISION + '/' in urls[0]
    def fail(*args): raise OSError('offline')
    monkeypatch.setattr(wget, 'tree_api_dataset', fail)
    with pytest.raises(RuntimeError, match='pinned resource tree'):
        wget.collect_dataset_manifest('https://example.invalid')


@pytest.mark.parametrize('mode', ['network_error', 'wrong_size', 'complete'])
def test_wget_replaces_existing_file_only_after_complete_download(tmp_path, monkeypatch, mode):
    dest = tmp_path / 'asset.bin'
    dest.write_bytes(b'old content')
    def run(args, check):
        if mode == 'network_error':
            raise subprocess.CalledProcessError(1, args)
        temporary = Path(args[args.index('-O') + 1])
        temporary.write_bytes(b'abc' if mode == 'complete' else b'x')
    monkeypatch.setattr(wget.subprocess, 'run', run)
    ok = wget.wget_file('https://example.invalid/asset', str(dest), expected_size=3)
    assert ok is (mode == 'complete')
    assert dest.read_bytes() == (b'abc' if mode == 'complete' else b'old content')


def test_retained_capture_reader_and_units(tmp_path):
    sys.path.insert(0, str(project_dir))
    reader = importlib.import_module('Refinement.refinement_group')
    assert cv2.imwrite(str(tmp_path / '000003_rgb.png'), np.zeros((4, 6, 3), np.uint8))
    assert cv2.imwrite(str(tmp_path / '000003_depth.png'), np.full((4, 6), 500, np.uint16))
    (tmp_path / '000003_camK.json').write_text(json.dumps({'fx': 10, 'fy': 11, 'cx': 2, 'cy': 1}))
    assert reader.list_capture_frame_names(str(tmp_path)) == ['000003']
    image, raw_depth, meter_depth, K = reader.load_capture_frame(str(tmp_path), '000003')
    assert image.shape == (4, 6, 3)
    np.testing.assert_array_equal(raw_depth, 500)
    np.testing.assert_allclose(meter_depth, 0.5)
    np.testing.assert_array_equal(K, [[10, 0, 2], [0, 11, 1], [0, 0, 1]])


def test_bundled_archives_keep_functional_sources_without_embedded_history():
    for archive in ['bop_toolkit.zip', 'blenderproc.zip']:
        with zipfile.ZipFile(project_dir / archive) as z:
            names = z.namelist()
            assert not any('.git' in Path(name).parts or '__pycache__' in Path(name).parts or name.endswith('.pyc') for name in names)
            if archive == 'bop_toolkit.zip':
                for name in ['scripts/eval_bop19_pose.py', 'scripts/eval_calc_errors.py', 'scripts/eval_calc_scores.py', 'bop_toolkit_lib/renderer.py']:
                    assert 'bop_toolkit/' + name in names
                assert 'BOP_PATH' in z.read('bop_toolkit/bop_toolkit_lib/config.py').decode()
