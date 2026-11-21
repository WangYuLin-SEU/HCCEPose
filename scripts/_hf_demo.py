"""Fixed public demo metadata and local validation shared by the download tools.
下载工具共用的固定公开示例元数据与本地校验。
"""
import fnmatch
import hashlib
import json
import math
import os

# Public resource revision; bytes and weight digests match the documentation.
# 公开资源版本；字节数和权重摘要与文档一致。
HF_REVISION = '4e482e540314622981ccd4fce5beddac7f9d3e1d'
CHECKPOINT_PATH = 'demo-bin-picking/HccePose/obj_01/best_score/0_9837step824500'
REQUIRED_DEMO_FILES = {
    'demo-bin-picking/models/models_info.json': (941, None),
    'demo-bin-picking/models/obj_000001.ply': (128911, None),
    CHECKPOINT_PATH: (349827430, '3308469a5082feb98181286026003e0b00ea8a349b8f0921006873a177dd4c70'),
    'demo-bin-picking/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt':
        (114453888, '5a958cf1d90d3529f02ba1021aab164a369316f187f98fb01b151c484d3cfd13'),
    'test_imgs_RGBD/000003_rgb.png': (208215, None),
    'test_imgs_RGBD/000003_depth.png': (59239, None),
    'test_imgs_RGBD/000003_camK.json': (184, None),
}


def validate_selected_demo(root, patterns):
    """Validate selected known demo files; report extra checkpoints without deleting them.
    校验本次选中的已知示例文件；发现额外权重时报错，不删除文件。

    Other presets may select additional training resources; this fixed manifest
    validates only the seven documented inference files matched by the patterns.
    其他预设可能选择更多训练资源；此固定清单只校验规则匹配到的七项推理文件。
    """
    selected = [name for name in REQUIRED_DEMO_FILES
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
    if CHECKPOINT_PATH in selected:
        checkpoint = os.path.join(root, *CHECKPOINT_PATH.split('/'))
        directory = os.path.dirname(checkpoint)
        if not os.path.isdir(directory):
            raise FileNotFoundError(checkpoint)
        extras = [os.path.join(directory, name) for name in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, name)) and name != os.path.basename(checkpoint)]
        if extras:
            raise RuntimeError('Unexpected checkpoint files: ' + ', '.join(sorted(extras)))
    for relative_path in selected:
        expected_size, expected_digest = REQUIRED_DEMO_FILES[relative_path]
        path = os.path.join(root, *relative_path.split('/'))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            raise RuntimeError(f'File size mismatch: {path}; expected {expected_size}, got {actual_size}')
        if expected_digest is not None:
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                raise RuntimeError(f'SHA-256 mismatch: {path}')
    camera_relative = 'test_imgs_RGBD/000003_camK.json'
    if camera_relative in selected:
        # Write a separate matrix only after validation; keep original calibration intact.
        # 全部校验通过后才另存矩阵，保留原始标定文件。
        camera_dir = os.path.join(root, 'test_imgs_RGBD')
        with open(os.path.join(camera_dir, '000003_camK.json'), encoding='utf-8') as handle:
            camera = json.load(handle)
        values = [float(camera[key]) for key in ('fx', 'fy', 'cx', 'cy')]
        if not all(math.isfinite(x) for x in values) or values[0] <= 0 or values[1] <= 0:
            raise ValueError('Invalid frame intrinsics in 000003_camK.json')
        fx, fy, cx, cy = values
        matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        with open(os.path.join(camera_dir, '000003_camera_matrix.json'), 'w', encoding='utf-8') as handle:
            json.dump(matrix, handle, indent=2)
    return selected
