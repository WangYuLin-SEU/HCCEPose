# 模型与输入

## 配套资源

物体网格、模型信息、检测器类别顺序和逐物体位姿权重必须配套。公开资源见 [HCCEPose 资源页](https://huggingface.co/datasets/SEU-WYL/HccePose)。仓库在 `sample_data/` 中附带少量输入样例；模型权重和训练数据需单独下载。

`dataset_root` 保留公开加载器使用的目录结构：

```text
objects/
  models/models_info.json
  models/obj_000001.ply
  yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt
  HccePose/obj_01/best_score/
```

`Tester` 初始化时会加载 `models_info.json` 所列的全部物体，因此每个物体都需准备位姿权重，并保持表中的物体顺序与检测器类别顺序一致。传给 `predict` 的物体编号列表只筛选本次预测，不限制初始化时的模型加载。

在入口脚本中填写数据目录、物体编号和图像路径。检测器和逐物体位姿权重应来自可信且允许当前用途的来源，加载前按提供方的记录核对文件哈希。

## 下载一个 RGB-D 示例

仓库自带输入统一存放在 `sample_data/images/`、`sample_data/rgbd/` 和 `sample_data/videos/`；`tests/` 是代码测试。根目录的 `s4_p3_*.py` 示例读取这些自带目录。Hugging Face 下载仍保留上游名称（`test_imgs/`、`test_imgs_RGBD/`、`test_videos/`），使用额外下载的文件时，将入口中的输入路径指向实际下载目录。下方代码明确将资源下载至 `data/`。

完成[环境安装](installation.md)后，在项目根目录运行最小下载命令：

```bash
python scripts/download_hf_assets.py --preset demo --dest data --endpoint hf
```

命令下载并校验下面的七个文件，另外保存相机矩阵。官方地址无法访问时，可将参数改为 `--endpoint mirror`；两条路径请求同一固定版本，均保留完整性校验。若服务返回错误，保留已有文件，恢复连接后重试。

若已在联网机器准备好全部七个文件，可按原层级复制 `data/`，再在目标机器离线校验：

```bash
HF_HUB_OFFLINE=1 python scripts/download_hf_assets.py --preset demo --dest data --endpoint hf
```

离线模式仍检查文件大小、权重目录和权重摘要；缺件或损坏文件仍会使命令停止。

下面是等价的显式写法，可保存为项目根目录的 Python 脚本。资源保存于 `data/`，保留 Hugging Face 的目录结构；其中 `data/demo-bin-picking/` 是 `dataset_root`，无需改名为 `objects/`。

```python
import hashlib
import json
import os
from huggingface_hub import snapshot_download

project_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(project_dir, "data")

# 固定公开版本中的文件大小；两个权重同时校验 LFS 公布的 SHA-256。
checkpoint_relative_path = "demo-bin-picking/HccePose/obj_01/best_score/0_9837step824500"
required_files = {
    "demo-bin-picking/models/models_info.json": (941, None),
    "demo-bin-picking/models/obj_000001.ply": (128911, None),
    checkpoint_relative_path: (
        349827430, "3308469a5082feb98181286026003e0b00ea8a349b8f0921006873a177dd4c70"),
    "demo-bin-picking/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt": (
        114453888, "5a958cf1d90d3529f02ba1021aab164a369316f187f98fb01b151c484d3cfd13"),
    "test_imgs_RGBD/000003_rgb.png": (208215, None),
    "test_imgs_RGBD/000003_depth.png": (59239, None),
    "test_imgs_RGBD/000003_camK.json": (184, None),
}

# 下载与校验使用同一份七文件清单，不包含训练 ZIP 或对称性辅助文件。
snapshot_download(
    repo_id="SEU-WYL/HccePose",
    repo_type="dataset",
    revision="4e482e540314622981ccd4fce5beddac7f9d3e1d",
    local_dir=data_root,
    allow_patterns=list(required_files),
    endpoint="https://huggingface.co",
    token=False,
)

# 加载器会扫描该目录的全部普通文件；额外文件只报错，不自动删除。
checkpoint_path = os.path.join(data_root, *checkpoint_relative_path.split("/"))
checkpoint_dir = os.path.dirname(checkpoint_path)
extra_files = [os.path.join(checkpoint_dir, name) for name in os.listdir(checkpoint_dir)
               if os.path.isfile(os.path.join(checkpoint_dir, name))
               and name != os.path.basename(checkpoint_path)]
if extra_files:
    raise RuntimeError("Unexpected checkpoint files: " + ", ".join(sorted(extra_files)))

# 网络不可达时 SDK 可能返回已有目录；逐项校验后再转换相机文件。
for relative_path, (expected_size, expected_sha256) in required_files.items():
    path = os.path.join(data_root, *relative_path.split("/"))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise RuntimeError(f"File size mismatch: {path}; expected {expected_size}, got {actual_size}")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch: {path}")

# 使用该帧的内参；将字段式 JSON 转为入口需要的 3×3 数组。
capture_dir = os.path.join(data_root, "test_imgs_RGBD")
with open(os.path.join(capture_dir, "000003_camK.json"), encoding="utf-8") as handle:
    camera = json.load(handle)
cam_K = [[camera["fx"], 0, camera["cx"]],
         [0, camera["fy"], camera["cy"]], [0, 0, 1]]
camera_path = os.path.join(capture_dir, "000003_camera_matrix.json")
with open(camera_path, "w", encoding="utf-8") as handle:
    json.dump(cam_K, handle, indent=2)
```

该调用只读取公开资源，不需要登录。`local_dir` 决定资源实际位置，`data/.cache/huggingface/` 存储下载元数据；再次运行可复用已有文件并重新校验。清单中的大小和权重摘要对应上述固定 HF 版本，共下载七个文件，不含训练 ZIP 或对称性辅助文件。校验失败时先检查报出的文件，再重新下载；示例不会自动删除现有文件。`best_score/` 中只应有清单指定的无后缀 checkpoint，不要改名、解压或放入说明文件。

确认资源损坏后，可在该下载调用中加入 `force_download=True` 强制重新下载；额外 checkpoint 或说明文件需自行检查后移出 `best_score/`。

RGB 和深度分别位于 `data/test_imgs_RGBD/000003_rgb.png`、`data/test_imgs_RGBD/000003_depth.png`，相机数组是新生成的 `data/test_imgs_RGBD/000003_camera_matrix.json`。不要将 `demo-bin-picking/camera.json` 套用到这帧：它的分辨率和内参不同。运行方式见[第一次预测](quickstart.md)与 [WAPR 接入](rgbd-wapr.md)。

下载工具也提供最小示例预设：

```bash
python scripts/download_hf_assets.py --preset demo --dest data --endpoint hf
```

它使用同一固定七文件清单，校验后另存相机矩阵。其他预设未指定 `--dest` 时仍写入旧阶段脚本使用的项目根目录，wget 备用入口也使用该旧布局。选择较大的训练预设前，可先查看 `python scripts/download_hf_assets.py --help`。

## 离线准备 RGB 推理

默认网络先用 torchvision 权重初始化 ResNet-34，再加载各物体的 HccePose 权重。可先在联网机器上使用新的 Torch Hub 缓存，下载主干权重并查看缓存目录：

```bash
python -c "from torchvision.models import ResNet34_Weights; ResNet34_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)"
python -c "from pathlib import Path; import torch; print(Path(torch.hub.get_dir()) / 'checkpoints')"
```

在**目标机**执行第二条命令，将 `resnet34-b627a593.pth` 复制到输出的目录。torchvision 在下载时校验公开的 SHA-256 前缀 `b627a593`。复制后再比对两台机器上文件的完整 SHA-256。参阅 [torchvision ResNet-34 权重说明](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet34.html)。

## 图像和相机

彩色图使用 OpenCV 的 BGR 顺序。`camera.json` 为与原图分辨率对应的 3×3 数组，单位为像素。

深度图必须事先与彩色图完成像素配准。WAPR 接口接收米制深度；整数毫米图应显式乘以 `1e-3`。尺寸相同并不代表已经配准。

## 其他公开数据与训练资源

运行单图和 WAPR 示例只需上面的七文件清单。需要其他任务时按目录选择资源，不要无差别下载完整数据仓库。

| 任务 | 公开目录或文件 |
| --- | --- |
| 料箱模型、权重与训练数据 | [demo-bin-picking](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/demo-bin-picking) |
| 纹理物体模型与权重 | [demo-tex-objs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/demo-tex-objs) |
| 图片输入 | [test_imgs](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_imgs) |
| 视频输入 | [test_videos](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_videos) |
| 配准 RGB-D 帧 | [test_imgs_RGBD](https://huggingface.co/datasets/SEU-WYL/HccePose/tree/4e482e540314622981ccd4fce5beddac7f9d3e1d/test_imgs_RGBD) |
| 材质库 | [cc0textures-512.zip](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/4e482e540314622981ccd4fce5beddac7f9d3e1d/cc0textures-512.zip) |
| 七个 BOP 数据集权重 | [BOP 评测与权重索引](bop-evaluation.md) |

Hugging Face 的 `local_dir` 保留上游层级。例如下载 `demo-tex-objs/` 后，它本身才是对应数据集根目录；不要再额外包一层同名目录。训练 ZIP 与无后缀 checkpoint 是不同类型：ZIP 按训练教程解压，checkpoint 保留原文件名直接由加载器读取。

自己的数据应使用清晰的外部数据根目录，由该根构造模型、训练场景和标签路径。物体模型采用毫米，深度换算按对应采集元数据，而非根据文件扩展名猜测。制作新模型的完整流程见[物体准备](object-preparation.md)。
