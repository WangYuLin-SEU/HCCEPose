# 渲染 PBR 训练数据

本页使用本仓库根目录中的准备、训练或评测脚本。请按[安装说明](installation.md#公开训练与渲染源码)准备对应依赖，再从仓库根目录执行命令。`examples/inference/` 中的脚本用于应用推理。

准备好[物体模型和对称信息](object-preparation.md)，按[环境安装](installation.md)配置 BlenderProc 后，就可以为新物体渲染训练场景。示例使用我们基于 [BlenderProc2](https://github.com/DLR-RM/BlenderProc) 修改的 `s2_p1_gen_pbr_data.py`，从不同视角拍摄带有随机材质和物体摆放的场景。

## 选择材质库

示例用的是 [cc0textures-512](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/4e482e540314622981ccd4fce5beddac7f9d3e1d/cc0textures-512.zip)，压缩包约 600 MB。下载并解压后，把材质目录的绝对路径传给渲染脚本即可。如果需要原版 CC0Textures，用 `s2_p0_download_cc0textures.py` 下载，约需 44 GB 磁盘空间。

```
HCCEPose
|--- s2_p0_download_cc0textures.py
|--- cc0textures
```

## 批量渲染

用 `s2_p1_gen_pbr_data.sh` 批量运行。它为每个场景启动一个新的 Python 进程，渲染结束后该进程占用的内存随之释放。公开 shell 入口会切换到数据集目录；材质、数据集及 Python 脚本路径都应使用绝对路径。从仓库根目录运行以下命令，并按实际位置替换材质和数据集路径。

```bash
chmod +x s2_p1_gen_pbr_data.sh
nohup ./s2_p1_gen_pbr_data.sh 0 42 /absolute/path/cc0textures-512 /absolute/path/demo-bin-picking /absolute/path/HCCEPose-upstream/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &
```

这条命令使用 GPU 0，读取指定的材质库和 `demo-bin-picking/models/` 下的网格。示例生成 42 个场景，每个场景 1,000 帧。场景按编号写入 `train_pbr/`，可通过命令的第二个参数调整场景数。

```
demo-bin-picking
|--- models
|--- train_pbr
      |--- 000000
      |--- 000001
      ...
      |--- 000041
```
