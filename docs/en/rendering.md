# Render PBR training data

This page uses preparation, training or evaluation scripts from the [public upstream source](https://github.com/WangYuLin-SEU/HCCEPose/tree/f75cc4d5db2fc298fc7b853f8c75273757872382). Obtain that revision as described in [Installation](installation.md#public-training-and-rendering-source), and run these commands from its root. The current project’s `examples/inference/` provides application inference.

Once you have [prepared the meshes and symmetry metadata](object-preparation.md) and [installed BlenderProc](installation.md), you can render training scenes for your objects. The example uses `s2_p1_gen_pbr_data.py`, our adaptation of [BlenderProc2](https://github.com/DLR-RM/BlenderProc), to render scenes with different viewpoints, materials and object placements.

## Choose a texture library

The example uses [cc0textures-512](https://huggingface.co/datasets/SEU-WYL/HccePose/blob/4e482e540314622981ccd4fce5beddac7f9d3e1d/cc0textures-512.zip), a download of about 600 MB. Unzip it and pass the absolute path to the extracted directory to the renderer. If you need the original CC0Textures library, download it with `s2_p0_download_cc0textures.py` and allow about 44 GB of disk space.

```
HCCEPose
|--- s2_p0_download_cc0textures.py
|--- cc0textures
```

## Render a batch of scenes

Run `s2_p1_gen_pbr_data.sh` to start a fresh Python process for each scene. When one scene finishes, its process exits and releases its memory. The public shell wrapper changes into the dataset directory. Supply absolute paths for textures, the dataset and the Python script. Run the following from the repository root, replacing the texture and dataset paths with your own.

```bash
chmod +x s2_p1_gen_pbr_data.sh
nohup ./s2_p1_gen_pbr_data.sh 0 42 /absolute/path/cc0textures-512 /absolute/path/demo-bin-picking /absolute/path/HCCEPose-upstream/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &
```

This command uses GPU 0, the selected textures and the meshes in `demo-bin-picking/models/`. It writes 42 numbered scenes to `train_pbr/`, each with 1,000 rendered frames. Change the second argument to render a different number of scenes.

```
demo-bin-picking
|--- models
|--- train_pbr
      |--- 000000
      |--- 000001
      ...
      |--- 000041
```
