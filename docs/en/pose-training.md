# Train HccePose(BF) for each object

This page uses preparation, training or evaluation scripts from the [public upstream source](https://github.com/WangYuLin-SEU/HCCEPose/tree/f75cc4d5db2fc298fc7b853f8c75273757872382). Obtain that revision as described in [Installation](installation.md#public-training-and-rendering-source), and run these commands from its root. The current project’s `examples/inference/` provides application inference.

HccePose(BF) learns a separate set of weights for each object. The script `s4_p2_train_bf_pbr.py` trains objects in sequence and can use one or more GPUs for each object.

## Configure the training script

Prepare the [meshes and metadata](object-preparation.md), [PBR scenes](rendering.md) and [front and back coordinate labels](coordinate-labels.md). Open the [training script](https://github.com/WangYuLin-SEU/HCCEPose/blob/f75cc4d5db2fc298fc7b853f8c75273757872382/s4_p2_train_bf_pbr.py) and edit the variables in `__main__`. Replace the placeholder `dataset_path` and set the object ID range.

| Parameter | Example setting | Purpose |
| --- | --- | --- |
| `start_obj_id`, `end_obj_id` | `1`, `5` | Train objects 1 through 5 in sequence. Set both to the same ID to train one object. |
| `total_iteration` | `50001` | Training iteration setting for each object. |
| `batch_size`, `lr` | `24`, `0.0002` | Batch size per GPU and learning rate. |
| `efficientnet_key` | `None` | Backbone setting used by the training script. |

For distributed data parallel (DDP) training, the total number of samples is:

```
total samples = total iteration × batch size × GPU number
```

## Run on one or more GPUs

The default `ide_debug=True` runs on a single GPU. Use it for debugging or a single-GPU training run:

```
nohup python -u s4_p2_train_bf_pbr.py > log4.file 2>&1 &
```

For DDP, set `ide_debug=False` and choose `--nproc_per_node` to match the number of GPUs. This example uses six. Start the job in a separate terminal session so that pausing your IDE does not interrupt communication between training processes.

```
screen -S train_ddp
nohup python -u -m torch.distributed.launch --nproc_per_node 6 s4_p2_train_bf_pbr.py > log4.file 2>&1 &
```

## Find the weights and monitor training

The script saves checkpoints by object ID under the dataset’s `HccePose/` directory. For example, `demo-tex-objs` uses this layout:

```
demo-tex-objs
|--- HccePose
    |--- obj_01
    ...
    |--- obj_10
|--- models
|--- train_pbr
|--- train_pbr_xyz_GT_back
|--- train_pbr_xyz_GT_front
```

During training, `test()` samples 1% of the same `train_pbr` split to monitor progress. For benchmark scores, run [BOP evaluation](bop-evaluation.md) on the corresponding test split.
