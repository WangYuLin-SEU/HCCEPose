#!/bin/bash
GPU_ID=$1
SCENE_NUM=$2
cc0textures=$3
dataset_path=$4
s2_p1_gen_pbr_data=$5
for (( SCENE_ID=0; SCENE_ID<$SCENE_NUM; SCENE_ID++ ))
do
    SCENE_ID_PADDED=$(printf "%06d" $SCENE_ID)
    echo "Running scene $SCENE_ID_PADDED on GPU $GPU_ID"
    export EGL_DEVICE_ID=$GPU_ID
    cd $dataset_path
    python $s2_p1_gen_pbr_data $GPU_ID $cc0textures
done
