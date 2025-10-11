#!/bin/bash
GPU_ID=$1
SCENE_START=$2
SCENE_NUM=$3
for (( SCENE_ID=$SCENE_START; SCENE_ID<$(($SCENE_START + $SCENE_NUM)); SCENE_ID++ ))
do
    SCENE_ID_PADDED=$(printf "%06d" $SCENE_ID)
    echo "Running scene $SCENE_ID_PADDED on GPU $GPU_ID"
    export EGL_DEVICE_ID=$GPU_ID
    cd /root/xxxxxx/demo-tex-objs
    python /root/xxxxxx/s2_p1_gen_pbr_data.py $GPU_ID
done
