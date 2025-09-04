#!/bin/bash

# 改进的演示脚本 - 使用改进后的模型
obj=pan
gpu=0

echo "obj: ${obj}"
echo "GPU: ${gpu}"

python evaluate.py \
    --exp expts/graff_palm_improved_seed1/ \
    --env-name graff-v0 \
    --obj ${obj} \
    --rewards grasp:10 aff:2 palm_orientation:5.0 \
    --obj_mass 0.5 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --model best \
    --viz_stability \
    --stability_frc 3 \
    --save_videos \
    --num_eval_episodes 20 \
    --mode test \
    --gpu ${gpu}
