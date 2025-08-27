#!/bin/bash

# 快速测试脚本 - 虎口朝上抓取优化
# 用于快速验证功能，较少的训练步数

cd graff

python train.py \
    --exp ./expts/graff_palm_quick_seed1 \
    --env-name 'graff-v0' \
    --use-gae \
    --log-interval 10 \
    --save-interval 50 \
    --num-steps 512 \
    --num-processes 4 \
    --lr 5e-4 \
    --entropy-coef 0.01 \
    --value-loss-coef 0.5 \
    --ppo-epoch 4 \
    --num-mini-batch 8 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-env-steps 1000000 \
    --use-proper-time-limits \
    --obj pan \
    --rewards grasp:1 aff:1 palm_orientation:2.0 \
    --obj_mass 1 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --seed 1 \
    --gpu-model 0 \
    --gpu-env 0
