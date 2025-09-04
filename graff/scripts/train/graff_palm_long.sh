#!/bin/bash

# 长时间训练脚本 - 虎口朝上抓取优化
# 增加训练步数和优化超参数

cd graff

python train.py \
    --exp ./expts/graff_palm_long_seed1 \
    --env-name 'graff-v0' \
    --use-gae \
    --log-interval 50 \
    --save-interval 500 \
    --num-steps 2048 \
    --num-processes 8 \
    --lr 3e-4 \
    --entropy-coef 0.01 \
    --value-loss-coef 0.5 \
    --ppo-epoch 10 \
    --num-mini-batch 32 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-env-steps 50000000 \
    --use-proper-time-limits \
    --obj pan \
    --rewards grasp:1 aff:1 palm_orientation:1.0 \
    --obj_mass 1 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --seed 1 \
    --gpu-model 0 \
    --gpu-env 0
