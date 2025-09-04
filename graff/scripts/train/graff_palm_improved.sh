#!/bin/bash

# 改进的虎口朝上抓取训练脚本
# 解决训练中的关键问题

cd graff

python train.py \
    --exp ./expts/graff_palm_improved_seed1 \
    --env-name 'graff-v0' \
    --use-gae \
    --log-interval 10 \
    --save-interval 100 \
    --num-steps 2048 \
    --num-processes 8 \
    --lr 3e-4 \
    --entropy-coef 0.02 \
    --value-loss-coef 0.5 \
    --ppo-epoch 10 \
    --num-mini-batch 32 \
    --gamma 0.995 \
    --gae-lambda 0.95 \
    --num-env-steps 5000000 \
    --use-proper-time-limits \
    --obj pan \
    --rewards grasp:10 aff:2 palm_orientation:5.0 \
    --obj_mass 0.5 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --seed 1 \
    --gpu-model 0 \
    --gpu-env 0
