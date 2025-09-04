#!/bin/bash

# 适中训练脚本 - 虎口朝上抓取优化
# 减少训练步数，更快看到结果

cd graff

python train.py \
    --exp ./expts/graff_palm_moderate_seed1 \
    --env-name 'graff-v0' \
    --use-gae \
    --log-interval 20 \
    --save-interval 100 \
    --num-steps 1024 \
    --num-processes 8 \
    --lr 3e-4 \
    --entropy-coef 0.01 \
    --value-loss-coef 0.5 \
    --ppo-epoch 8 \
    --num-mini-batch 16 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-env-steps 5000000 \
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
