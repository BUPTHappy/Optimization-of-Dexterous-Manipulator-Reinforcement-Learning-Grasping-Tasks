#!/bin/bash

# 继续训练脚本 - 从现有模型继续训练
# 增加虎口朝上奖励权重

cd graff

python train.py \
    --exp ./expts/graff_palm_continue \
    --env-name 'graff-v0' \
    --use-gae \
    --log-interval 10 \
    --save-interval 100 \
    --num-steps 1000 \
    --num-processes 4 \
    --lr 1e-4 \
    --entropy-coef 0.005 \
    --value-loss-coef 0.5 \
    --ppo-epoch 4 \
    --num-mini-batch 8 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-env-steps 100000 \
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
    --gpu-env 0 \
    --load_model 9
