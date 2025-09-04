#!/bin/bash

# 对比原始模型和虎口朝上模型的抓取效果
# 生成两个视频进行对比

obj=pan
gpu=0

echo "=== 对比原始模型和虎口朝上模型的抓取效果 ==="
echo "物体: ${obj}"
echo "GPU: ${gpu}"

# 生成原始模型视频
echo ""
echo "1. 正在生成原始模型（无虎口约束）的抓取视频..."
python evaluate.py \
    --exp expts/graff_trained/ \
    --env-name graff-v0 \
    --obj ${obj} \
    --rewards grasp:1 aff:1 \
    --obj_mass 1 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --model best \
    --viz_stability \
    --stability_frc 3 \
    --save_videos \
    --num_eval_episodes 10 \
    --mode test \
    --gpu ${gpu}

echo "原始模型视频生成完成！"
echo "保存位置: expts/graff_trained/videos/"

# 生成虎口朝上模型视频
echo ""
echo "2. 正在生成虎口朝上模型的抓取视频..."
python evaluate.py \
    --exp expts/graff_palm_quick_seed1/ \
    --env-name graff-v0 \
    --obj ${obj} \
    --rewards grasp:1 aff:1 palm_orientation:2.0 \
    --obj_mass 1 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --model best \
    --viz_stability \
    --stability_frc 3 \
    --save_videos \
    --num_eval_episodes 10 \
    --mode test \
    --gpu ${gpu}

echo "虎口朝上模型视频生成完成！"
echo "保存位置: expts/graff_palm_quick_seed1/videos/"

echo ""
echo "=== 对比完成 ==="
echo "你现在可以对比两个视频："
echo "1. 原始模型: expts/graff_trained/videos/"
echo "2. 虎口朝上模型: expts/graff_palm_quick_seed1/videos/"
echo ""
echo "观察要点："
echo "- 虎口朝上模型是否保持手掌向上的姿态"
echo "- 抓取成功率是否有变化"
echo "- 抓取动作是否更加稳定"
