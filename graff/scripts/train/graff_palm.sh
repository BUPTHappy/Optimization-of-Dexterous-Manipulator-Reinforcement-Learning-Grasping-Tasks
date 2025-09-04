#!/bin/bash

expdir=./expts      # 实验目录
gpu_model=0         # 模型使用的GPU
gpu_env=0           # 环境使用的GPU  
seed=1              # 随机种子

# 检查CUDA可用性
echo "检查CUDA可用性..."
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('CUDA设备数量:', torch.cuda.device_count())"

# 检查预训练模型是否存在
if [ ! -f "./expts/graff_trained/models/best.pt" ]; then
    echo "错误: 预训练模型 ./expts/graff_trained/models/best.pt 不存在!"
    exit 1
fi

echo "找到预训练模型: ./expts/graff_trained/models/best.pt"

# 继续训练配置 - 专门强化虎口朝上约束
expname=graff_palm_enhanced_seed${seed}
expdir=${expdir}

screen -dmS $expname bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu_model}
    cd graff
    mkdir -p $expdir/$expname/logs
    mkdir -p $expdir/$expname/models
    
    # 复制预训练模型，重命名为1.pt（这样load_model 1就能加载它）
    cp ./expts/graff_trained/models/best.pt $expdir/$expname/models/1.pt
    
    echo 'CUDA_VISIBLE_DEVICES设置为: '${gpu_model}
    echo '开始从预训练模型继续训练，强化虎口朝上约束...'
    echo '预训练模型已复制到: $expdir/$expname/models/1.pt'
    
    python train.py \
    --exp $expdir/$expname \
    --env-name 'graff-v0' \
    --use-gae \
    --log-interval 5 \
    --save-interval 25 \
    --num-steps 800 \
    --num-processes 12 \
    --lr 2e-5 \
    --entropy-coef 0.003 \
    --value-loss-coef 0.5 \
    --ppo-epoch 3 \
    --num-mini-batch 8 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-env-steps 2000000 \
    --use-proper-time-limits \
    --obj pan \
    --rewards grasp:5 aff:1 palm_orientation:8.0 \
    --obj_mass 0.8 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --seed ${seed} \
    --gpu-model ${gpu_model} \
    --gpu-env ${gpu_env} \
    --load_model 1 |& tee $expdir/$expname/logs/train_log.txt
    
    echo '训练完成!'
"

