#!/bin/bash

expdir=./expts      # path to folder with model checkpoints
gpu_model=0         # gpu to use for model
gpu_env=0           # gpu to use for env
seed=1              # seed to use

# Check if CUDA is available
echo "Checking CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

# graff with palm orientation constraint - minimal changes from original
expname=graff_palm_seed${seed}; expdir=${expdir}; 
screen -dmS $expname bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu_model}
    cd graff
    mkdir $expdir/$expname; mkdir $expdir/$expname/logs; 
    echo 'CUDA_VISIBLE_DEVICES set to: '${gpu_model}
    echo 'Starting training with GPU...'
    python train.py \
    --exp $expdir/$expname \
    --env-name 'graff-v0' \
    --use-gae \
    --log-interval 10 \
    --save-interval 100 \
    --num-steps 2000 \
    --num-processes 16 \
    --lr 5e-5 \
    --entropy-coef 0.001 \
    --value-loss-coef 0.5 \
    --ppo-epoch 4 \
    --num-mini-batch 20 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-env-steps 60000000 \
    --use-proper-time-limits \
    --obj pan \
    --rewards grasp:5 aff:1 palm_orientation:2.0 \
    --obj_mass 0.8 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --seed ${seed} \
    --gpu-model ${gpu_model} \
    --gpu-env ${gpu_env} |& tee $expdir/$expname/logs/train_log.txt"
