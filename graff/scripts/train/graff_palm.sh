#!/bin/bash

expdir=./expts      # path to folder with model checkpoints
gpu_model=0         # gpu to use for model
gpu_env=0           # gpu to use for env
seed=1              # seed to use

# Check if CUDA is available
echo "Checking CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

# Check if pretrained model exists
if [ ! -f "./expts/graff_trained/models/best.pt" ]; then
    echo "Error: Pretrained model ./expts/graff_trained/models/best.pt not found!"
    exit 1
fi

echo "Found pretrained model: ./expts/graff_trained/models/best.pt"

# Continue training from existing best model with enhanced palm orientation
expname=graff_enhanced_palm_continue_seed${seed}; expdir=${expdir}; 
screen -dmS $expname bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu_model}
    cd graff
    mkdir -p $expdir/$expname/logs
    mkdir -p $expdir/$expname/models
    
    # Copy pretrained model and rename to 1.pt so load_model 1 can load it
    cp ./expts/graff_trained/models/best.pt $expdir/$expname/models/1.pt
    
    echo 'CUDA_VISIBLE_DEVICES set to: '${gpu_model}
    echo 'Continuing training from graff_trained/models/best.pt with enhanced palm orientation...'
    echo 'Pretrained model copied to: $expdir/$expname/models/1.pt'
    
    python train.py \
    --exp $expdir/$expname \
    --env-name 'graff-v0' \
    --use-gae \
    --log-interval 10 \
    --save-interval 50 \
    --num-steps 2000 \
    --num-processes 12 \
    --lr 1e-5 \
    --entropy-coef 0.0005 \
    --value-loss-coef 0.5 \
    --ppo-epoch 4 \
    --num-mini-batch 20 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-env-steps 10000000 \
    --use-proper-time-limits \
    --obj pan \
    --rewards grasp:5 aff:1 palm_orientation:5.0 \
    --obj_mass 0.8 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --seed ${seed} \
    --gpu-model ${gpu_model} \
    --gpu-env ${gpu_env} \
    --load_model 1 |& tee $expdir/$expname/logs/train_log.txt"

echo "Enhanced palm orientation continue training started in screen session: $expname"
echo "To monitor progress: screen -r $expname"
echo "To view logs: tail -f $expdir/$expname/logs/train_log.txt"
echo ""
echo "Training parameters optimized for fine-tuning:"
echo "- Lower learning rate (1e-5) for stable fine-tuning"
echo "- Reduced entropy coefficient (0.0005) to maintain learned behaviors"
echo "- Shorter training (10M steps instead of 60M)"
echo "- Higher palm_orientation reward weight (5.0) to encourage adaptation"
echo "- More frequent saves (every 50 intervals) to monitor progress"
