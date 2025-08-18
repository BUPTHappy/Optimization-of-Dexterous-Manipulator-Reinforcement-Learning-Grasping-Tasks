expdir=./expts      # path to folder with model checkpoints
  gpu_model=0         # gpu to use for model
  gpu_env=0           # gpu to use for env (改为0，因为之前遇到设备ID问题)
  seed=1              # seed to use

  # graff with palm orientation constraint
  expname=graff_palm_seed${seed}; expdir=${expdir};
  screen -dmS $expname bash -c "
      mkdir -p $expdir/$expname; mkdir -p $expdir/$expname/logs; 
      python train.py \
      --exp $expdir/$expname \
      --env-name 'graff-v0' \
      --use-gae \
      --log-interval 10 \
      --save-interval 100 \
      --num-steps 2000 \
      --num-processes 8 \
      --lr 5e-5 \
      --entropy-coef 0.001 \
      --value-loss-coef 0.5 \
      --ppo-epoch 4 \
      --num-mini-batch 20 \
      --gamma 0.99 \
      --gae-lambda 0.95 \
      --num-env-steps 60000000 \
      --use-proper-time-limits \
      --obj all \
      --rewards grasp:1 aff:1 palm_orientation:0.3 \
      --obj_mass 1 \
      --obj_rot \
      --policy cnn-mlp \
      --cnn_arch custom \
      --camera egocentric \
      --inputs proprio loc rgb depth aff \
      --seed ${seed} \
      --gpu-model ${gpu_model} \
      --gpu-env ${gpu_env} |& tee $expdir/$expname/logs/train_log.txt"