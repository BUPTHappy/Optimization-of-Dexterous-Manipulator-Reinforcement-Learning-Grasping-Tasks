
obj=pan
gpu=0

echo "obj: ${obj}"
echo "GPU: ${gpu}"


# 简化版本，减少内存使用
python evaluate.py \
    --exp expts/graff_palm_quick_seed1/ \
    --env-name graff-v0 \
    --obj pan \
    --rewards grasp:1 aff:1 palm_orientation:2.0 \
    --obj_mass 1 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --model best \
    --num_eval_episodes 5 \
    --mode test \
    --save_videos \
    --gpu 0


