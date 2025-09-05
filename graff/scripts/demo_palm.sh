#!/bin/bash

obj=pan
gpu=0

echo "=== Enhanced Palm Orientation Demo ==="
echo "Object: ${obj}"
echo "GPU: ${gpu}"
echo "Using enhanced palm orientation reward system"

python evaluate.py \
    --exp expts/graff_enhanced_palm_continue_seed1/ \
    --env-name graff-v0 \
    --obj ${obj} \
    --rewards grasp:1 aff:1 palm_orientation:5.0 \
    --obj_mass 0.8 \
    --obj_rot \
    --policy cnn-mlp \
    --cnn_arch custom \
    --camera egocentric \
    --inputs proprio loc rgb depth aff \
    --model best \
    --viz_stability \
    --stability_frc 3 \
    --save_videos \
    --num_eval_episodes 100 \
    --mode test \
    --gpu ${gpu}

echo ""
echo "=== Demo completed ==="
echo "Check the generated videos to see if the robot adapts its grasping posture"
echo "The enhanced reward system should encourage:"
echo "1. Proper palm orientation (no inversion)"
echo "2. Adaptive approach based on object shape"
echo "3. Better alignment with object handles/features"
