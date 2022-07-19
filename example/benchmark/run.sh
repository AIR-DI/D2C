#!/usr/bin/env bash

TRAIN_STEPS=500000
BM_NAME='d4rl'
DATA_SOURCE='mujoco'
for ENV_U in 'Hopper' 'HalfCheetah' 'Walker2d'; do
ENV=$ENV_U'-v2'
DATA=${ENV_U,,}'_random-v2'
for SEED in 101 110 120 130 140; do

GPU_DEVICE=0
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --train.agent_ckpt_name='220514' \
  --model.model_name='bcq' \
  --train.batch_size=100 \
  --env.env_external.benchmark_name=$BM_NAME \
  --env.env_external.env_name=$ENV \
  --env.env_external.data_name=$DATA \
  --env.env_external.data_source=$DATA_SOURCE \
  --train.total_train_steps=$TRAIN_STEPS \
  --train.seed=$SEED &
sleep 2

done
wait
done