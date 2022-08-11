#!/usr/bin/env bash

TRAIN_STEPS=1000000
BM_NAME='d4rl'
DATA_SOURCE='mujoco'
for ENV_U in 'Hopper' 'HalfCheetah' 'Walker2d'; do
ENV=$ENV_U'-v2'
for DATA_TYPE in 'random-v2' 'medium-v2' 'medium_replay-v2' 'medium_expert-v2'; do
DATA=${ENV_U,,}'_'$DATA_TYPE
for SEED in 20 30 40; do

GPU_DEVICE=0
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python demo.py \
  --train.agent_ckpt_name='220811' \
  --model.model_name='td3_bc' \
  --model.td3_bc.hyper_params.alpha=2.5 \
  --train.batch_size=256 \
  --env.external.benchmark_name=$BM_NAME \
  --env.external.env_name=$ENV \
  --env.external.data_name=$DATA \
  --env.external.data_source=$DATA_SOURCE \
  --env.external.state_normalize=True \
  --env.external.score_normalize=True \
  --train.total_train_steps=$TRAIN_STEPS \
  --train.seed=$SEED &
sleep 2

done
wait
done
done