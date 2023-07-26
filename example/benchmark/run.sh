#!/usr/bin/env bash

TRAIN_STEPS=1000000
for SEED in 20 30; do
BM_NAME='d4rl'
DATA_SOURCE='mujoco'
for ENV_U in 'Hopper' 'HalfCheetah' 'Walker2d'; do
ENV=$ENV_U'-v2'
for DATA_TYPE in 'random-v2' 'medium-v2' 'medium_replay-v2' 'medium_expert-v2'; do
# shellcheck disable=SC2041
#for DATA_TYPE in 'medium-v2' 'medium_replay-v2'; do
DATA=${ENV_U,,}'_'$DATA_TYPE
#DIR='data/d4rl/mujoco/'
#DATA1=$DIR${ENV_U,,}'_medium_expert-v2'
#for RATIO1 in 0.03 0.01; do
#DATA2=$DIR${ENV_U,,}'_random-v2'
#RATIO2=1

GPU_DEVICE=0
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python demo.py \
  --train.agent_ckpt_name='221228' \
  --model.model_name='bc' \
  --train.batch_size=256 \
  --env.external.benchmark_name=$BM_NAME \
  --env.external.env_name=$ENV \
  --env.external.data_name=$DATA \
  --env.external.data_source=$DATA_SOURCE \
  --env.external.state_normalize=True \
  --env.external.score_normalize=True \
  --train.total_train_steps=$TRAIN_STEPS \
  --train.seed=$SEED \
  --train.wandb.entity='d2c' \
  --train.wandb.project='test_bc' \
  --train.wandb.name='bc-'$DATA'-seed'$SEED &

sleep 2

done
wait
done
done