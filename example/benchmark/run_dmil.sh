#!/usr/bin/env bash

TRAIN_STEPS=1000000
BM_NAME='d4rl'
DATA_SOURCE='mujoco'

for RATIO in 0.02 0.05 0.1 1.0; do
for ENV_U in 'Hopper' 'HalfCheetah' 'Walker2d'; do
ENV=$ENV_U'-v2'
# shellcheck disable=SC2041
for DATA_TYPE in 'expert-v2'; do
DATA=${ENV_U,,}'_'$DATA_TYPE

for SEED in 0; do

GPU_DEVICE=0
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python demo.py \
  --train.agent_ckpt_name='221203/expert_ratio_'$RATIO \
  --model.model_name='dmil' \
  --train.batch_size=256 \
  --env.external.benchmark_name=$BM_NAME \
  --env.external.env_name=$ENV \
  --env.external.data_name=$DATA \
  --env.external.data_source=$DATA_SOURCE \
  --env.external.state_normalize=True \
  --env.external.score_normalize=True \
  --train.total_train_steps=$TRAIN_STEPS \
  --train.data_split_ratio=$RATIO \
  --train.seed=$SEED \
  --train.wandb.entity='d2c' \
  --train.wandb.project='dmil_1203' \
  --train.wandb.name=$ENV_U'-expert_ratio_'$RATIO &
sleep 2

done
done
done
wait
done