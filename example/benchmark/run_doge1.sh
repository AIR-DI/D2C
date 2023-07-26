#!/usr/bin/env bash

TRAIN_STEPS=1000000
BM_NAME='d4rl'
DATA_SOURCE='mujoco'
ALPHA_LIST1='hopper_medium-v2 hopper_medium_replay-v2 hopper_random-v2 halfcheetah_random-v2 walker2d_random-v2'
#ALPHA_LIST2='antmaze_umaze-v2 antmze_umaze_diverse-v2 antmaze_medium_diverse-v2 antmaze_medium_play-v2 antmaze_large_play-v2 antmaze_large_diverse-v2'
# shellcheck disable=SC1061
# shellcheck disable=SC1073
#for STD in 0.2 0.3; do
for ENV_U in 'HalfCheetah'; do
ENV=$ENV_U'-v2'
#for DATA_TYPE in 'medium-v2' 'medium_replay-v2' 'medium_expert-v2' 'random-v2'; do
#for DATA_TYPE in 'medium-v2' 'medium_replay-v2'; do
DATA=${ENV_U,,}'_'$DATA_TYPE
DIR='data/d4rl/mujoco/'
DATA1=$DIR${ENV_U,,}'_medium_expert-v2'
for RATIO1 in 0.03 0.01; do
DATA2=$DIR${ENV_U,,}'_random-v2'
RATIO2=1
# shellcheck disable=SC2199
# shellcheck disable=SC2076
if [[ "${ALPHA_LIST1[@]}" =~ "$DATA" ]]; then
  alpha=17.5
else
  alpha=7.5
fi

if [[ "$DATA" =~ "antmaze" ]]; then
  lr_distance=0.0001
  initial_lambda=1
  train_d_steps=1000000
else
  lr_distance=0.001
  initial_lambda=5
  train_d_steps=100000
fi

for SEED in 20; do
GPU_DEVICE=0
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python demo_data_mix.py \
  --train.agent_ckpt_name='221110-data_mix/me'$RATIO1'_r'$RATIO2 \
  --model.model_name='doge' \
  --model.doge.hyper_params.alpha=$alpha \
  --model.doge.hyper_params.initial_lambda=$initial_lambda\
  --model.doge.hyper_params.train_d_steps=$train_d_steps\
  --model.doge.hyper_params.optimizers.distance[1]=$lr_distance\
  --train.batch_size=256 \
  --env.external.benchmark_name=$BM_NAME \
  --env.external.env_name=$ENV \
  --env.external.data_source=$DATA_SOURCE \
  --env.external.state_normalize=True \
  --env.external.score_normalize=True \
  --train.total_train_steps=$TRAIN_STEPS \
  --train.seed=$SEED \
  --train.wandb.entity='d2c' \
  --train.wandb.project='data_mix' \
  --train.wandb.name='doge-'$ENV_U'-me'$RATIO1'_r'$RATIO2 \
  --data1=$DATA1 \
  --ratio1=$RATIO1 \
  --data2=$DATA2 \
  --ratio2=$RATIO2 &
sleep 2

done
done
done