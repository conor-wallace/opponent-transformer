#!/bin/sh
env="MPE"
scenario="simple_tag" 
num_landmarks=2
num_agents=4
num_good_agents=1
algo="oracle"
exp="check"
seed=0

echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
--n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "bhd445" \
--opponent_dir "../../opponent_transformer/envs/mpe/pretrained_opponents/tag"
