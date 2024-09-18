#!/bin/sh
env="StarCraft2"
map="3s_vs_5z"
algo="oracle"
exp="single"
seed=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 10 --num_mini_batch 2 --episode_length 400 \
--num_env_steps 20_000_000 --ppo_epoch 5 --gain 1 --hidden_size 64 --use_value_active_masks --stacked_frames 4 --use_stacked_frames --use_eval --eval_episodes 32 --user_name "bhd445" \
--opponent_dir "../../opponent_transformer/envs/starcraft2/pretrained_opponents/3s_vs_5z"