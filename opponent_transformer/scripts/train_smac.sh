#!/bin/sh
env="StarCraft2"
map="6h_vs_8z"
algo="ppo"
exp="single"
seed=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_smac.py
