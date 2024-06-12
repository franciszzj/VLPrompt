#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --job-name=kings_sgg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=262144
#SBATCH --time=2-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version
conda activate hilo

PORT=$(shuf -i 10000-65535 -n 1)
BIN=$1
PART=$2

torchrun \
  --nproc_per_node 1 \
  --master_port $PORT \
  tools/llama2_chat.py \
  data/psg/meta/llama2-7b/ \
  $BIN \
  $PART
