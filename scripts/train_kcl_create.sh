#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --job-name=kings_sgg
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=262144
#SBATCH --time=2-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

conda activate hilo

CONFIG=$1
GPUS=4
PORT=$(shuf -i 10000-65535 -n 1)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
  --nproc_per_node=$GPUS \
  --master_port=$PORT \
  tools/train.py \
  $CONFIG \
  --auto-resume \
  --no-validate \
  --launcher pytorch