#!/bin/bash -l
#SBATCH --output=/jmain02/home/J2AD019/exk01/%u/logs/%j.out
#SBATCH --job-name=kings_sgg
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

CONFIG=$1
GPUS=8
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