#!/bin/sh
#SBATCH --job-name=TryTF
#SBATCH --partition=3day-long
#SBATCH --mem=32gb
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
export CUDA_VISIBLE_DEVICES=1
/apps/conda/ycai223/envs/RNNTF/bin/python trainflies.py


