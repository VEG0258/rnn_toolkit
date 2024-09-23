#!/bin/sh
#SBATCH --job-name=TryTF
#SBATCH --partition=3day-long
#SBATCH --mem=32gb
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

/apps/conda/ycai223/envs/torch/bin/python trainflies.py


