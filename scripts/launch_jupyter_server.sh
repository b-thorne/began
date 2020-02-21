#!/bin/bash
#SBATCH --job-name="jupyter-server"
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=25G
#SBATCH -t 04:00:00

./envs-gpu/bin/jupyter-lab --port 8765 --no-browser >> logs/jupyter.log