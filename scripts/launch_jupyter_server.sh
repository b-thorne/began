#!/bin/bash
#SBATCH --job-name="jupyter-server"
#SBATCH -p gpu
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks-per-node=28
#SBATCH --mem=100G
#SBATCH -t 04:00:00

./envs-gpu/bin/jupyter-lab --port 8765 --no-browser >> logs/jupyter.log