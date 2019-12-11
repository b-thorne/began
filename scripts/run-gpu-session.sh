#!/bin/bash
./envs-gpu/bin/jupyter-lab ./notebooks --no-browser --port=6868 --ip=0.0.0 &
./envs-gpu/bin/tensorboard --logdir=$TF_LOGDIR --host=0.0.0.0 --port=6869 &
