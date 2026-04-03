#!/bin/bash
#SBATCH --partition=ENSTA-l40s
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --job-name=open_kettle_robocasa_project
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source .venv/bin/activate
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

mkdir -p logs

python rew_shaping_and_curriculum_train.py
