#!/bin/bash
#SBATCH --partition=ENSTA-l40s
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --job-name=kettle_robocasa_v2
#SBATCH --output=logs/v2_%j.out
#SBATCH --error=logs/v2_%j.err

source .venv/bin/activate
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0

python methods_combined_train.py
