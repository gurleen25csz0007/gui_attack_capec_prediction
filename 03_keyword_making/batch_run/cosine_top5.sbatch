#!/bin/bash

#SBATCH --job-name=capec_cosine_top5
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/gurleen/paper/methodology/03_keyword_making/evaluation
#SBATCH --output=/home/gurleen/paper/methodology/03_keyword_making/batch_run/logs/cosine_top5_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/03_keyword_making/batch_run/logs/cosine_top5_%j.err

set -euo pipefail

export KEYWORD_EVAL_CPU_THREADS=16
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Evaluator: fixed MiniLM encoder + cosine Top-5"

/home/gurleen/miniconda3/envs/torchgpu/bin/python \
    /home/gurleen/paper/methodology/03_keyword_making/evaluation/evaluate_all_methods.py
