#!/bin/bash

#SBATCH --job-name=keyword_cosine_top5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/03_keyword_making/evaluation/logs/cosine_top5_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/03_keyword_making/evaluation/logs/cosine_top5_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/03_keyword_making/evaluation

mkdir -p "$EVALUATION_DIR/logs"

export PYTHONUNBUFFERED=1
export KEYWORD_EVAL_CPU_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export RAYON_NUM_THREADS=16
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Evaluation started: $(date)"
echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Evaluator: fixed MiniLM encoder + cosine Top-5"

/home/gurleen/miniconda3/envs/torchgpu/bin/python \
    "$EVALUATION_DIR/evaluate_all_methods.py"

echo "Evaluation finished: $(date)"
