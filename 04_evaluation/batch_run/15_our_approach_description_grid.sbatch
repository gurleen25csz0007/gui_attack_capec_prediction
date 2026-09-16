#!/bin/bash

# Description-only final fusion. Submitted after both CGDPF and RAE phases.
#SBATCH --job-name=our_desc_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/our_desc_grid_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/our_desc_grid_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
OUTPUT_ROOT="$EVALUATION_DIR/batch_run/outputs"
CGDPF_POINTER="$OUTPUT_ROOT/combined/latest_CGDPF_run.txt"
RAE_DIR="$OUTPUT_ROOT/RAE_XMC"

cd "$EVALUATION_DIR"
source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

for required in "$CGDPF_POINTER" "$RAE_DIR/best_model.pt" "$RAE_DIR/test_metrics.json"; do
    if [[ ! -s "$required" ]]; then
        echo "Missing completed prerequisite: $required" >&2
        exit 1
    fi
done

export SCORE_CACHE_RUN="$(<"$CGDPF_POINTER")"
unset RAE_SCORE_CACHE_RUN || true
python -u our_approach.py

