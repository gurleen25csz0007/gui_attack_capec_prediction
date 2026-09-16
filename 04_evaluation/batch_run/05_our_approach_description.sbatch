#!/bin/bash

# Rerun only the description-based Our Approach using completed score caches.

#SBATCH --job-name=our_approach_description
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/our_approach_description_rerun_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/our_approach_description_rerun_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
BATCH_DIR="$EVALUATION_DIR/batch_run"
OUTPUT_ROOT="$BATCH_DIR/outputs"
CGDPF_POINTER="$OUTPUT_ROOT/combined/latest_CGDPF_run.txt"
RAE_CACHE_POINTER="$OUTPUT_ROOT/combined/latest_our_approach_run.txt"

cd "$EVALUATION_DIR"
source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

for pointer in "$CGDPF_POINTER" "$RAE_CACHE_POINTER"; do
    if [[ ! -s "$pointer" ]]; then
        echo "Missing completed-run pointer: $pointer" >&2
        exit 1
    fi
done

export SCORE_CACHE_RUN="$(<"$CGDPF_POINTER")"
export RAE_SCORE_CACHE_RUN="$(<"$RAE_CACHE_POINTER")"

echo "[$(date '+%F %T')] Starting description Our Approach rerun"
python -u our_approach.py
echo "[$(date '+%F %T')] Finished description Our Approach rerun"

