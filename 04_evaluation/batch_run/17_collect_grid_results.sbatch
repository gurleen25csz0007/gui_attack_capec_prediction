#!/bin/bash

# Refresh the summary only after both final fusion jobs succeed.
#SBATCH --job-name=collect_grid_results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/collect_grid_results_%j.out
#SBATCH --error=/home/gurleen/paper/methodology/04_evaluation/batch_run/logs/collect_grid_results_%j.err

set -euo pipefail

EVALUATION_DIR=/home/gurleen/paper/methodology/04_evaluation
OUTPUT_ROOT="$EVALUATION_DIR/batch_run/outputs"

cd "$EVALUATION_DIR"
source /home/gurleen/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
export EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"

python -u collect_results.py

