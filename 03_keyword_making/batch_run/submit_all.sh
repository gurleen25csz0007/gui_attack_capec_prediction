#!/bin/bash

set -euo pipefail

PROJECT_DIR=/home/gurleen/paper/methodology/03_keyword_making
BATCH_FILE="$PROJECT_DIR/batch_run/run_all.sbatch"

PIPELINE_JOB=$(sbatch --parsable "$BATCH_FILE")

echo "Submitted complete two-GPU pipeline: $PIPELINE_JOB"
echo "Batch file: $BATCH_FILE"
echo "The keyword-only MiniLM cosine Top-5 comparison starts after all five methods finish."
echo "Outputs: $PROJECT_DIR/batch_run/outputs"
echo "Logs: $PROJECT_DIR/batch_run/logs"
