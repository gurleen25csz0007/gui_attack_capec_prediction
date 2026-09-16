#!/bin/bash

# Submit the two Our Approach cases as separate, ordered Slurm jobs.

set -euo pipefail

BATCH_DIR=/home/gurleen/paper/methodology/04_evaluation/batch_run

DESCRIPTION_SUBMISSION="$(
    sbatch --parsable "$BATCH_DIR/05_our_approach_description.sbatch"
)"
DESCRIPTION_JOB="${DESCRIPTION_SUBMISSION%%;*}"
KEYWORD_SUBMISSION="$(
    sbatch --parsable --dependency="afterok:${DESCRIPTION_JOB}" \
        "$BATCH_DIR/06_our_approach_keywords.sbatch"
)"
KEYWORD_JOB="${KEYWORD_SUBMISSION%%;*}"

echo "Submitted description Our Approach: $DESCRIPTION_JOB"
echo "Submitted dependent keyword Our Approach: $KEYWORD_JOB"
echo "Summary refresh runs inside job: $KEYWORD_JOB"
