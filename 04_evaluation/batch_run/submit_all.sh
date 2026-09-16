#!/bin/bash

# Submit the independent CWE/CAPEC jobs, then make fusion wait for both.

set -euo pipefail

BATCH_DIR=/home/gurleen/paper/methodology/04_evaluation/batch_run

CWE_SUBMISSION="$(sbatch --parsable "$BATCH_DIR/01_cve_to_cwe.sbatch")"
CAPEC_SUBMISSION="$(sbatch --parsable "$BATCH_DIR/02_cve_to_capec.sbatch")"
CWE_JOB="${CWE_SUBMISSION%%;*}"
CAPEC_JOB="${CAPEC_SUBMISSION%%;*}"
FUSION_SUBMISSION="$(
    sbatch --parsable \
        --dependency="afterok:${CWE_JOB}:${CAPEC_JOB}" \
        "$BATCH_DIR/03_fusion_models.sbatch"
)"
FUSION_JOB="${FUSION_SUBMISSION%%;*}"

echo "Submitted CVE-to-CWE job:   $CWE_JOB"
echo "Submitted CVE-to-CAPEC job: $CAPEC_JOB"
echo "Submitted dependent fusion job: $FUSION_JOB"
echo "Fusion dependency: afterok:${CWE_JOB}:${CAPEC_JOB}"
echo "Outputs: $BATCH_DIR/outputs"
echo "Logs: $BATCH_DIR/logs"
