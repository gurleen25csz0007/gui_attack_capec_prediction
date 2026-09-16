# Stage-04 three-job batch run

The evaluation is split into three Slurm files. Every file requests one GPU,
four CPU cores, 32 GB system RAM, and at most 48 hours. Models within a file
run sequentially, so each receives exclusive access to the available 40 GB
GPU memory.

1. `01_cve_to_cwe.sbatch` runs the four description/keyword frozen and
   fine-tuned CVE-to-CWE models.
2. `02_cve_to_capec.sbatch` runs the equivalent four direct CVE-to-CAPEC
   models.
3. `03_fusion_models.sbatch` runs RAE-XMC, both CGDPF cases, description-only
   Our Approach, keyword-aware Our Approach, and the final result collector.

Submit all three with correct dependencies:

```bash
bash /home/gurleen/paper/methodology/04_evaluation/batch_run/submit_all.sh
```

The CWE and CAPEC jobs may run concurrently if the cluster grants two GPUs.
The fusion job is submitted with `afterok` and cannot start until both finish
successfully. If only one GPU is available, Slurm queues the jobs normally.

Results are isolated under `batch_run/outputs/`. Group logs and separate
stdout/stderr files for every model are under `batch_run/logs/`. The complete
comparison is `batch_run/outputs/model_results_summary.csv`.

## Six-model grid rerun on two GPUs

Run the alpha/beta/lambda grid evaluations with:

```bash
bash /home/gurleen/paper/methodology/04_evaluation/batch_run/submit_grid_evaluations.sh
```

The submission uses two GPUs as three parallel pairs. Description and keyword
CGDPF run first; description and keyword RAE-XMC wait for both CGDPF jobs;
description and keyword Our Approach then wait for both RAE-XMC jobs. A final
CPU-only job refreshes `model_results_summary.csv` after both fusion jobs pass.
Every dependency uses Slurm `afterok`, so downstream models cannot consume a
failed or incomplete prerequisite run.

If the local watcher is interrupted while Slurm jobs continue, resume without
duplicating them using either `--resume-rae RAE_DESC_JOB RAE_KEYWORD_JOB` or
`--resume-our OUR_DESC_JOB OUR_KEYWORD_JOB`.

The final summary contains only the description and KeyBERT-keyword variants
of Our Approach. Per-run calibrated-RAE and original-hybrid rows are diagnostic
components, not extra Our Approach models, and are excluded by the collector.
