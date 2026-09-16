# Two-A100 batch run

This folder is a self-contained Slurm run location. The batch job uses the
`torchgpu` Conda environment, requests two A100 GPUs for up to 48 hours, and
keeps its generated results and logs inside this folder.

Submit the complete pipeline with:

```bash
bash /home/gurleen/paper/methodology/03_keyword_making/batch_run/submit_all.sh
```

`submit_all.sh` submits `run_all.sbatch` and prints the Slurm job ID. The
`.sbatch` file contains the actual resource request and pipeline logic.

Execution order:

1. Occlusion, phrase similarity, modified KeyBERT, and base Qwen are queued
   as one-GPU tasks. Slurm runs at most two simultaneously. Qwen reuses the
   existing completed generation only when its ordered CVE IDs and
   descriptions exactly match the current dataset; otherwise it runs the 7B
   model and keeps a resumable checkpoint.
2. TextRank runs as the fifth, CPU-only keyword method with 16 CPU cores.
3. The job verifies that every method produced `training_final.csv`,
   `validation_final.csv`, and `testing_final.csv`.
4. The keyword-only MiniLM cosine Top-5 comparison runs last with one GPU and
   16 CPU cores. It uses MiniLM only as a fixed encoder and verifies that all
   five methods occur in the final comparison CSV.

Hyperparameter selection uses validation data only:

- Standard Occlusion searches 80 threshold combinations in one shared pass.
- Phrase similarity searches 64 threshold combinations in one shared pass:
  every validation CVE and unique candidate/trimmed phrase is embedded once,
  then reused across the full grid.
- Modified KeyBERT evaluates all 4 target-similarity thresholds together from
  shared document, candidate, target-part, and merged-phrase embeddings.
- Each winning configuration is rerun once for the final test result, and its
  three final CSVs are published directly under `outputs/<method>/`.
- Occlusion, phrase similarity, and modified KeyBERT all select their winning
  configuration by validation macro-F1 and report the same macro/micro
  CWE/CAPEC metrics.
- Modified KeyBERT compares phrases against four independent target parts—CWE
  name, CWE description, CAPEC name, and CAPEC description—without embedding
  the target ID into the comparison text.
- Modified KeyBERT ranks the threshold-passing candidates and retains the top
  five before merging. Only overlapping or adjacent phrases within that
  top-five set can be merged, and each merged phrase is re-scored against the
  same CWE/CAPEC target threshold.
- Qwen and TextRank are fixed baselines with no label-scored extractor
  threshold.

The final evaluation uses keywords only. It compares normalized MiniLM
embeddings with the CAPEC catalogue by cosine similarity and retrieves five
CAPECs per CVE. It evaluates the training, validation, and testing splits and
writes the complete multilabel and Hit@K metric set to CSV.

Large split and inspection files from non-winning tuning configurations are
removed automatically. Their small validation metric files and tuning summary
are retained for reproducibility.

Method results are written under `outputs/<method>/`. The final comparison is
`outputs/evaluation/minilm_cosine_top5/comparison_summary.csv`. Per-task
logs are under `logs/`.

The batch file uses the common typed-GRES spelling `gpu:a100:2`. If this
cluster names its A100 resource differently, change only that `#SBATCH` line
to the site's spelling (for example, `gpu:2` plus an A100 constraint).
