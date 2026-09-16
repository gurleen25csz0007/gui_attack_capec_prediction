# Keyword-only CAPEC evaluation

`evaluate_all_methods.py` compares these five keyword methods:

- Occlusion
- Phrase Similarity
- KeyBERT Modified
- TextRank
- Qwen Keywords

The evaluator uses `sentence-transformers/all-MiniLM-L6-v2` as a fixed text
encoder for each keyphrase and CAPEC Name + Description, then ranks CAPECs by
cosine similarity. It does not train a classifier or fine-tune MiniLM. For a
CVE with multiple keyphrases, each CAPEC receives the maximum score from any
of those keyphrases. Only the keyword column is used as the CVE query.

The five highest-ranked CAPEC IDs form the predicted multilabel set. The CSV
metrics include micro, macro, weighted, and sample precision/recall/F1, sample
Jaccard, Hamming loss, exact match, Hit@1, and Hit@5.

Run the existing method outputs as a standalone Slurm job:

```bash
cd /home/gurleen/paper/methodology/03_keyword_making
mkdir -p batch_run/logs
sbatch batch_run/cosine_top5.sbatch
```

Alternatively, submit the equivalent evaluation batch file:

```bash
mkdir -p /home/gurleen/paper/methodology/03_keyword_making/evaluation/logs
sbatch /home/gurleen/paper/methodology/03_keyword_making/evaluation/run_evaluation.sbatch
```

The main result is written to:

`outputs/evaluation/minilm_cosine_top5/comparison_summary.csv`

Each method also gets `metrics.csv` and prediction CSVs for the training,
validation, and testing splits.
