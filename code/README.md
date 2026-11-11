# Contrastive learning for Tox21 assays

Test of forecast **T3v3-2026-0102 - contrastive learning in Health, Toxicology and Mutagenesis** (position 111; 46 supporting records).

## The question

Does contrastive pretraining on molecular fingerprints help predict Tox21 assay labels for compounds whose chemical scaffolds were absent from training?

## What we did

We used 7,831 public Tox21 records with 12 in-vitro assay labels. RDKit could not parse or sanitize eight SMILES strings, so those rows were listed and excluded; 7,823 compounds remained. Missing assay labels were preserved and masked during training and scoring.

We grouped compounds by Bemis-Murcko scaffold and kept each scaffold in only one partition. This tests transfer to new chemical series and is stricter than a random split. We ran two different splits, comparing a fingerprint MLP after contrastive pretraining, the same MLP trained from scratch, and task-wise logistic regression. Contrastive pretraining saw training compounds only and used two views created by hiding 10% of fingerprint bits.

## What we found

The mean test macro average precision was higher for contrastive pretraining in both splits: 0.407 vs 0.384 for the supervised MLP in split 1, and 0.328 vs 0.303 in split 2. Logistic regression scored 0.321 and 0.270. The three-seed probability averages scored 0.420 vs 0.389 in split 1 and 0.336 vs 0.308 in split 2. Contrastive learning improved average precision on all 12 assays in both splits.

Paired scaffold bootstrap intervals for the contrastive ensemble advantage over the supervised MLP were +0.008 to +0.052 and +0.012 to +0.040. There were 615 held-out scaffolds in each split. Training took about 38-40 seconds per contrastive run, including pretraining, compared with about 4 seconds for the supervised MLP.

This is a consistent result on one in-vitro dataset, not evidence of human or clinical toxicity prediction.

## Files

- main.pdf, main.tex: paper and LaTeX source.
- experiment.py, experiment_split2.py: two scaffold splits and model runs.
- consolidate.py, audit_scaffold_split.py: combined results and explicit scaffold-overlap audit.
- protocol.md: split and model settings.
- results_full.json: combined audit, seed metrics, per-assay scores, and bootstrap.
- metrics_run_complete.csv, metrics_ensemble_complete.csv, metrics_per_task_comparison.csv, metrics_summary_full.csv, bootstrap_scaffold_complete.csv, audit_scaffold_complete.csv: combined outputs.
- test_predictions.csv, split2_test_predictions.csv: per-compound test predictions.
- support_per_task.csv and split2_support_per_task.csv: observed-label counts.
- membership_*.csv: scaffold membership by partition.
- excluded_invalid.csv: eight source rows excluded after RDKit parsing.
- fig_results.png: paper figure.
- data/: source Tox21 archive and attribution.

## Reproduce

Use Python 3.13 with PyTorch, RDKit, NumPy, pandas, scikit-learn, and Matplotlib.

py -3.13 experiment.py
py -3.13 experiment_split2.py
py -3.13 consolidate.py
py -3.13 audit_scaffold_split.py

The raw CSV is the Tox21 file served by DeepChem from the NIH Tox21 challenge.
