# Contrastive pretraining on fingerprints improves Tox21 assay ranking for unseen chemical scaffolds

From a v3 forecast: contrastive pretraining improves Tox21 prediction on all 12 assays for unseen chemical families.

**Paper:** [paper.pdf](paper.pdf) · **Page:** [singularitysd.com/research/tox21.html](https://singularitysd.com/research/tox21.html) · Preprint, not yet peer reviewed.

## Abstract

We test whether contrastive pretraining on molecular fingerprints helps predict Tox21 in-vitro assay labels for compounds whose chemical scaffolds were absent from training. Two scaffold-grouped splits were fixed in advance, each with 615 held-out scaffolds. A fingerprint network pretrained to match two randomly masked views of the same molecule, then fine-tuned on the 12 assays, reached a mean macro average precision of 0.407 in the first split, against 0.384 for the same network trained from scratch and 0.321 for per-assay logistic regression. Averaging three initialisations, the contrastive model gained 0.031 over the supervised one (paired scaffold bootstrap 95% interval 0.008 to 0.052). The second split reproduced the gain: 0.028 (interval 0.012 to 0.040). In both splits the contrastive ensemble was better on all 12 assays. The cost is about 38 seconds of training instead of 4, and no data beyond the training compounds is needed.

## How it was made

This study started from the Singularity forecast **T3v3-2026-0102**: a method predicted to enter a new field. It was then tested once, with the plan fixed before looking at the results.

## What is in this repository

| file | what it is |
|---|---|
| `code/` | full code, protocol and intermediate results |
| `experiment.py` | reproduces the study end to end |
| `fig_results.png` | figure |
| `main.tex` | LaTeX source of the paper |
| `paper.pdf` | the paper |
| `results.json` | every number reported in the paper |

## Reproduce

```bash
pip install -r code/requirements.txt
python experiment.py
```

It rewrites the results and figures.

## Cite

```bibtex
@misc{singularity_tox21_contrastive_scaffolds_2026,
  title  = {Contrastive pretraining on fingerprints improves Tox21 assay ranking for unseen chemical scaffolds},
  author = {Singularity},
  year   = {2026},
  url    = {https://github.com/singularity-sd/tox21-contrastive-scaffolds}
}
```

## License

Code: MIT (`LICENSE`). Paper, text and figures: CC BY 4.0 (`LICENSE-paper.md`). Third-party data keep their own licence where stated.

---

Part of [Singularity](https://singularitysd.com): forecasting where science goes next, then testing it.
