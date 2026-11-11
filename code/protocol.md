# Protocol: T3v3-2026-0102

## Question
Test whether contrastive learning helps predict in-vitro toxicology assay labels for molecular scaffolds absent from training.

## Data and exclusions
Use the public Tox21 CSV with 7,831 records, 12 assay tasks, SMILES, and molecule IDs. RDKit Morgan fingerprints use radius 2 and 1,024 bits. Eight SMILES records that RDKit could not parse or sanitize are listed in excluded_invalid.csv and removed before splitting, leaving 7,823 compounds. Missing labels remain missing and are masked; they are not changed to negatives.

## Splits
Compute canonical SMILES and Bemis-Murcko scaffolds. Acyclic compounds receive unique scaffold keys based on canonical SMILES. Run two GroupShuffleSplit partitions by scaffold:
- Split 1: test fraction 0.15, seed 2026; validation fraction 0.12 of the remaining groups, seed 2054.
- Split 2: same fractions, seeds 2028 and 2056.

The partitions contain 5,955/776/1,092 and 6,179/707/937 compounds (train/validation/test). We verify that no scaffold group crosses partitions in either split. This is a strict scaffold transfer test, not the random split used by every MoleculeNet benchmark report.

## Models
- Contrastive MLP: two independent 10% fingerprint-bit-dropout views; InfoNCE loss, temperature 0.20, 80 pretraining epochs; pretrain only on training compounds. Transfer the encoder into a 12-task classifier.
- Supervised MLP: same architecture and supervised procedure, trained from random initialization.
- Logistic fingerprint baseline: one class-balanced logistic regression per task, using only observed training labels.
- Neural models use masked binary cross-entropy, task-specific positive weights computed from training labels, AdamW, learning rate 0.0005 during supervised fine-tuning, up to 70 epochs, validation macro average precision early stopping, patience 10, and seeds 17, 29, 43.

## Outcomes and limits
Primary metric: macro average precision over assay tasks with both classes present. Also report macro ROC AUC, per-task results, run variation, ensemble probability averages, training time, and paired scaffold bootstrap. This measures ranking of labels in one in-vitro dataset; it does not measure human, clinical, or real-world toxicity.
