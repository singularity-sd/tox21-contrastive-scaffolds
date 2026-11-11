"""Reproduces the study by running the original scripts in code/ in their historical order.

1. code/experiment.py          - split 1 (scaffold seeds 2026/2054): contrastive MLP, supervised MLP, logistic
2. code/experiment_split2.py   - split 2 (seeds 2028/2056), same models and settings
3. code/audit_scaffold_split.py - explicit check that no scaffold crosses partitions
4. code/consolidate.py            - combined tables, paired scaffold bootstrap, fig_results.png
Settings fixed before the runs: code/protocol.md. Data: code/data/tox21.csv.gz (public Tox21 CSV, as
distributed with DeepChem). Requires Python 3.12 with the packages in code/requirements.txt (RDKit, PyTorch CPU).
"""

import os
import runpy
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code"

for step in ["experiment.py", "experiment_split2.py", "audit_scaffold_split.py", "consolidate.py"]:
    script = CODE / step
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))
    print(f"== {step}", flush=True)
    runpy.run_path(str(script), run_name="__main__")
