# LCV: Semantic Fidelity Evaluation for Synthetic Tabular Data

<div align="center">

[![CI](https://github.com/ahmed-fouad-lagha/LCV/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmed-fouad-lagha/LCV/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Unsupervised semantic evaluation of logical consistency in synthetic tabular data.**

---

</div>

## Overview

LCV (Logical Constraint Validator) for evaluating synthetic tabular data quality beyond distributional similarity.

Standard evaluation pipelines measure Euclidean and distributional agreement, which can miss row-level semantic inconsistencies — incompatible categorical combinations, physically implausible numeric relations, or violated domain constraints. LCV addresses this gap through neurosymbolic learning and validation of latent tabular constraints.

**What LCV adds:**
- Semantic scoring of row-level plausibility
- Symbolic rule mining and violation diagnostics
- Reporting alongside marginal, joint, temporal, stylized-facts, downstream, and privacy metrics

**Core workflow:**
1. Learn structural regularities from real data using an unsupervised model
2. Score each synthetic row by semantic deviation severity
3. Aggregate violations into dataset-level quality diagnostics

## Setup

```bash
git clone https://github.com/ahmed-fouad-lagha/LCV.git
cd LCV
pip install -r requirements.txt

# Environment check
python main.py list
```

## Quick Start (Absolute Parity)

Establish a ground truth evaluation during generation, then reproduce it perfectly in standalone mode.

```bash
# 1. Download and Generate
python main.py download census_acs
python main.py generate census_acs --rows 100 --seed 42 --output synthetic.csv

# 2. Standalone Evaluation
python main.py evaluate ~/.src/cache/census_acs.parquet synthetic.csv --type cross_sectional --seed 42
```

## CLI Usage

```bash
# Evaluate with High-Order Logic (Supports Multi-Antecedents)
python main.py evaluate real.csv syn.csv \
  --type cross_sectional \
  --rule-max-antecedents 2 \
  --rule-min-confidence 0.98

# Evaluate Time-Series sequence integrity
python main.py evaluate real_ts.parquet syn_ts.csv --type time_series --seed 42

# Run adversarial privacy audit
python main.py audit real.csv syn.csv --attacks 300
```

## Python API

```python
from src.catalog import load_dataset
from src.fidelity import fidelity_report

real = load_dataset("census_acs")
syn = pd.read_csv("synthetic.csv")

# Report (Fidelity + Logic + Utility + Privacy)
report = fidelity_report(
    real,
    syn,
    dataset_type="cross_sectional",
    random_state=42
)

# Access the 4-Pillar Scorecard
summary = report['summary']
print(f"Holistic Score: {summary['holistic_integrity']}%")
print(f"Pillars Score:  {summary['pillars']}")
# {'fidelity': 86.4, 'logic': 20.5, 'utility': 60.7, 'privacy': 100.0}
```

## License

MIT — see [LICENSE](LICENSE).
