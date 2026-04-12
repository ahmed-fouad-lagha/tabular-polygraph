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
pytest tests -q

# Cross-sectional experiment (fidelity + downstream + privacy)
python examples/01_cross_sectional.py

# Macro scenarios experiment (baseline + stressed scenarios + temporal fidelity)
python examples/02_macro_scenarios.py

# Privacy audit walkthrough (MI, singling-out, linkability, DP demo)
python examples/03_privacy_audit.py
```

Expected output files are written under `examples/`:
- `output_census_train.csv`
- `output_census_eval.csv`
- `output_macro_baseline.csv`
- `output_macro_*.csv`

## CLI Usage

```bash
# Download a specific dataset
python main.py download fred_macro
python main.py download bls
python main.py download world_bank
python main.py download census_acs

# Download all 4 real datasets
python main.py download all

# Check what's cached
python main.py download status

# Force re-download (ignore cache)
python main.py download fred_macro --force

# Limit cache size (e.g., first 1000 rows only)
python main.py download bls --sample 1000

# Generate synthetic data: python main.py generate <id>
python main.py generate fred_macro --rows 10000 --output syn.csv

# Evaluate fidelity
python main.py evaluate real.csv syn.csv --type cross_sectional

# Evaluate fidelity with logical rule controls
python main.py evaluate real.csv syn.csv \
  --type cross_sectional \
  --rule-min-confidence 0.7 \
  --rule-min-support 0.01 \
  --rule-min-lift 1.0 \
  --rule-max-antecedents 2 \
  --rule-max-rules 200

# Run privacy audit
python main.py audit real.csv syn.csv --attacks 300
```

## Python API

```python
from src.generators import GaussianCopulaGenerator
from src.catalog import load_dataset
from src.fidelity import fidelity_report

real = load_dataset("census_acs")

gen = GaussianCopulaGenerator()
gen.fit(real)
syn = gen.generate(10000, seed=42)

# Basic evaluation
report = fidelity_report(real, syn, include_logical=True)
print(report["summary"])

# With tuned symbolic rule mining
report = fidelity_report(
    real,
    syn,
    include_logical=True,
    rule_min_confidence=0.7,
    rule_min_support=0.01,
    rule_min_lift=1.0,
    rule_max_antecedents=2,
    rule_max_rules=200,
)
print(report.get("logical", {}))
```

## License

MIT — see [LICENSE](LICENSE).
