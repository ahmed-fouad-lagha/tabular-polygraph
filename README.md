# LCV: Semantic Fidelity Evaluation for Synthetic Tabular Data

<div align="center">

[![CI](https://github.com/ahmed-fouad-lagha/LCV/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmed-fouad-lagha/LCV/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Unsupervised semantic evaluation of logical consistency in synthetic tabular data.**

</div>

---

## Overview

LCV (Logical Constraint Validator) is a research framework for evaluating synthetic tabular data quality beyond distributional similarity. Most synthetic data evaluation pipelines emphasize Euclidean and distributional agreement, which can miss row-level semantic inconsistencies such as incompatible categorical combinations or physically implausible numeric relations. LCV addresses this gap with a semantic evaluation layer built around neurosymbolic extraction of latent tabular constraints.

Core workflow:
1. Learn structural regularities from real data using an unsupervised model.
2. Score each synthetic row by semantic deviation severity.
3. Aggregate violations into dataset-level quality diagnostics.

This complements existing fidelity metrics (marginal, joint, temporal, stylized facts, downstream, privacy) with logic-aware validation.

## Setup

```bash
git clone https://github.com/ahmed-fouad-lagha/LCV.git
cd LCV
python3 -m pip install -r requirements.txt

# Optional sanity checks
python3 main.py list
pytest tests/test.py -v --tb=short
```

## Quick Usage

```bash
# Generate synthetic data from a built-in profile
python3 main.py generate hmda --rows 10000 --output syn.csv

# Evaluate fidelity
python3 main.py evaluate real.csv syn.csv --type cross_sectional

# Evaluate fidelity with logical rule controls
python3 main.py evaluate real.csv syn.csv \
  --type cross_sectional \
  --rule-min-confidence 0.7 \
  --rule-min-support 0.01 \
  --rule-min-lift 1.0 \
  --rule-max-antecedents 2 \
  --rule-max-rules 200

# Run privacy audit
python3 main.py audit real.csv syn.csv --attacks 300
```

Python API example:

```python
from src.generators import GaussianCopulaGenerator
from src.catalog import load_seed
from src.fidelity import fidelity_report

real = load_seed("hmda")
gen = GaussianCopulaGenerator()
gen.fit(real)
syn = gen.sample(10000, seed=42)

report = fidelity_report(real, syn, include_logical=True)
print(report["summary"])

# Optional: tune symbolic logical rule mining
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

MIT License. See [LICENSE](LICENSE).
