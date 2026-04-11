# LCV: Semantic Fidelity Evaluation for Synthetic Tabular Data

<div align="center">

**Unsupervised semantic evaluation of logical consistency in synthetic tabular data.**

[![CI](https://github.com/ahmed-fouad-lagha/LCV/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmed-fouad-lagha/LCV/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

</div>

## Overview

LCV (Logical Constraint Validator) is a research framework for evaluating synthetic tabular data quality beyond distributional similarity.

Standard evaluation pipelines measure Euclidean and distributional agreement, which can miss row-level semantic inconsistencies — incompatible categorical combinations, physically implausible numeric relations, or violated domain constraints. LCV addresses this gap through neurosymbolic learning and validation of latent tabular constraints.

**What LCV adds:**
- Neural semantic scoring of row-level plausibility
- Symbolic rule mining and violation diagnostics
- Integrated reporting alongside marginal, joint, temporal, stylized-facts, downstream, and privacy metrics

**Core workflow:**
1. Learn structural regularities from real data using an unsupervised model
2. Score each synthetic row by semantic deviation severity
3. Aggregate violations into dataset-level quality diagnostics

## Setup

```bash
git clone https://github.com/ahmed-fouad-lagha/LCV.git
cd LCV
pip install -r requirements.txt

# Optional: verify installation
python main.py list
pytest tests
```

## CLI Usage

```bash
# Generate synthetic data from a built-in profile
python main.py generate hmda --rows 10000 --output syn.csv

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

real = load_dataset("hmda")

gen = GaussianCopulaGenerator()
gen.fit(real)
syn = gen.sample(10000, seed=42)

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