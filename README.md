# LCV (Logical Constraint Validator)

<div align="center">

**Unsupervised Semantic Fidelity Evaluation for Synthetic Tabular Data via Neurosymbolic Extraction.**


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![JOSS](https://img.shields.io/badge/JOSS-under%20review-orange)](https://joss.theoj.org)

</div>

LCV is a research framework for evaluating synthetic tabular data quality beyond distributional similarity. It targets a practical gap: synthetic datasets can match classical statistical metrics while still containing logically inconsistent records that degrade downstream utility.

---

## Overview

LCV provides a semantic evaluation layer built around neurosymbolic extraction of latent tabular constraints.

Core idea:
1. Learn structural regularities from real data using an unsupervised model.
2. Score each synthetic row by semantic deviation severity.
3. Aggregate violations into dataset-level quality diagnostics.

This complements existing fidelity metrics (marginal, joint, temporal, stylized facts, downstream, privacy) with logic-aware validation.

## Research Motivation

Most synthetic data evaluation pipelines emphasize Euclidean and distributional agreement. In practice, this can miss row-level semantic inconsistencies (for example, incompatible categorical combinations or physically implausible numeric relations).

LCV is designed to detect and quantify those inconsistencies continuously, not just through hard-coded binary rules.

## Setup

```bash
python3 -m pip install -r requirements.txt

# Optional sanity checks
python3 main.py list
pytest tests/ -v --tb=short
```

## Quick Usage

```bash
# Generate synthetic data from a built-in profile
python3 main.py generate hmda --rows 10000 --output syn.csv

# Evaluate fidelity
python3 main.py evaluate real.csv syn.csv --type cross_sectional

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
```

## License

MIT License — free to use for any purpose including commercial. See [LICENSE](LICENSE).
