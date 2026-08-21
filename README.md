# Beyond Fidelity: When Statistical Quality Metrics Miss Structural Violations in Synthetic Tabular Data

[![CI](https://github.com/ahmed-fouad-lagha/tabular-polygraph/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmed-fouad-lagha/tabular-polygraph/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](requirements.lock)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A Neuro-Symbolic Hybrid Integrity Framework for logically-consistent synthetic tabular data.**

---

</div>

The Hybrid Integrity Framework (HIF) provides a technically rigorous foundation for evaluating synthetic tabular data quality beyond simple distributional similarity. Standard evaluation pipelines measure Euclidean and marginal agreement, which can miss row-level semantic inconsistencies, such as incompatible categorical combinations, physically implausible numeric relations, or violated domain constraints. HIF addresses this gap through neurosymbolic learning and acts as a Tabular Polygraph to detect "Semantic Hallucinations".

<p align="center">
	<img src="assets/hif_architecture.svg" alt="Hybrid Integrity Framework Architecture" width="88%"/>
	<br>
	<em>Figure 1: The Hybrid Integrity Framework (HIF)</em>
</p>

The HIF Hybrid Integrity Framework provides:
- **Algebraic Integrity Certificates** via the Multiplicative Integrity (MI) model.
- **Neural-Symbolic Oracles (LSE)** for categorical manifold discovery and auditing.
- **Neighbor-Invariant Continuity (NIC)** for verifying continuous manifold residuals.
- Reporting alongside marginal, joint, stylized-facts, and downstream utility metrics.

#### Core workflow
1. Learn structural regularities from real data using an unsupervised model
2. Score each synthetic row by semantic deviation severity
3. Aggregate violations into dataset-level quality diagnostics

#### Performance
- **Spearman Monotonicity ($\rho = -0.98$)**: Strong monotonic response under targeted semantic corruption in the Census ACS calibration protocol.
- **Utility Correlation is dataset-dependent**: Strong on Census ACS in current runs, weak on Adult under the same protocol.
- **Hallucination Detection**: Identifies row-level logical consistency gaps that can be attenuated by aggregate KS/TVD-style summaries.

## Setup

```bash
git clone https://github.com/ahmed-fouad-lagha/tabular-polygraph.git
cd tabular-polygraph
pip install -e .

# Verify environment
tabular-polygraph list
```

## Quick Start

Evaluate synthetic data against real ground truth. The evaluate command generates a 4-Pillar Scorecard covering Fidelity, Logic (Integrity), Utility, and Privacy.

> **Reproducibility:** the exact datasets behind every number in the manuscript are
> bundled in `data/cache/*.parquet`. `load_dataset` uses these snapshots by default, so
> a fresh clone reproduces all paper results offline, independent of the live Census
> API / UCI sources. Delete `data/cache/` to instead use freshly downloaded data.

```bash
# 1. Download sample data (cached in ~/.tabular_polygraph/cache/)
tabular-polygraph download census_acs

# 2. Generate synthetic data
tabular-polygraph generate census_acs --rows 100 --output synthetic.csv

# 3. Audit for Hallucinations (Semantic Integrity)
tabular-polygraph evaluate ~/.tabular_polygraph/cache/census_acs.parquet synthetic.csv --type cross_sectional --hif-epochs 10
```

## CLI Reference

### Generate Synthetic Data

```bash
tabular-polygraph generate <dataset_id_or_path> \
  --rows <number_of_rows> \
  --generator <type> \
  --seed <integer> \
  --output <filename.csv>
```

### Audit Synthetic Data (Fidelity Report)

```bash
tabular-polygraph evaluate <real_data_path> <synthetic_data_path> \
  --type <cross_sectional|time_series|panel> \
  --hif-epochs <integer> \
  --seed <integer> \
  --target <target_column_for_utility> \
  --output <report.json>
```

## Python API

```python
import pandas as pd
from tabular_polygraph import hif_score  # one-liner row-level audit
from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity import fidelity_report

# 1. Load Data
real = load_dataset("census_acs")
syn = pd.read_csv("synthetic.csv")

# 2. Generate 4-Pillar Scorecard (Stats, Logic, Utility, Privacy)
report = fidelity_report(real, syn, dataset_type="cross_sectional")
print(f"Hybrid Integrity Score: {report['summary']['logic_score']}%")

# 3. Detect Row-Level Hallucinations
# hif_score returns per-row penalty scores [0 = valid, 1 = hallucination]
hif_results = hif_score(real, syn)
syn["hallucination_score"] = hif_results["row_penalties"]

# Identify the top Hallucinations
hallucinations = syn.sort_values("hallucination_score", ascending=False).head(5)
print(hallucinations)
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
