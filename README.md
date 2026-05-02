# The Tabular Polygraph: Neurosymbolic Hallucination Detection in Synthetic Data

<div align="center">

<img src="assets/logo.png" alt="Tabular Polygraph" width="20%"/>

[![CI](https://github.com/xxxxx/tabular-polygraph/actions/workflows/ci.yml/badge.svg)](https://github.com/xxxxx/tabular-polygraph/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A Neuro-Symbolic Hybrid Integrity Framework for logically-consistent synthetic tabular data.**

---

</div>

The Hybrid Integrity Framework (HIF) provides a technically rigorous foundation for evaluating synthetic tabular data quality beyond simple distributional similarity. Standard evaluation pipelines measure Euclidean and marginal agreement, which can miss row-level semantic inconsistencies — incompatible categorical combinations, physically implausible numeric relations, or violated domain constraints. HIF addresses this gap through neurosymbolic learning and acts as a Tabular Polygraph to detect "Semantic Hallucinations".

<p align="center">
	<img src="assets/hif_architecture.png" alt="Hybrid Integrity Framework Architecture" width="88%"/>
	<br>
	<em>Figure 1: The Hybrid Integrity Framework (HIF)</em>
</p>

The HIF Hybrid Integrity Framework provides:
- **Algebraic Integrity Certificates** via the Multiplicative Integrity (MI) model.
- **Neural-Symbolic Oracles (LSE)** for categorical manifold discovery and auditing.
- **Neighbor-Invariant Continuity (NIC)** for verifying continuous manifold residuals.
- Reporting alongside marginal, joint, stylized-facts, downstream, and TAMIS privacy metrics.

#### Core workflow
1. Learn structural regularities from real data using an unsupervised model
2. Score each synthetic row by semantic deviation severity
3. Aggregate violations into dataset-level quality diagnostics

#### Performance
- **Spearman Monotonicity ($\rho = -1.0$)**: Perfect monotonic response under targeted semantic corruption in both current Census and Adult validation runs.
- **Utility Correlation is dataset-dependent**: Strong on Census ACS in current runs, weak on Adult under the same protocol.
- **Hallucination Detection**: Identifies row-level logical consistency gaps that can be attenuated by aggregate KS/TVD-style summaries.

## Setup

```bash
git clone https://github.com/xxxx/tabular-polygraph.git
cd tabular-polygraph
pip install -e .

# Verify environment
tabular-polygraph list
```

## Quick Start

Evaluate synthetic data against real ground truth. The evaluate command generates a 4-Pillar Scorecard covering Fidelity, Logic (Integrity), Utility, and Privacy.

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
from tabular_polygraph.catalog import load_dataset
from tabular_polygraph.fidelity import fidelity_report
from tabular_polygraph.fidelity.logical import hif_score

# 1. Load Data
real = load_dataset("census_acs")
syn = pd.read_csv("synthetic.csv")

# 2. Generate 4-Pillar Scorecard (Stats, Logic, Utility, Privacy)
report = fidelity_report(real, syn, dataset_type="cross_sectional")
print(f"Hybrid Integrity Score: {report['summary']['hybrid_integrity']}%")

# 3. Detect Row-Level Hallucinations
# hif_score returns per-row penalty scores [0 = valid, 1 = hallucination]
hif_results = hif_score(real, syn)
syn['hallucination_score'] = hif_results['row_penalties']

# Identify the top Hallucinations
hallucinations = syn.sort_values('hallucination_score', ascending=False).head(5)
print(hallucinations)
```

## Reproduce Benchmarks

To reproduce the empirical results and tables presented in the manuscript, run the following validation suite:

### 1. HIF Empirical Validation (Census ACS & Adult)
These scripts verify the framework's sensitivity to semantic corruption across multiple seeds and noise levels.

```bash
# Census ACS Validation
python scripts/04_hif_validation.py \
  --dataset census_acs \
  --corruption-strategy manifold_rupture \
  --rows 5000 \
  --seeds 42,43,44,45,46 \
  --corruption-levels 0,0.1,0.2,0.4,0.6 \
  --target employment_status \
  --output-dir results/census

# Adult Validation
python scripts/04_hif_validation.py \
  --dataset adult \
  --corruption-strategy manifold_rupture \
  --rows 2000 \
  --seeds 42,43,44,45,46 \
  --corruption-levels 0,0.1,0.2,0.4,0.6 \
  --output-dir results/adult
```

Summaries are generated in `results/<dataset>/hif_validation_summary.md`, confirming:
- **Monotonicity**: Strong rank-monotonic sensitivity to corruption levels (often $\rho = -1.0$ in our current benchmarks).
- **Practical separability**: HIF tracks targeted manifold ruptures under `manifold_rupture` even when aggregate fidelity metrics are less specific.
- **External validity**: Correlation with rule-violation behavior.
- **Utility coupling**: Can be strong or weak depending on dataset and target protocol.

### 2. Cross-Architecture Audit (Table 2 & 3)
Evaluates HIF across diverse architectures (Gaussian Copula, Vine Copula, CTGAN) to reproduce the primary comparative benchmarks.

```bash
python scripts/05_cross_domain_audit.py \
  --rows 1000 \
  --seeds 5 \
  --epochs 500 \
  --output-dir results
```

### 3. Utility Improvement through HIF Filtering (Table 1)
These commands reproduce the results showing how selecting records that satisfy the neuro-symbolic manifold laws recovers predictive performance lost during generation.

```bash
# Census ACS Utility Audit (Table 1, Top Row)
python scripts/07_utility_filtering.py \
  --dataset census_acs \
  --seeds 5 \
  --generator vine \
  --rows 10000

# Adult Utility Audit (Table 1, Bottom Row)
python scripts/07_utility_filtering.py \
  --dataset adult \
  --seeds 5 \
  --generator vine \
  --target income \
  --rows 5000
```

> [!NOTE]
> The **HIF Oracle (Combined)** variant selects the top 50% of synthetic records by integrity score. On the Adult dataset, we use 5,000 rows as a memory-safe default for systems with limited RAM.
> Canonical manuscript-facing aggregates are written to `results/paper_numbers.json`.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
