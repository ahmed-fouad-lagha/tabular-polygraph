# High Fidelity, Broken Dependencies: Row-Level Integrity Auditing for Synthetic Tabular Data

<div align="center">

[![CI](https://github.com/ahmed-fouad-lagha/tabular-polygraph/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmed-fouad-lagha/tabular-polygraph/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](requirements.lock)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A Hybrid Integrity Framework (HIF) for row-level auditing of synthetic tabular data.**

---

</div>

The Hybrid Integrity Framework (HIF) is a row-level diagnostic for synthetic tabular data. Aggregate metrics can show close marginal alignment even when individual synthetic rows are inconsistent with conditional structure learned from real reference data. HIF combines conditional sentinel classifiers, numeric residual checks, and mined implication rules to assign every synthetic row a conditional-consistency score. It is a diagnostic relative to the available reference data, not a universal validity certificate.

<p align="center">
	<img src="assets/hif_architecture.svg" alt="Hybrid Integrity Framework Architecture" width="88%"/>
	<br>
	<em>Figure 1: The Hybrid Integrity Framework (HIF)</em>
</p>

HIF provides:

- **Logical Sentinel Ensemble (LSE):** conditional classifiers over binned feature values, including quantized numeric hubs.
- **Neighbor-Invariant Continuity (NIC):** residual checks for numeric non-hub features conditioned on non-numeric context.
- **Implication-rule checks:** automatically mined, high-confidence rules used as a hard component when present.
- **Multiplicative aggregation:** combines active component validities while limiting compensation between them.

#### Core workflow
1. Fit binnings, sentinels, NIC regressors, and implication rules on real reference data.
2. Score each synthetic row for conditional inconsistency.
3. Aggregate row scores into a cohort-level diagnostic and inspect row-level traces.

#### Evidence and scope

The accompanying manuscript reports cross-generator audits on five public benchmarks, leakage-free utility analyses, and representation-drift checks. HIF can identify dependency violations that aggregate metrics do not localize, but its filtering utility is target- and dataset-dependent; density-based baselines can be stronger for near-deterministic continuous arithmetic constraints.

## Setup

```bash
git clone https://github.com/ahmed-fouad-lagha/tabular-polygraph.git
cd tabular-polygraph
pip install -e .

# Verify environment
tabular-polygraph list
```

## Quick Start

Evaluate synthetic data against real reference data. The evaluation reports fidelity metrics and the HIF integrity diagnostic; downstream utility is included when a target is supplied, and privacy auditing is opt-in.

> **Reproducibility:** the exact datasets behind every number in the manuscript are
> bundled in `data/cache/*.parquet`. `load_dataset` uses these snapshots by default, so
> a fresh clone reproduces all paper results offline, independent of the live Census
> API / UCI sources. Delete `data/cache/` to instead use freshly downloaded data.

```bash
# 1. Generate synthetic data from the bundled Census ACS snapshot
tabular-polygraph generate census_acs --rows 100 --output synthetic.csv

# 2. Audit the synthetic rows against the real reference data
tabular-polygraph evaluate data/cache/census_acs.parquet synthetic.csv --type cross_sectional --hif-epochs 10
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

# 2. Generate a fidelity report, including the HIF diagnostic
report = fidelity_report(real, syn, dataset_type="cross_sectional")
print(f"Hybrid Integrity Score: {report['summary']['logic_score']}%")

# 3. Inspect row-level conditional-inconsistency penalties
# hif_score returns penalties in [0, 1]; higher values indicate greater
# inconsistency with the learned reference conditionals.
hif_results = hif_score(real, syn)
syn["hif_penalty"] = hif_results["row_penalties"]

# Inspect the five rows with the highest penalties
flagged_rows = syn.sort_values("hif_penalty", ascending=False).head(5)
print(flagged_rows)
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
