# Ablation Study: census_acs

Generator: ctgan | Seeds: 5 | Rows: 2000

| Ablation | Retention% | F1 (mean ± SEM) | Violation Rate |
|---|---|---|---|
| No filtering | 100.0% | 0.360 ± 0.018 | 0.0% |
| LSE-only | 27.2% | 0.812 ± 0.011 | 72.8% |
| NIC-only | 85.6% | 0.382 ± 0.018 | 10.7% |
| Rules-only | 97.2% | 0.368 ± 0.018 | 2.8% |
| LSE + NIC | 16.2% | 0.834 ± 0.007 | 83.6% |
| Full HIF (arith.) | 95.3% | 0.357 ± 0.015 | 4.7% |
| Full HIF (geom.) | 16.0% | 0.838 ± 0.007 | 83.8% |
