# Ablation Study: census_acs

Generator: gaussian_copula | Seeds: 5 | Rows: 2000

| Ablation | Retention% | F1 (mean ± SEM) | Violation Rate |
|---|---|---|---|
| No filtering | 100.0% | 0.937 ± 0.003 | 0.0% |
| LSE-only | 95.1% | 0.939 ± 0.003 | 4.9% |
| NIC-only | 97.0% | 0.941 ± 0.004 | 3.0% |
| Rules-only | 99.1% | 0.938 ± 0.003 | 0.9% |
| LSE + NIC | 97.7% | 0.940 ± 0.003 | 2.3% |
| Full HIF (arith.) | 99.8% | 0.934 ± 0.004 | 0.3% |
| Full HIF (geom.) | 98.3% | 0.937 ± 0.002 | 1.7% |
