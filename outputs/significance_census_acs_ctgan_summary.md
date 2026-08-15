# Statistical Significance: census_acs

Generator: ctgan | N seeds: 10 | Rows: 2000

## Utility Filtering Results

| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |
|---|---|---|---|
| Full synthetic | 100.0% | 0.381 ± 0.020 | 0.507 ± 0.011 |
| Rule-only | 92.9% | 0.367 ± 0.014 | 0.499 ± 0.008 |
| **HIF Oracle** | **63.0%** | **0.441 ± 0.030** | **0.547 ± 0.015** |

## Statistical Tests

- **Paired t-test** (Full vs HIF): t=-3.423, p=0.0076
- **Wilcoxon signed-rank**: W=0.0, p=0.0020
- **95% CI** for F1 difference (HIF - Full): [0.0203, 0.0993]
- **Mean F1 improvement**: 0.0598
