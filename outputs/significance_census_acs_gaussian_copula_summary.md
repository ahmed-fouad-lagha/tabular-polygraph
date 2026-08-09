# Statistical Significance: census_acs

Generator: gaussian_copula | N seeds: 10 | Rows: 2000

## Utility Filtering Results

| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |
|---|---|---|---|
| Full synthetic | 100.0% | 0.940 ± 0.002 | 0.940 ± 0.002 |
| Rule-only | 99.1% | 0.937 ± 0.002 | 0.938 ± 0.002 |
| **HIF Oracle** | **99.2%** | **0.937 ± 0.002** | **0.937 ± 0.002** |

## Statistical Tests

- **Paired t-test** (Full vs HIF): t=1.652, p=0.1330
- **Wilcoxon signed-rank**: W=13.0, p=0.1602
- **95% CI** for F1 difference (HIF - Full): [-0.0079, 0.0012]
- **Mean F1 improvement**: -0.0033
