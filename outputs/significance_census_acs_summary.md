# Statistical Significance: census_acs

Generator: gaussian_copula | N seeds: 3 | Rows: 2000

## Utility Filtering Results

| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |
|---|---|---|---|
| Full synthetic | 100.0% | 0.916 ± 0.008 | nan ± nan |
| Rule-only | 99.5% | 0.922 ± 0.009 | nan ± nan |
| **HIF Oracle** | **85.9%** | **0.924 ± 0.009** | **nan ± nan** |

## Statistical Tests

- **Paired t-test** (Full vs HIF): t=-6.705, p=0.0215 *
- **Wilcoxon signed-rank**: W=0.0, p=0.2500
- **95% CI** for F1 difference (HIF - Full): [0.0026, 0.0119]
- **Mean F1 improvement**: 0.0073
