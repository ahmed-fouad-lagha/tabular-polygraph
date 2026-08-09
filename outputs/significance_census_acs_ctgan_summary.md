# Statistical Significance: census_acs

Generator: ctgan | N seeds: 10 | Rows: 2000

## Utility Filtering Results

| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |
|---|---|---|---|
| Full synthetic | 100.0% | 0.427 ± 0.029 | 0.518 ± 0.018 |
| Rule-only | 97.0% | 0.426 ± 0.029 | 0.521 ± 0.018 |
| **HIF Oracle** | **65.7%** | **0.532 ± 0.044** | **0.601 ± 0.029** |

## Statistical Tests

- **Paired t-test** (Full vs HIF): t=-4.234, p=0.0022
- **Wilcoxon signed-rank**: W=0.0, p=0.0020
- **95% CI** for F1 difference (HIF - Full): [0.0485, 0.1599]
- **Mean F1 improvement**: 0.1042
