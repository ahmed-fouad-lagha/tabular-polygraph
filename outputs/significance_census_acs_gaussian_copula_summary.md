# Statistical Significance: census_acs

Generator: gaussian_copula | N seeds: 10 | Rows: 2000

## Utility Filtering Results

| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |
|---|---|---|---|
| Full synthetic | 100.0% | 0.942 ± 0.003 | 0.942 ± 0.003 |
| Rule-only | 98.9% | 0.941 ± 0.003 | 0.941 ± 0.003 |
| **HIF Oracle** | **98.7%** | **0.942 ± 0.003** | **0.942 ± 0.003** |

## Statistical Tests

- **Paired t-test** (Full vs HIF): t=-0.130, p=0.8998
- **Wilcoxon signed-rank**: W=18.0, p=0.6523
- **95% CI** for F1 difference (HIF - Full): [-0.0028, 0.0032]
- **Mean F1 improvement**: 0.0002
