# Statistical Significance: census_acs

Generator: ctgan | N seeds: 10 | Rows: 2000

## Utility Filtering Results

| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |
|---|---|---|---|
| Full synthetic | 100.0% | 0.397 ± 0.008 | nan ± nan |
| Rule-only | 96.9% | 0.407 ± 0.012 | nan ± nan |
| **HIF Oracle** | **59.0%** | **0.529 ± 0.022** | **nan ± nan** |

## Statistical Tests

- **Paired t-test** (Full vs HIF): t=-7.394, p=0.0000 ***
- **Wilcoxon signed-rank**: W=0.0, p=0.0020
- **95% CI** for F1 difference (HIF - Full): [0.0916, 0.1725]
- **Mean F1 improvement**: 0.1321
