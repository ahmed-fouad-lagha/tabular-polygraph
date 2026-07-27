# Statistical Significance: online_purchases

Generator: ctgan | N seeds: 10 | Rows: 2000

## Utility Filtering Results

| Variant | Retention% | F1 (mean ± SEM) | Accuracy (mean ± SEM) |
|---|---|---|---|
| Full synthetic | 100.0% | 0.337 ± 0.007 | nan ± nan |
| Rule-only | 90.8% | 0.336 ± 0.006 | nan ± nan |
| **HIF Oracle** | **2.2%** | **0.737 ± 0.069** | **nan ± nan** |

## Statistical Tests

- **Paired t-test** (Full vs HIF): t=-5.722, p=0.0003 ***
- **Wilcoxon signed-rank**: W=0.0, p=0.0020
- **95% CI** for F1 difference (HIF - Full): [0.2416, 0.5577]
- **Mean F1 improvement**: 0.3997
