# Held-Out Error Type Detection

Tests HIF on error types it was NOT designed for.


## Corruption Level = 0.1

| Error Type | HIF F1 (ROC-AUC / PR-AUC) | IF F1 (ROC / PR) | LOF F1 (ROC / PR) |
|---|---|---|---|
| random_injection | 0.390 (0.896 / 0.543) | 0.118 (0.533 / 0.110) | 0.330 (0.857 / 0.526) |
| semantic_hallucination | 0.263 (0.825 / 0.421) | 0.114 (0.506 / 0.105) | 0.268 (0.714 / 0.291) |
| row_duplication | 0.026 (0.505 / 0.104) | 0.098 (0.505 / 0.105) | 0.143 (0.483 / 0.098) |
| feature_dropout | 0.068 (0.630 / 0.193) | 0.010 (0.273 / 0.064) | 0.162 (0.494 / 0.100) |
| covariate_shift | 0.000 (0.154 / 0.084) | 0.184 (0.585 / 0.156) | 0.049 (0.245 / 0.062) |

## Corruption Level = 0.2

| Error Type | HIF F1 (ROC-AUC / PR-AUC) | IF F1 (ROC / PR) | LOF F1 (ROC / PR) |
|---|---|---|---|
| random_injection | 0.415 (0.887 / 0.703) | 0.181 (0.493 / 0.196) | 0.440 (0.853 / 0.667) |
| semantic_hallucination | 0.281 (0.836 / 0.598) | 0.188 (0.514 / 0.204) | 0.408 (0.732 / 0.443) |
| row_duplication | 0.032 (0.510 / 0.205) | 0.205 (0.508 / 0.204) | 0.298 (0.498 / 0.204) |
| feature_dropout | 0.068 (0.618 / 0.320) | 0.035 (0.280 / 0.132) | 0.290 (0.483 / 0.199) |
| covariate_shift | 0.000 (0.152 / 0.171) | 0.304 (0.594 / 0.282) | 0.125 (0.253 / 0.129) |

## Corruption Level = 0.4

| Error Type | HIF F1 (ROC-AUC / PR-AUC) | IF F1 (ROC / PR) | LOF F1 (ROC / PR) |
|---|---|---|---|
| random_injection | 0.418 (0.883 / 0.834) | 0.416 (0.508 / 0.403) | 0.613 (0.861 / 0.823) |
| semantic_hallucination | 0.246 (0.816 / 0.756) | 0.402 (0.497 / 0.397) | 0.591 (0.710 / 0.638) |
| row_duplication | 0.026 (0.492 / 0.396) | 0.407 (0.496 / 0.398) | 0.532 (0.490 / 0.391) |
| feature_dropout | 0.070 (0.625 / 0.545) | 0.157 (0.283 / 0.284) | 0.525 (0.509 / 0.415) |
| covariate_shift | 0.000 (0.155 / 0.345) | 0.401 (0.496 / 0.410) | 0.295 (0.217 / 0.268) |
