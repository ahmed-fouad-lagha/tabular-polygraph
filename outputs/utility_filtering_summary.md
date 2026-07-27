# Downstream Utility Filtering Summary

| Dataset          | Generator   | Variant                      | Retention   | F1 (mean ± SEM)   | Acc (mean ± SEM)   |
|:-----------------|:------------|:-----------------------------|:------------|:------------------|:-------------------|
| census_acs       | ctgan       | Full synthetic               | 100.0%      | 0.1237 ± 0.0206   | 0.1813 ± 0.0203    |
| census_acs       | ctgan       | HIF Clean (Threshold <= 0.5) | 9.8%        | 0.2350 ± 0.0110   | 0.2773 ± 0.0124    |
| census_acs       | ctgan       | HIF Top-80% (Oracle)         | 80.0%       | 0.1314 ± 0.0203   | 0.1873 ± 0.0199    |
| census_acs       | ctgan       | Rule-Filtered                | 83.0%       | 0.1223 ± 0.0160   | 0.1813 ± 0.0197    |
| census_acs       | ctgan       | TRTR Real Baseline           | 100.0%      | 0.7046 ± 0.0047   | 0.7013 ± 0.0053    |
| online_purchases | gaussian    | Full synthetic               | 100.0%      | 0.8552 ± 0.0132   | 0.8800 ± 0.0104    |
| online_purchases | gaussian    | HIF Clean (Threshold <= 0.5) | 34.5%       | 0.7548 ± 0.0259   | 0.8070 ± 0.0173    |
| online_purchases | gaussian    | HIF Top-80% (Oracle)         | 80.0%       | 0.8404 ± 0.0353   | 0.8700 ± 0.0224    |
| online_purchases | gaussian    | Rule-Filtered                | 91.6%       | 0.8212 ± 0.0347   | 0.8580 ± 0.0209    |
| online_purchases | gaussian    | TRTR Real Baseline           | 100.0%      | 0.9891 ± 0.0008   | 0.9890 ± 0.0010    |
