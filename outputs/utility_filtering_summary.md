# Downstream Utility Filtering Summary

| Dataset | Generator | Variant | Retention | F1 (mean ± SEM) | Acc (mean ± SEM) |
|---|---|---|---|---|---|
| census_acs | gaussian | Full synthetic | 100.0% | 0.6077 ± 0.0086 | 0.6123 ± 0.0072 |
| census_acs | gaussian | HIF Clean (Threshold <= 0.5) | 87.2% | 0.6000 ± 0.0098 | 0.6027 ± 0.0090 |
| census_acs | gaussian | HIF Top-80% (Oracle) | 80.0% | 0.6088 ± 0.0082 | 0.6093 ± 0.0076 |
| census_acs | gaussian | Rule-Filtered | 98.3% | 0.6118 ± 0.0099 | 0.6150 ± 0.0089 |
| census_acs | gaussian | TRTR Real Baseline | 100.0% | 0.7057 ± 0.0028 | 0.7020 ± 0.0030 |
| census_acs | ctgan | Full synthetic | 100.0% | 0.0899 ± 0.0159 | 0.1937 ± 0.0022 |
| census_acs | ctgan | HIF Clean (Threshold <= 0.5) | 9.5% | 0.2573 ± 0.0302 | 0.3240 ± 0.0208 |
| census_acs | ctgan | HIF Top-80% (Oracle) | 80.0% | 0.1009 ± 0.0212 | 0.2020 ± 0.0069 |
| census_acs | ctgan | Rule-Filtered | 94.3% | 0.0878 ± 0.0201 | 0.1910 ± 0.0047 |
| census_acs | ctgan | TRTR Real Baseline | 100.0% | 0.7057 ± 0.0028 | 0.7020 ± 0.0030 |
| census_acs | tvae | Full synthetic | 100.0% | 0.5411 ± 0.0091 | 0.5433 ± 0.0080 |
| census_acs | tvae | HIF Clean (Threshold <= 0.5) | 74.8% | 0.5603 ± 0.0075 | 0.5560 ± 0.0076 |
| census_acs | tvae | HIF Top-80% (Oracle) | 80.0% | 0.5605 ± 0.0078 | 0.5583 ± 0.0075 |
| census_acs | tvae | Rule-Filtered | 98.0% | 0.5564 ± 0.0085 | 0.5570 ± 0.0069 |
| census_acs | tvae | TRTR Real Baseline | 100.0% | 0.7057 ± 0.0028 | 0.7020 ± 0.0030 |
| online_purchases | gaussian | Full synthetic | 100.0% | 0.8552 ± 0.0132 | 0.8800 ± 0.0104 |
| online_purchases | gaussian | HIF Clean (Threshold <= 0.5) | 34.5% | 0.7548 ± 0.0259 | 0.8070 ± 0.0173 |
| online_purchases | gaussian | HIF Top-80% (Oracle) | 80.0% | 0.8404 ± 0.0353 | 0.8700 ± 0.0224 |
| online_purchases | gaussian | Rule-Filtered | 91.6% | 0.8212 ± 0.0347 | 0.8580 ± 0.0209 |
| online_purchases | gaussian | TRTR Real Baseline | 100.0% | 0.9891 ± 0.0008 | 0.9890 ± 0.0010 |
| online_purchases | ctgan | Full synthetic | 100.0% | 0.1923 ± 0.0371 | 0.2800 ± 0.0417 |
| online_purchases | ctgan | HIF Clean (Threshold <= 0.5) | 2.5% | 0.4500 ± 0.0766 | 0.5410 ± 0.0569 |
| online_purchases | ctgan | HIF Top-80% (Oracle) | 80.0% | 0.2089 ± 0.0370 | 0.3060 ± 0.0334 |
| online_purchases | ctgan | Rule-Filtered | 54.2% | 0.2445 ± 0.0326 | 0.3270 ± 0.0318 |
| online_purchases | ctgan | TRTR Real Baseline | 100.0% | 0.9891 ± 0.0008 | 0.9890 ± 0.0010 |
| online_purchases | tvae | Full synthetic | 100.0% | 0.4539 ± 0.0282 | 0.4680 ± 0.0312 |
| online_purchases | tvae | HIF Top-80% (Oracle) | 80.0% | 0.4667 ± 0.0359 | 0.4860 ± 0.0339 |
| online_purchases | tvae | Rule-Filtered | 83.7% | 0.4630 ± 0.0180 | 0.4760 ± 0.0213 |
| online_purchases | tvae | TRTR Real Baseline | 100.0% | 0.9891 ± 0.0008 | 0.9890 ± 0.0010 |
