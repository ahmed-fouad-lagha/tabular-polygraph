# Cover Letter for SN Computer Science Submission

```
Dear Editors,

We submit "Row-Level Structural Integrity Diagnostics for Synthetic Tabular Data" for consideration in SN Computer Science.

This paper introduces the Hybrid Integrity Framework (HIF), the first row-level diagnostic that detects conditional dependency violations in synthetic tabular data by combining conditional classifiers, spectral continuity checks, and mined implication rules into a non-compensatory multiplicative score. Across four generators (Gaussian Copula, Vine Copula, CTGAN, TVAE) and five datasets (Census ACS, Adult Income, Credit Default, Online Purchases, Supermarket Sales), we demonstrate a persistent Integrity-Fidelity Decoupling: generators achieving KS > 0.96 simultaneously fail 85–100% of dependency integrity tests on constrained domains — most sharply for a Vine Copula arm whose cross-feature dependence is deliberately ablated as a positive control. We further characterize a label-conditional curation effect: pruning low-HIF rows improves downstream utility only when the prediction target is among the auditor's dependency hubs (ΔF1 up to +39.4 points, p < 0.001), with much weaker or absent effects otherwise. Two controlled experiments establish this as a checkable necessary condition.

Three 2025 citations are arXiv preprints:
- Long et al. (2025) — arXiv:2502.04055 (ICLR 2025 workshop)
- Yu et al. (2025) — arXiv:2511.17590 (IEEE BigData 2025)
- Jiang et al. (2025) — arXiv:2509.11950 (ICLR 2026 Oral, accepted)

Appendix (formal results, algorithms, sample complexity, scalability) can be moved to Supplementary Material if preferred.

All authors approve this submission. The work is original and not under review elsewhere.

Sincerely,
Ahmed Fouad Lagha (corresponding author)
Eötvös Loránd University, Budapest
```
