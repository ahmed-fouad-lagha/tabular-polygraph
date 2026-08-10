# Peer Review: Row-Level Structural Integrity Diagnostics for Synthetic Tabular Data

**Manuscript:** `manuscript/main.tex` (Springer Nature `sn-jnl`, mathphys-num style)
**Reviewer notes:** grounded in a full read of the manuscript, appendix, and committed outputs. Independent reproducibility verification was performed (see §Verification), so claims about the numerical content below are checked against the shipped artifacts, not taken on trust.

---

## 1. Summary of the Manuscript

The paper addresses a genuine and under-served problem: evaluating synthetic tabular data at the **record level** for violations of joint conditional structure, which aggregate marginal metrics (KS, TVD, moment matching) are blind to by construction. The proposed framework, HIF, combines three layers:

1. **LSE** — Random Forest "sentinels" predict each high-dependency categorical "hub" feature from all other features; hub selection via cross-validated accuracy $S(h)$; per-record penalty via a conformal-style quantile-calibrated confidence floor ($\delta_h = Q_{0.05}$ of OOB probabilities).
2. **NIC** — Gradient-boosting regressors predict each continuous feature from an MCA/SVD spectral embedding of the categorical context; robust MAD/quantile thresholds yield a continuous penalty.
3. **Rules** — Apriori-style implication rules ($\min_{conf}=0.95$) as hard constraints.

Validities are combined via the **geometric mean** (non-compensatory "Logical AND"), with a fixed integrity frontier $H < 0.5$. Validation spans 5 datasets × 4 generators (Gaussian/Vine copulas, CTGAN, TVAE), with three independent regimes: (i) a cross-architecture maturity audit, (ii) downstream TSTR utility under filtering, and (iii) held-out error families vs. IF/LOF. Three theorems in the appendix provide intersection soundness, asymptotic convergence, and a Type I error bound.

**Main claims:** (1) HIF detects row-level dependency ruptures that marginal metrics report as near-perfect fidelity (the "Dependency Gap"); (2) pruning low-HIF records improves downstream utility when structural errors are present and preserves it otherwise; (3) HIF beats geometric outlier detectors on dependency violations at a matched operating point; (4) "Integrity–Fidelity Decoupling" — Vine copulas with $\mathrm{KS} \ge 0.969$ fail 86–100% of integrity tests on constrained domains.

---

## 2. Recommendation and Score

**Recommendation: Accept with major revision.**

| Dimension | Score (1–4) | Rationale |
|---|---|---|
| Soundness | 3 | Core methodology is sound, tests appropriate, honesty high; construct validity is partially circular and multiple-comparison handling is absent (see M1, M2). |
| Originality | 3 | Every component is an established technique; the contribution is the integrative problem formulation (row-level joint-conditional auditing with per-record attribution) plus a well-characterized decoupling finding. Clear value, no single paradigm-shifting component. |
| Clarity | 3 | Well organized and unusually honest; some over-long paragraphs, and "violation rate" is used with two meanings (see Min 1). |
| Significance | 3 | Real problem, practical tool, exemplary reproducibility; utility gains are real but modest and conditional (4/16 configurations). |
| **Overall** | **7/10** | Solid contribution with clear value for an applied data-science venue; the major comments below are addressable with targeted analyses and rewriting rather than fundamental redesign. |

Score-floor rules apply: methodology sound, statistical reporting rigorous (paired tests, exact $p$, CIs, effect sizes), limitations explicitly acknowledged, and reproducibility is exceptional — this cannot score below 6.

---

## 3. Strengths

- **The problem is real and the motivation is concrete.** The Supermarket Sales arithmetic constraint ($\text{Total} = \text{Price} \times \text{Quantity} \times (1+\text{Tax})$) is an externally verifiable invalidity, and the running example (a 14-year-old with a PhD) makes the dependency-gap failure mode tangible. Most synthetic-data evaluation literature still stops at marginal alignment; the manuscript correctly identifies this as a gap with downstream consequences.
- **The calibration evidence is clean and compelling.** The permutation protocol that preserves marginals while breaking joint structure is the right experimental design for demonstrating aggregate-blindness: MM and KS remain static ($\rho \approx 0$) while HIF decays monotonically ($\rho = -0.98$). This is the strongest part of the paper.
- **The Integrity–Fidelity Decoupling finding is genuinely informative.** Across 16 dataset–generator configurations, HIF correlates strongly with joint correlation fidelity ($\rho = +0.82$) and essentially not at all with marginal KS ($\rho = 0.038$). The finding that Vine copulas hit $\mathrm{KS} \ge 0.969$ while failing 86–100% of integrity tests on constrained domains is a useful cautionary result for the community.
- **Honest experimental design and honest reporting.** Disclosed `contamination` calibration for baselines, a matched HIF operating point (HIF†), a dedicated threats-to-validity paragraph, $N=3$ calibration experiments explicitly labeled descriptive, and — unusually — negative and null results reported throughout (Credit configs show no utility recovery; one significant degradation is reported, Credit Gaussian Copula $\Delta F1 = -0.021$, $p=0.013$; IF and LOF win on covariate shift and row duplication).
- **Reproducibility is outstanding** (see §Verification): bundled dataset snapshots, pinned dependency versions, and byte-identical re-runs.
- **The rule-only comparison, once reported, is a genuine anti-circularity argument** — see M3b; the committed outputs already contain the numbers.

---

## 4. Major Comments

### M1. Construct validity: what does $H < 0.5$ actually certify? (gating)

The central construct — "conditional dependency violation" — is defined by the auditor itself. A sentinel trained on real data flags low-probability conditional regions; a NIC regressor flags high residuals; a rule layer flags broken mined implications. For a **real generator** (not an injected-corruption experiment), the auditor cannot intrinsically distinguish:

- (a) a structural hallucination ("impossible junction"), i.e., a genuinely invalid configuration; from
- (b) a valid-but-rare configuration that the generator correctly produces but that sits below the training-set confidence floor (the paper itself acknowledges the coupling of rare tail patterns with high-penalty regions in the threats paragraph).

The controlled-corruption experiments guarantee violations by construction, so they validate *sensitivity and calibration*, not *specificity on real generator output*. The manuscript's strongest externally verifiable anchor — the Supermarket Sales arithmetic identity — is present but treated as one qualitative data point; it is the only place where "flagged = invalid" is checkable without reference to the auditor's own model.

**This matters for the "Dependency Gap" headline claim** (Fig. 1, abstract): if flagged records on real generators are partly rare-but-valid, the "hallucination" interpretation overstates.

Suggested remedies (any one substantially de-risks the claim):
1. Elevate the arithmetic-constraint experiment to a primary construct-validity study: show, for the flagged vs. unflagged cohorts, the empirical rate of confirmed arithmetic violations (this is directly computable, and the LSE 100%/NIC 0%/Rules 63% decomposition in §4.1 suggests the machinery for it already exists).
2. Quantify out-of-support flagging: for one real-generator cohort, report what fraction of flagged records fall inside vs. outside the training data's convex/nearest-neighbor support (e.g., distance-to-nearest-real-row distribution for flagged vs. unflagged).
3. Reframe consistently: $H$ certifies "inconsistent with the learned conditional manifold," and reserve "invalid/hallucinated" for the constraint-verified cases. Some prose already gestures at this; the abstract and Fig. 1 caption should follow.

### M2. Multiple-comparison control across the utility-filtering tests (gating)

16 dataset–generator configurations are tested for $\Delta F1$; 5 reach $p < 0.05$ at $\alpha=0.05$ (4 positive, 1 negative). Under a global null one expects $\approx 0.8$ false positives, so the aggregate pattern is clearly not noise — but *per-effect* inference is fragile. Under Bonferroni ($\alpha = 0.05/16 = 0.0031$):

| Config | $\Delta F1$ | $p$ | Survives Bonferroni? |
|---|---|---|---|
| Purchases Vine | +0.399 | <0.001 | ✓ |
| Census Vine | +0.195 | <0.001 | ✓ |
| Census CTGAN | +0.104 | 0.002 | ✓ (marginal) |
| Purchases CTGAN | +0.128 | 0.037 | ✗ |
| Credit Gaussian | −0.021 | 0.013 | ✗ |

The headline abstract/intro claims include the $+12.8$ Purchases-CTGAN effect (abstract, §1), which is the least robust of the positive results, and its 95% CI is never reported. This is defensible (each test answers an independent question), but it should be addressed explicitly rather than left implicit. Recommend: (a) report a one-sentence FDR or Bonferroni sensitivity note, and (b) report CIs for all four positive effects — the two largest effects are on retained cohorts of 13.8% and 76.8%, so the magnitude noise is worth quantifying. The $+0.104$ (Census CTGAN) claim would then rest on the CI $[+0.049, +0.160]$, which is honest but wide.

### M3. Baseline coverage (gating partially)

**(a) Missing a density/likelihood baseline.** The held-out comparison pits HIF only against IF and LOF. These are the natural *geometric* baselines, but they are also the family least likely to detect marginal-preserving conditional violations — so "HIF > IF/LOF on dependency violations" is partly expected. The strongest competing hypothesis for "is this row consistent with the learned joint distribution" is a direct density estimate (e.g., likelihood under the trained generator, VAE/autoencoder reconstruction error, or a nonparametric density model). Adding at least one such detector would substantially strengthen the claim that HIF's advantage is due to *logic-aware conditional* modeling rather than merely "any joint-model beats any geometric model." If this is infeasible, an explicit sentence explaining why it is not a meaningful baseline would help.

**Resolution: addressed, with a material consequence.** A learned-density baseline (BIC-selected GMM scored by negative log-likelihood, contamination-calibrated exactly like IF/LOF) was added to both held-out scripts and both datasets (`fit_gmm`/`detect_gmm` in `scripts/05_heldout_error_baselines.py`; results in `outputs/heldout_errors_*` and `outputs/heldout_online_purchases/`). The result honestly weakens the headline claim: on Census ACS the GMM nearly ties HIF on dependency violations (ROC-AUC 0.820 vs. 0.830; PR-AUC 0.797 vs. 0.789; matched F1 0.710 vs. 0.720), and on Online Purchases it *beats* HIF on the target family (ROC-AUC 0.955 vs. 0.797; matched F1 0.891 vs. 0.665) and on feature dropout and random noise. The manuscript now reports this in full (Tables 4–5, §6, Conclusion, Limitations) and re-frames HIF's advantage as domain-dependent: competitive-or-better than learned-density baselines on categorical-heavy, non-arithmetic conditional structure (where it also adds per-dependency attribution), while learned-density models win on near-deterministic continuous arithmetic constraints. This is the honest reading of the data and is a net credibility gain versus omitting the baseline.

**(b) The Rule-only baseline is described but never reported.** §5.2 introduces "the Rule-only Baseline" as a comparison, but no result for it appears in the manuscript. The committed outputs already contain the numbers, and they are a *point in the paper's favor*:

- Census ACS CTGAN: Full $0.427 \pm 0.029$ → Rule-only $0.426 \pm 0.029$ (retention 97.0%) vs. HIF-filtered $0.532 \pm 0.044$ (retention 65.7%).
- Census ACS Gaussian Copula: Rule-only $0.937$ vs. HIF $0.937$ (both ≈ full cohort).

Rule-only pruning (~3% of records) changes nothing; the entire utility recovery comes from the LSE/NIC conditional-dependency layers. This directly addresses the reviewer attack that "the audit validates its own usefulness on its own flags" — please put these numbers in the text.

### M4. Novelty framing should match the actual contribution

All components are established: RF conditional predictors, quantile/OOB calibration (conformal lineage), MCA/SVD embedding, gradient boosting, apriori mining, geometric-mean aggregation, accuracy-based feature scoring. The contribution is genuinely *integrative*, and for an applied venue that is sufficient — but the current framing ("we introduce the Hybrid Integrity Framework," "two primary contributions" in §1) overstates. Recommend tightening to: *a principled, validated integration of existing building blocks that reframes synthetic-data auditing as row-level joint-conditional plausibility with per-record attribution*, and add explicit positioning against the two closest lines of work that are currently only partially covered: (i) classifier/detectability-based synthetic-data evaluation (train-on-real-vs-synthetic probes), and (ii) conditional-independence testing as an audit — the citations already present (Valiant, Pensia, Neykov, Paninski) should be connected to the practical claim, since the paper currently name-checks sample-complexity limits without using them to bound HIF's own reliability.

### M5. Scope and strength of the utility claim

The abstract's "Targeted Remediation" framing is the contribution reviewers will quote, and its boundary conditions are worth stating more prominently: only 4/16 configurations show significant recovery; the largest effects come with large-cohort destruction (Purchases Vine $+0.399$ at 13.8% retention; Purchases CTGAN $+0.128$ at 15.1% retention); and whether filtering helps depends on the strength of the violation→target association, not the violation rate (correctly argued in §5.2, but worth putting on the abstract's terms). Recommend an explicit deployment-facing statement: at what retention/utility trade-offs is filtering a net win.

### M6. Inferential scope

All inferential statistics (significance tests, held-out comparisons, calibration) are confined to Census ACS; the cross-architecture audit is descriptive. The manuscript says this clearly (which earns credit), but reviewers will press on whether "HIF outperforms baselines" generalizes. A cheap mitigation: run the IF/LOF comparison on a second dataset (Online Purchases, which has the starkest dependency failures) to show the ROC-AUC pattern is not Census-specific.

**Resolution: addressed, with nuance.** The full held-out protocol was replicated on Online Purchases (`outputs/heldout_online_purchases/`). The geometric-baseline claim replicates cleanly: HIF beats IF and LOF on dependency violations (ROC-AUC 0.797 vs. 0.473/0.583), feature dropout, and random noise on the second dataset. However, the new learned-density baseline beats HIF there (see M3a), so the replication simultaneously confirms the geometric claim and delimits the density claim — exactly what the manuscript now reports. The cross-architecture audit remains descriptive; the §5.1 caveat still covers it.

---

## 5. Minor Comments

1. **"Violation rate" is used with two meanings.** In Table 2 (cross-architecture) it means "fraction of synthetic records flagged by HIF"; in Table 5 (ablation) the column is labeled "Violation Rate" but contains the *rejection* rate (e.g., "No filtering 0.0%"). Consider renaming the ablation column "% Flagged / Retention loss," and note that "No filtering — 0.0%" is trivially true.
2. **PR-AUC is reported once** (§6, dependency violations: 0.789) despite §5.1 declaring threshold-independent ranking quality a primary metric. Report PR-AUC for all five error families in Table 4.
3. **Missing CIs** for the Purchases effects ($+0.128$, $+0.399$) in the text; the paper reports CIs for the Census significance tests — do the same here (see M2).
4. **Fig. 1 caption** says "KS ≈ 0.96" for Gaussian Copula 1D marginals of Quantity/Total; Table 2 reports dataset-level KS $0.903 \pm 0.006$ for that configuration. These are different quantities (per-column vs. aggregate) and likely consistent, but the caption should say so to avoid a reviewer "contradiction" flag.
5. **§3 (Related Work) is one wall of text**; the "Evaluation Metrics and the Dependency Gap" paragraph is a single multi-claim paragraph of several hundred words. Split into 2–3 paragraphs.
6. **Hub stability** is reported as a list (appendix) but not quantified across seeds; a one-line "hubs identical across the 10 seeds" (if true) would close a natural question given $S(h)$ is a CV accuracy.
7. **"Integrity–Fidelity Decoupling"** is a central term used repeatedly; consider defining it once near its first use (it appears in the abstract, §4.1, and §7 with slightly different wording each time).
8. **Online Purchases TVAE** (Table 3) reports F1 $0.889$ at 1.1% retention with an N/A in the HIF column; the $0.889$ itself is estimated on a 1.1% cohort and is as unreliable as the HIF column — mark both N/A for consistency.
9. **Example consistency:** the "14-year-old with a PhD" example sits in a Census ACS methodological context, but the Census hubs are income/housing/poverty/education/tenure; the example reads as a demographic-education constraint that better matches the Adult benchmark. If it is illustrative, say so.
10. **Terminology audit:** "integrity," "fidelity," "validity," "plausibility" are used near-synonymously across sections; a short definitions paragraph (as done well for the frontier) would sharpen the whole paper.

**Resolution for #6 and #10:** Hub selection is deterministic by construction (sentinel discovery uses a fixed `random_state=42` with non-shuffled CV folds — the `seed` argument to `hif_score` is not propagated to hub discovery), and the hub sets were verified identical across all 10 held-out seeds for every benchmark dataset; the appendix (§ LSE Implementation Details) now states this explicitly with the Census ACS and Online Purchases hub lists. A Terminology paragraph is added at the start of §4 (Mathematical Foundation) defining *fidelity* (aggregate), *integrity* (row-level, the audited property), *plausibility* (per-column marginal), and *validity* (ground-truth possibility), resolving the near-synonymy across sections. (#7 was already addressed earlier: "Integrity–Fidelity Decoupling" is now defined at its first substantive use in §4.1; #9 was already addressed by marking the PhD example as illustrative.)

---

## 6. Reproducibility Assessment (independently verified)

This is the strongest aspect of the submission and deserves explicit credit.

- **Data:** exact byte-identical copies of all five dataset snapshots are committed (`data/cache/*.parquet`, md5-verified), so the paper is reproducible offline, independent of the live Census API/UCI.
- **Environment:** generator stack pinned in `pyproject.toml` (`sdv==1.32.0`, `ctgan==0.11.1`, `pyvinecopulib==0.7.5`, …); installed versions verified.
- **Bit-exact re-runs:** all 11 paired t-tests reproduce exactly (e.g., Census CTGAN $t=-4.234$, $p=0.0022$, CI $[+0.049, +0.160]$; Online Vine $\Delta F1 = +0.399$, $p<0.001$); the calibration permutation CSV regenerates byte-identical; the held-out ROC-AUC/F1 tables match; Figure 1 (dependency gap) matches (modulo matplotlib trailer bytes); all four tables and the appendix's hub/hyperparameter/scalability numbers match committed outputs.
- **Tests:** 173 passing (1 xfailed); pre-commit clean; CI green.

If a revision is submitted, I would only add a DOI/Zenodo release of the repo snapshot (github URLs age); everything else meets or exceeds current reproducibility norms.

---

## 7. Novelty and Significance Assessment

- **Novelty:** integrative, not paradigmatic. No single component is new, but the *problem formulation* — per-record, per-attribution auditing of joint conditional integrity with a non-compensatory aggregate — is a genuine reframing of a real evaluation gap, and the decoupling evidence (KS-fidelity ≁ integrity) is a novel empirical finding the field should know about. For an applied data-science venue this is sufficient.
- **Significance:** moderate-to-good. The tool is practically useful, cheap ($O(N \cdot K)$; ~1.5 s per 10k rows), and the honest characterization of *when* filtering helps/hurts is more useful to practitioners than an over-optimistic positive result would be. The main limit on significance is that the utility payoffs are selective and the default threshold ($H<0.5$) is heuristic, both acknowledged.

---

## 8. Questions for Authors

1. For real generator cohorts (not injected corruption), what fraction of HIF-flagged records can be *independently* confirmed as invalid (e.g., arithmetic-identity verification on Supermarket Sales)? What is the confirmation rate among unflagged records?
   **Resolution:** Run as a dedicated external-anchor study (`scripts/09_arithmetic_identity_verification.py`, $N=10$ seeds, both constrained datasets × 4 generators, output under `outputs/identity_verification.csv` and `_summary.csv`). Real-data arithmetic identities (Supermarket: $\text{Total} = \text{Unit Price} \times \text{Quantity} \times 1.05$; Online Purchases: $\text{item\_total} = \text{item\_subtotal} + \text{item\_tax}$, $\text{item\_subtotal} = \text{purchase\_price} \times \text{quantity}$) are satisfied to $\sim 10^{-13}$ relative error in the real data, and all four generators violate them wholesale (base violation rate $0.87$--$1.00$). Flagged records ($H < 0.5$) are confirmed as identity-violating in $\ge 98.9\%$ of cases in all eight dataset--generator configurations (1.000 in six). Among unflagged records the confirmation rate is also high (98.4--100\%) because the constraints are violated almost everywhere; the only config with a substantial valid cohort (Gaussian Copula on Online Purchases, 45\% unflagged) shows confirmed-invalid 0.720 unflagged vs. 0.989 flagged, and the HIF penalty correlates with identity severity there (Spearman $\rho = 0.47$). Where HIF flags essentially the whole cohort, its penalty is only weakly monotone in arithmetic severity ($\rho \le 0.18$). Reported in §4.1 with Table~\ref{tab:identity_verification}; interpretation: flagged = externally invalid at near-universal rates, but HIF's *ranking* of arithmetic-constraint severity is limited — exactly the regime where the learned-density baseline of §6 is sharper.
2. Do the Census ACS ROC-AUC patterns in Table 4 replicate on a second dataset (e.g., Online Purchases)?
3. How sensitive are the utility-recovery conclusions to the $H<0.5$ threshold choice (the appendix shows score-level insensitivity on Census ACS — is that also true for downstream F1)?
   **Resolution:** Addressed with a dedicated downstream-F1 threshold sweep (`scripts/10_threshold_utility_sensitivity.py`, Census ACS CTGAN, $N=10$ seeds, output under `outputs/threshold_utility_sensitivity_{raw,summary}.csv`). Filtering at $H \in \{0.3, 0.5, 0.7\}$ yields F1 $0.723/0.531/0.451$ at retention $30.9\%/65.7\%/92.6\%$, each significantly above the unfiltered $0.427$ ($p \le 0.020$; CIs $[+0.215,+0.375]$, $[+0.049,+0.160]$, $[+0.005,+0.042]$). The recovery effect is monotone in filtering strictness and remains significant at the loose $H=0.7$ operating point; the default $H=0.5$ reproduces the reported $+0.104$. Reported in §5.1 with Table~\ref{tab:threshold_utility}.
4. Is there a computable relationship between the sample-complexity bounds cited in §3 (Valiant; Pensia; Neykov) and HIF's own reliability given the hub-conditioned sample sizes actually used (2,000 rows, up to 10 features)?
   **Resolution:** Addressed with a dedicated instantiation script (`scripts/11_sample_complexity_bounds.py`, output `outputs/sample_complexity_bounds.csv`). Inverting the identity/uniformity-testing bound $m \gtrsim \sqrt{S}/\varepsilon^2$ at HIF's per-hub, per-category cell sizes yields per-cell detectability floors $\varepsilon_{\text{cell}} = (S_{\text{cell}} / n_{\text{cell}}^2)^{1/4}$ of $0.27$--$1.0$ across the five benchmarks (cohort-level analogue $0.15$--$0.19$). The worst floors occur where long-tail hub categories leave small conditioning cells (Adult \texttt{native\_country} median 4 rows/cell; Credit \texttt{pay\_*} median 12; Online Purchases \texttt{quantity} min 1); Census ACS and Supermarket Sales, with balanced hub categories ($n_{\text{cell}} \ge 100$), achieve $\varepsilon \approx 0.27$ and $0.32$. This formalizes the granularity below which an individual sentinel cell cannot certify deviations and motivates the confidence-floor safety minimum and multi-hub geometric aggregation. Reported in §3 and Appendix~\ref{app:sample_complexity} with Table~\ref{tab:sample_complexity}.

---

## 9. Verification Note

Where this review cites specific numbers (effects, $p$-values, CIs, KS values, retention, rule-only baselines), they were checked against the committed artifacts under `outputs/` and the bundled snapshots. The Rule-only numbers in M3b and the Bonferroni arithmetic in M2 are derived from those artifacts and are correct as stated.

**Post-review addendum (GMM baseline + second-dataset replication).** The numbers added in response to M3a/M6 were verified from the regenerated artifacts: Census ACS GMM values in `outputs/heldout_errors_summary.csv` (GMM ROC-AUC/PR-AUC) and `outputs/heldout_errors_matched_summary.csv` (`gmm_f1_threshold_mean`, `gmm_f1_matched_mean`); Online Purchases values in `outputs/heldout_online_purchases/`. Non-GMM rows of the Census outputs are byte-identical to the committed versions, confirming the rerun did not perturb prior results. The GMM component sweep (BIC over $k \in [1,6]$, full covariance, `reg_covar=1e-4`) matches the implementation described in the manuscript; results in §6, Tables 4–5, Conclusion, and Limitations reflect the CSV values exactly.
