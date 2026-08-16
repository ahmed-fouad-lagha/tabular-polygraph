# Cover Letter / Response to Reviewers

**Manuscript:** *Row-Level Structural Integrity Diagnostics for Synthetic Tabular Data*

**Authors:** Ahmed Fouad Lagha, Zakarya Farou, Imre Lendák

**Venue:** SN Computer Science

**Code and data:** https://github.com/ahmed-fouad-lagha/tabular-polygraph

---

Dear Editor,

We are resubmitting the revised manuscript *Row-Level Structural Integrity
Diagnostics for Synthetic Tabular Data*. In this revision we corrected three
metric-level bugs identified during our pre-submission review, regenerated every
experiment output from the corrected implementation, and re-derived all reported
numbers to match. The regenerated results preserve the paper's main findings and
let us state several of them more honestly than before. The full repository
(revision, golden-value tests, lockfile, and CI) accompanies this resubmission;
all results are reproducible from the committed scripts and committed outputs.

## 1. Metric-correctness fixes

The integrity scores the manuscript reports are computed by an auditable,
open-source implementation. A line-by-line audit against the published
references surfaced three defects, all now fixed and pinned by regression tests:

- **Joint Correlation Distance (JCD) was sign-blind.** The numeric--numeric
  block of the association matrix used `|ρ|`, so a perfectly *inverted*
  correlation scored 100/100. JCD now uses the signed Spearman ρ and the
  corrected Frobenius normalizer `2·√(n(n−1))` over the full mixed-type matrix,
  so inverted dependence is penalized (regression test:
  `test_jcd_perfectly_inverted_is_penalized`).
- **Moment Matching was not location-invariant.** The mean discrepancy divided
  by `|mean_real|`, so identical N(0,1) samples scored 59/100. The divisor is now
  `max(std_real, eps)`, making the metric invariant to location shifts of the
  mean (regression test: `test_moment_matching_identical_is_location_invariant`).
- **α-Precision / β-Recall deviated from the cited construction.** The coverage
  curve is now computed from synthetic-nearest-neighbor distances as in Alaa et
  al. (ICML 2022), rather than from real-data radii, and `authenticity` was
  inverted relative to the published direction (a verbatim copy previously
  printed 1.000). Authenticity now follows Alaa et al.: verbatim copy ≈ 1.0,
  fresh draw ≈ 0.51, collapsed ≈ 0.05. The remaining deviation of our β-Recall
  (no local-coverage factor; thresholds against real-data radii) is disclosed in
  the manuscript, and β-Recall is explicitly described as a coarse support
  indicator rather than a reproduction of the published metric.
- **Console output.** KS is printed as a percentage, a TVD-complement line was
  added, and failed metric computations surface a WARNING instead of a silent 0.0.
- **`random_state` is now threaded end-to-end** (orchestrator → pipeline →
  downstream and α/β metrics → loaders → all experiment scripts), so every seed
  in the paper's `N=10` runs draws its own real-data sample and is fully
  reproducible.
- **Generator fixes.** No generator mutates the global RNG anymore;
  Gaussian Copula uses cumulative-midpoint marginals that round-trip exactly
  (previously `['A','B','C']` collapsed to `['A','A','B']`); the Vine Copula
  seeds each retry batch separately, eliminating duplicate rows when filters are
  active.

## 2. Full regeneration of all experiment outputs

Every experiment script (`scripts/02`–`15`) was re-run with the corrected
implementation, and all committed outputs under `outputs/` and
`outputs/heldout_online_purchases/` were regenerated. Each run now uses
per-seed real-data draws. `logs/regenerate_all.sh` reproduces the full pipeline
and is resumable (`--resume`); script 03 saves incrementally. The manuscript
(abstract, Tables 1–5, both appendix tables, Figure 1) has been updated to the
regenerated numbers throughout, and the PDF rebuilds cleanly.

## 3. Verification

- **Golden-value tests** (`tests/fidelity/metrics/test_golden_values.py`, 12
  tests) compare KS, TVD, JCD, Moment Matching, and α/β against independently
  hand-computed references and pin the regression fixes above.
- **Lockfile and CI.** `requirements.lock` (pip-compile, 247 pins) fixes the
  dependency stack; CI installs `.[test,vine,ctgan,tvae]`, caches on the
  lockfile, and no longer excludes the Vine/CTGAN generators from coverage. The
  full suite passes under the locked stack (201 passed + 1 expected failure,
  82.8% coverage); a final float-precision fix makes the golden-value assertion
  for KS robust across scipy versions in CI.

## 4. What the regenerated results honestly establish

The corrected metrics changed some numbers, not the paper's core argument. We
report the changed claims explicitly:

- **The utility-recovery result is label-conditional, not structural.**
  Integrity filtering improves downstream utility precisely when the auditor's
  selected dependency hubs include the prediction target. On Census ACS the
  regenerated gains are **+12.6 F1 points for CTGAN** (paired t-test p = 0.001;
  95% CI [+0.068, +0.185]) and **+17.9 points for the Vine Copula** (p < 0.001;
  CI [+0.144, +0.214]), while on a largely error-free cohort (Gaussian Copula)
  filtering is utility-neutral (p = 0.90). Two controlled experiments isolate
  the mechanism: across nine Census ACS targets, **9 of the 10 hub
  target–generator pairs recover significantly** (the sole exception is
  CTGAN×tenure, p = 0.080), whereas **only 1 of 8 non-hub pairs does** on
  identical retained cohorts (Vine Copula×cost_burden_pct, p = 0.022); and
  withholding the target from hub selection eliminates the gain. We therefore
  present the effect as conditional label curation rather than generic
  structural repair.
- **Bonferroni.** Across the 16-configuration battery (α = 0.05/16 ≈ 0.003),
  only the **two Vine Copula recoveries** remain significant; the CTGAN gains
  (p = 0.008 and p = 0.016) fall below the corrected threshold and are reported
  as suggestive rather than conclusive.
- **Threshold sweep.** The recovery is significant at the strict (H ≥ 0.7) and
  default (H ≥ 0.5) frontiers and grows with strictness, but **is no longer
  significant at the loose H ≥ 0.3 frontier** (p = 0.92), where only the most
  egregious records are removed.
- **No configuration shows a significant degradation** from filtering; the
  nearest is Credit Gaussian Copula (ΔF1 = −0.020, p = 0.073).
- **Real-data reference floor.** The auditor flags genuine held-out real data at
  **0.0–14.0%** across domains, and the auditor scores its own training rows
  higher than fresh held-out rows (a framework-level overfitting signature). We
  report this reference row per dataset so every generator is read against the
  metric's own error floor.
- **Held-out baselines.** At matched operating points HIF† leads the learned-
  density baseline on Census ACS dependency violations (F1 0.719 vs. 0.710 for
  GMM†), but **GMM leads on Online Purchases** (0.890 vs. 0.668). We report this
  boundary explicitly: HIF's advantage is concentrated in predominantly
  continuous spaces with complex non-arithmetic conditional structure, and we
  recommend the density baseline for near-deterministic continuous arithmetic
  domains.
- **Vine Copula arm.** We now report the Vine Copula arm as a *designed positive
  control* whose categorical columns are sampled independently from their
  marginals (disclosed in the released implementation), which is exactly why it
  shows the highest marginal fit (KS ≥ 0.97) alongside pervasive dependency
  violations.

## 5. Additional disclosures (this revision)

In response to the review's concern that HIF can be "gamed", we have added
explicit, quantified disclosures to the manuscript (no numbers changed):

- **Mode collapse is not caught everywhere.** A degenerate generator emitting
  one real row 2,000 times scores HIF 1.000 with 0.0% violations on Census ACS,
  Adult, and Credit Default — a necessary-but-not-sufficient boundary we now
  state as a dedicated limitation, with the same construction caught at 100%
  violations on the transaction domains by the mined arithmetic rule layer.
  TVAE's Adult cohort (HIF 0.990, α-Precision 0.350) is used as the concrete
  worked example, and we recommend reading HIF jointly with a support-coverage
  diagnostic.
- **NIC is scoped as a categorical-context check.** We now state explicitly that
  NIC conditions only on the categorical manifold and cannot detect
  continuous–continuous arithmetic identities by construction, crediting the
  rule layer and LSE for that detection where it occurs.
- **NIC threshold calibration.** NIC calibrates its thresholds on in-sample
  residuals; we disclose that a 50/50 held-out re-calibration yields thresholds
  0.96×–1.54× looser on our benchmarks (≤ 1.05× on Census ACS), reporting the
  current measured effect rather than the larger gap observed on the
  pre-regeneration code.
- **Rule-layer hard application.** The 0.95-confidence mined rules are applied
  as hard penalties; the union false-positive rate on valid data is disclosed,
  and the formal union bound in Remark 1 now carries the rule-layer term.
- **TSTR protocol.** The TSTR holdout is carved from rows the generators were
  fitted on; we disclose that absolute TSTR levels are optimistically biased,
  while the paired filtering comparisons are protected.

## 6. Reproducibility

All results are reproducible from the committed repository: scripts `02`–`15`,
committed outputs, golden-value tests, the pip-compiled lockfile, and a
green CI run on the locked stack. A full reproduction is available via
`logs/regenerate_all.sh` (resumable).

We thank the reviewers for the rigor that surfaced these defects, and we hope
the revision addresses them. We would be glad to clarify any point.

Sincerely,

Ahmed Fouad Lagha, on behalf of all authors
