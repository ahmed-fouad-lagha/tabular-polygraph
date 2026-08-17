# Pre-Submission Review — *Row-Level Structural Integrity Diagnostics for Synthetic Tabular Data*

Target venue: SN Computer Science. Review date: 2026-08-11. Commit: `a80df9c`.

Findings are numbered to match the review conversation. Every finding marked
**[measured]** was verified by executing the code in this repository; the
measurement command is given so you can re-check it.

---

## 0. Verdict

The engineering is solid and the statistics are sound: all 120 cells of Table 1,
all 56 of Table 2, all 144 across Tables 4–5, and all 16 paired tests reproduce
exactly from the committed CSVs. The prose is unusually candid about negative
results.

The problem is not the arithmetic. It is the **gap between what the code does and
what the prose says it does**. Three findings change what the paper can claim:

1. **The headline utility-recovery result is label-conditioned, not structural**
   (§1, §1b). The downstream target is itself the #1 LSE hub; when it is excluded
   from the hub set all four significant gains vanish, and on identical retained
   cohorts every one of 10 hub targets recovers while none of 8 non-hub targets
   does. The mechanism is identified, so the claim can be restated precisely
   rather than dropped.
2. **The Vine Copula baseline cannot model categorical dependence by
   construction** (§2), so the flagship "even rigorous statistical models
   hallucinate" claim is a tautology of the implementation.
3. **α-Precision and β-Recall are used and interpreted with no attribution**
   (§3), and β-Recall deviates from the published formula it is named after.

Everything else is fixable by rewriting, relabelling, or one script run. My
estimate: a focused week, mostly writing rather than computing.

---

## 1. Priority checklist

### Must fix — changes what the paper claims

- [x] **§1/§1b** Reframe contribution (ii). The utility recovery is
      label-conditioned: it vanishes when the target is withheld from the hub set
      (§1) and it never appears for a non-hub target on the identical retained
      cohort (§1b, 10/10 hub targets vs 0/8 non-hub). The "find a non-hub target
      where the effect holds" escape route has been tested and closed. Present it
      as label-dependent curation. Affects: abstract (both headline numbers),
      §1 contributions, §5.2, Table 3, §6.2 deployment guidance, Conclusion.
      **Done** — reframed as "Label-Conditional Curation" throughout: abstract
      (247 words, under the SN CS cap), §1 contribution (ii), new §4.3 *The
      Recovery Effect Is Label-Conditional* with a two-panel Table 4 reporting
      both experiments, Table 3 caption, §6.2 deployment rule, sixth limitation
      on label-noise overlap, Conclusion. Builds clean at 35 pages.
- [x] **§2** Reframe the Vine Copula arm as a designed positive control, or
      replace it with a mixed-type vine. Affects: abstract, §1, §5.1, §6.1,
      Conclusion. **Done (author approved Option A: reframe, no re-run).** The
      Vine arm is now a designed positive control whose categorical columns are
      sampled independently from their marginals and whose categorical--numeric
      dependence is deliberately not modelled (mechanism verified in
      `vine_copula.py`; disclosed in the released implementation). Reframed in
      the abstract ("a Vine Copula control", still 248 words < 250 cap), §1
      (removed "even rigorous statistical models ... confidently hallucinate"),
      §1 related work (dropped Vine as exemplar of statistical models), §5.1
      (arm's near-perfect KS attributed to resampling the empirical marginals),
      §6.1, and the Conclusion. All numbers unchanged; rebuilt clean at 35
      pages, zero `???`, zero undefined refs, zero overfull hboxes.
- [x] **§3** Cite Alaa et al. (ICML 2022) for α-Precision/β-Recall; fix or
      explicitly disclose the β-Recall variant. **Done** — cited, and the
      hypersphere-radius deviation disclosed in the new Comparison Metrics
      paragraph.
- [x] **§4** Rewrite the SHAP passage — that benchmark was never run. **Done** —
      now explicitly analytical, with the TreeSHAP complexity claim corrected.
- [x] **§5** Narrow the novelty claim to the constraint-mining bridge; cite and
      distinguish TabStruct (ICLR 2026 Oral) and Curated LLM (ICML 2024). **Done** —
      contribution (i) now states the bridge explicitly ("first framework to apply
      functional-dependency and denial-constraint discovery to per-record auditing of
      synthetic tables"); new related-work paragraph "Record-Level Auditing of Synthetic
      Data" distinguishes α-Precision/β-Recall (embedding support), Data-SUITE (per-feature
      conformal, real data), DOMIAS (privacy signal), TabStruct (dataset-level, needs causal
      ground truth), and Curated LLM (supervised label-conditioned curation); bib gained
      `seedat2022suite`, `vanbreugel2023domias`, `seedat2024curated`, `jiang2025tabstruct`,
      `huhtala1999tane`, `chu2013direct`. Clean rebuild 35→36 pages, zero `???`, zero
      undefined/multiply-defined, zero Overfull.
- [x] **§6** Swap the inverted threshold labels (Table 7, appendix). Headline
      number unaffected. **Done** — fixed in all three locations (§4.2 prose,
      appendix table, appendix hyperparameter prose) plus `scripts/10`, which now
      emits a `retention_frontier` column. CSVs regenerated bit-identical.
- [x] **§7** Explain or eliminate the 15× gap between Tables 1 and 2. **Done** —
      explained as a protocol difference, verified in code and CSVs: Table 2
      (`scripts/09_arithmetic_identity_verification.py`) re-fits HIF on the projection onto
      the five identity-bearing columns (`real.columns.intersection(synthetic.columns)`,
      `_exp_utils.py:90`), where the arithmetic identity becomes a near-deterministic hub,
      whereas Table 1 (`scripts/03_cross_architecture_benchmark.py`) audits the full feature
      space across all hubs. Supermarket Gaussian Copula: 2.6% (full) vs 33.1% (projection),
      both at H < 0.5. Caption note (Table 2) + narrative sentence in §5.1 added. Clean
      rebuild 36→37 pages, zero warnings, zero `???`.
- [x] **§8** Reframe the covariate-shift and row-duplication rows as sanity
      checks; unbold the sub-chance "winners". **Done** — flag-fraction
      disclosure in §5.3 now covers IF/LOF/GMM with means and per-family ranges
      (GMM ≈71%, 57–80%; LOF ≈79%, 63–87%; IF ≈39%, 30–42%); LOF's ≈79% inside
      the dependency-violation sentence corrected to ≈86%; both overclaim
      clauses ("edge on row duplication", "IF on covariate shift") removed and a
      new negative-controls paragraph added (row duplication = in-distribution
      copies → chance; covariate shift = verbatim real records → sub-chance is
      correct; the 2πq/(π+q) flag-rate identity explains LOF 0.535 and GMM 0.550);
      all at/below-chance bolded cells un-bolded in Tables 5 and 6; both captions
      note the negative controls and random baselines; first limitation rewritten
      (no row-level method, including the baselines, detects distribution-level
      drift — the domain of two-sample tests). Rebuilt clean at 35 pages, zero
      `???`, zero undefined refs, zero overfull hboxes.
- [x] **§9** Fix the three wrong-paper bibliography entries. **Done** — CoDi,
      Angelopoulos and Barber all corrected against arXiv/Crossref/Project
      Euclid.

### Should fix — materially strengthens the paper

 - [x] **§10** Add a real-data reference row to Table 1; report HIF's measured
       false-positive rate. **Done** — new `scripts/14_real_data_reference.py`
       (deterministic, `outputs/real_data_reference.csv`): auditor fit on real
       rows, scored on a genuine held-out remainder (2,000 extra rows adult/credit;
       residual 462 census_acs; 80/20 split purchases/supermarket). Table 1 now has
       a "Real data (held-out)" row per dataset (KS/α/β = —): Supermarket
       0.980/0.0%, Purchases 0.917/6.8%, Credit 0.825/13.9%, Adult 0.922/4.3%.
       Caption + evaluation-protocol paragraph document the protocol, the error
       floor (0.0–13.9%), and the overfitting signature (replay > held-out: census
       0.985 vs 0.964, adult 0.977 vs 0.922). Paragraph-2 TVAE claim rewritten:
       Adult TVAE (0.981/0.5%) exceeds real data; Credit TVAE (0.827/13.5%) sits at
       the real floor. §18 appendix numbers reconciled to canonical values (0.0/1.3/
       4.3/6.8/13.9%). README:138 fixed (script 03 does NOT emit a real-data
       baseline; script 14 documented). Note: §10's ad-hoc census 0.906/4.1% not
       reproduced (1.3% here) — canonical values supersede. Clean rebuild 38 pages.
 - [x] **§11** Add a "necessary but not sufficient" limitation with the
       mode-collapse numbers. **Done** — Seventh limitation in §6.2: 2,000 copies
       of one real row score HIF 1.000/0.0% on Census ACS, Adult, Credit; caught at
       100% violations on Online Purchases/Supermarket (H = 0.0464 = (1e-4)^(1/3));
       TVAE Adult (0.990 vs α-Prec 0.350) as the worked example. No headline numbers
       changed.
 - [x] **§12** Reconcile NIC's predictor set with Figure 1's motivating example.
       **Done** — NIC subsection now scopes NIC as a categorical-context continuity
       check (conditions only on φ(x_cat), cannot see continuous–continuous
       identities by construction); detection credited to rule layer + LSE
       (Supermarket NIC component rate 0.0% cited). No code change.
 - [x] **§13** Fix NIC's in-sample threshold calibration. **Done (disclosure
       only)** — code fix declined (would shift every reported number); sensitivity
       disclosed with freshly measured 50/50 ratios 0.96×–1.54× looser (Census
       ACS ≤ 1.05×); the review-era ~2× does not reproduce on the current pipeline.
 - [x] **§14** Disclose the rule layer's 0.95-confidence behaviour; add a rule term
       to Theorem 3. **Done** — mining thresholds + hard 0/1 application +
       ε-floor disclosed in the rule-layer paragraph; **Remark 1** (the demoted
       Theorem 3) union bound now carries the γ_rule term.
 - [x] **§15** Disclose the TSTR leak, or refit generators on the train split.
       **Done (disclosure only)** — "Protocol disclosure" paragraph in §5.2:
       absolute TSTR levels optimistically biased, paired ΔF1 protected, TRR clean;
       refit declined (would invalidate committed outputs).
 - [x] **§16** Correct the "categorical-heavy Census ACS" framing and the GMM
       explanation. **Done** — verified Census ACS = 9 numeric + `state` (51 levels), all five
       hubs (`household_income`, `housing_cost`, `poverty_status`, `education`, `tenure`)
       quantile-binned continuous; GMM baseline encodes 9 standardized numeric + 51 one-hot
       `state` dims (60-D). Five locations corrected: §1 LSE layer (hubs "categorical columns
       or continuous features reduced to quantile bins"); §5.3 GMM-boundary and GMM-limitation
       paragraphs (60-D mostly-binary support, not "categorical-heavy"/"one-hot indicators"
       broadly); §5.5 ablation framing (LSE critical because hubs are binned continuous);
       Conclusion ("predominantly continuous Census ACS"; distinctive value = complex
       structure not reducing to arithmetic identities). Clean rebuild 37 pages, zero warnings.
 - [x] **§17** Fix scope overclaims: feature counts, row counts, "thousands of
       columns", LOF's flag rate, ρ = −0.98 provenance. **Done** — (a) sample-complexity
       paragraph now states real cohort sizes n∈{664,1000,2000} over 7–24 features
       (main.tex:88) and appendix ε_joint uses per-dataset n (appendix.tex:108); (b)
       scalability text no longer claims "thousands of columns"/D>10,000 (rows only, 10
       features, max D=24); (c) LOF flag-rate claims verified against
       `heldout_errors_raw.csv` (cross-family mean 79.1%, dep-family 85.5%, ≈86% text OK);
       (d) ablation claim fixed — LSE-only filters 4.5% not "<1%"; (e) decoupling
       correlations reframed as seed-level n=160 with non-independence caveat
       (0.038/0.325/0.824 seed-level vs 0.045/0.334/0.827 config-level, both verified);
       (f) p≤0.1 ρ corrected −0.98→−0.88 (pooled, verified); line-179 ρ reframed as
       per-seed −1.00/pooled −0.98 with "nearly tautological" caveat; (g) conditional-swap
       now described as drawing from real conditional pools (MM 89.04→89.21, KS 0.926→0.927,
       verified) instead of "preserves marginals"; (h) Table 3 N/A cells replaced — full
       F1 reported for all four, Purchases TVAE filtered F1 0.322 (2/10 seeds) with Δ=−0.567
       as exclusionary counterexample in prose; (i) dispersion labelled: SD (Tables 1,3),
       SEM (Tables 2,7). Clean rebuild 37 pages, zero warnings, zero `???`.
 - [x] **§18** Demote the three Theorems to Propositions/Remarks; state that HIF
       offers no conformal coverage guarantee. **Done** — appendix.tex: Theorem 1 →
       Proposition 1, Theorem 2 → Proposition 2 (both retain proof/proof-sketch),
       Theorem 3 → Remark 1; new pre-appendix paragraph states δ_h is a per-hub OOB
       5th-percentile, not a conformal quantile (no exchangeability, marginal ≠
       per-record), so HIF offers no conformal coverage guarantee; Remark 1's
       illustrative purity numbers (η=0.66 → 96%) replaced with the measured
       false-rejection rate 4–15% on held-out real data (4.1% Census ACS, 6.5%
       Purchases, 10.1% Adult, 15.5% Credit) via \eqref{eq:purity}; main.tex:161,171
       references updated to Proposition 1/2, Remark 1. Clean rebuild 37 pages,
       zero warnings, zero `???`.
- [x] **§19** Cite Isolation Forest, LOF, Apriori, SHAP; define JCD and Moment
      Matching. **Done** — all four citations added; JCD, Moment Matching, the KS
      complement and TVD defined in the new Comparison Metrics paragraph, with
      the note that JCD is a similarity despite its name.
- [x] **§20** Delete the three econometrics citations. **Done** —
      `\cite{bollerslev1986, dickey1979, johansen1991}` stripped from the
      Conclusion (kawawa2026tradefm kept) and all three entries deleted from
      `references.bib`. Rebuilt clean: 34 pages, zero `???`, zero undefined
      refs, zero overfull hboxes.
 - [x] **§21** Commit a multiple-comparisons script; fix README `--seeds`. **Done** —
       new `scripts/15_multiple_comparisons.py` recomputes the 16-config battery from
       `outputs/full_benchmark.csv` (paired t-test, Wilcoxon, 95% CI, Cohen's d) with
       Bonferroni α=0.05/16=0.0031; every manuscript p-value reproduced exactly
       (census CTGAN 0.0022, Vine <0.001, Purchases CTGAN 0.0369, Credit Gaussian 0.0131;
       five of sixteen reach p<0.05; the same three survive Bonferroni as claimed).
       Writes `outputs/multiple_comparisons.csv` and documents the four non-estimable
       transaction configs. README.md:130/133 `--seeds 3` → `--seeds 42,43,44` (script 02
       treats `--seeds` as a list, so `3` meant one seed). Both `05_*` docstring commands
       fixed (`05a` `--seeds 42,43,44` crashed on int parse → `--seeds 10 --corruption-levels 0.4`;
       `05b` `--levels` → `--corruption-levels` + full seed list). README documents script 15.

### Venue compliance — mechanical, do last

- [x] **§22** Add `\ref` for both figures; reorder table citations. **Done** —
      Fig. 1 cited in §1, Fig. 2 in §3; both in ascending order. Table citations
      are in order within the body; the two appendix tables are forward-referenced
      from §4.2, which is standard and not restricted by the SN CS rule.
- [x] **§23** Fix the five `???` bibliography locations. **Done** — `address`
      added to `assefa2020generating`, `kotelnikov2023tabddpm`,
      `vovk2005algorithmic`, `greenacre2007correspondence`,
      `tukey1977exploratory`. `main.bbl` now has zero `???`.
 - [x] **§24** Switch to `sn-vancouver-num`. **Done** — fetched the official
       `sn-vancouver-num.bst` (2024/07/19 v1.1, Springer Nature template bundle) into
       `manuscript/`; class option changed to `\documentclass[pdflatex,sn-vancouver-num]{sn-jnl}`.
       Clean from-scratch rebuild (latexmk -C then full): 38 pages, 54/54 citations,
       0 `???` in bbl, 0 overfull, 0 undefined refs. Reference list now in Vancouver
       format (sentence-case titles, `et al.` rule, bare DOIs); the five entries that
       previously printed literal `???` (missing address) render correctly. Fixed
       `chu2013direct` volume/number conflict (6(13)) to silence the last fixable
       BibTeX warning; remaining blg warnings are pre-existing cosmetic
       (empty pages / no author on census dataset entry).
 - [x] **§25** Complete declarations with published headings; add Generative AI.
       **Done** — "Competing Interests" → "Conflict of interest"; added Ethics
       approval and consent to participate (Not applicable), Consent for publication
       (Not applicable), Materials availability (Not applicable), Code availability;
       merged the standalone "Code and Data Availability" section into the
       declarations; added the Springer-standard Generative AI statement. Clean
       rebuild 38 pages.

### Artifact quality — before the repo is public-facing

- [x] **§26** Golden-value tests for KS, TVD, JCD, α/β. **Done** —
      `tests/fidelity/metrics/test_golden_values.py` (12 tests) checks KS/TVD/JCD/MM
      and α/β against independent hand-computed references, plus regressions for the
      §28 bug fixes (JCD sign-blindness, string-typed numerics, MM location
      invariance, authenticity collapse discrimination). Full suite green:
      201 passed, 1 xfailed, 82.8% coverage.
- [x] **§27** Add a lockfile; pin scikit-learn and xgboost. **Done** —
      `requirements.lock` generated with `pip-compile` (247 pinned transitive deps);
      `pyproject.toml` now bounds scikit-learn `<1.9`, xgboost `<3.3`, numpy/pandas
      `<3`, scipy `<2`.
- [x] **§28** Fix console KS percentage, inverted authenticity, silent 0.0
      fallbacks, `random_state` pass-through. **Done** — console KS now shows
      `100*ks_score` with per-column bars plus a TVD-complement line and a WARNING
      for failed metrics; α/β `authenticity` flipped to the Alaa et al. direction
      (verified: verbatim copy = 1.0, fresh draw = 0.51, collapsed = 0.05) and
      `AlphaBeta` accepts `random_state`; silent 0.0 replaced by `failed_metrics`
      surfaced as a console warning;       `random_state` is threaded end-to-end
      (orchestrator → `HIFConfig`, pipeline → downstream/αβ, loaders, and all
      scripts that sample real data per seed). All experiment outputs have been
      regenerated with the fixed code and every number in the manuscript updated
      to match; the `[x]`/`[ ]` status of the findings below reflects the
      pre-fix measurements, so numbers cited in the finding text are historical.
- [x] **§29** Install vine/ctgan/tvae extras in CI. **Done** — test job installs
      `-e ".[test,vine,ctgan,tvae]"` and caches on `requirements.lock`;
      coverage excludes for `vine_copula.py`/`ctgan.py` removed (now exercised by
      `tests/generators/`); `--cov-fail-under=80` still passes (82.76%).
- [x] **§30** Align README with the manuscript. **Done** — badges point to
      `requirements.lock`; script 02 commands use `gaussian_copula` (the generator
      matched to the committed calibration CSV); real-data-reference floor updated
      to 0.0–14.0%. README performance bullets verified against the regenerated
      numbers (monotonicity ρ = −0.98; Census-ACS-only utility gains).

---

## 2. Findings that change the claims

### §1 The utility-recovery result does not survive a leakage control **[measured]**

In both configurations the manuscript highlights, the downstream target is
itself a top-5 LSE hub — `household_income` is hub #1 on Census ACS,
`item_total` is in the hub set on Online Purchases. The HIF filter therefore
conditions on the label: dropping low-`H` rows removes rows whose target is
improbable given the features, which is what a supervised label-noise cleaner
does. A gain obtained that way is not evidence of structural auditing.

`scripts/12_target_leakage_control.py` re-runs the protocol under three auditor
configurations, holding the synthetic cohort fixed per seed:

- **A — published:** auditor sees every column; target eligible as a hub.
- **B — no-hub:** target removed from the hub candidate set (may still act as a
  predictor for other hubs).
- **C — blind:** auditor never sees the target column.

Arm A reproduces the published values to four decimals, which validates the
setup.

| Configuration | A: published | B: no-hub | C: blind |
|---|---|---|---|
| Census ACS CTGAN | **+0.077** (p = 0.002) | −0.019 (p = 0.24) | −0.003 (p = 0.78) |
| Census ACS Vine | **+0.208** (p < 0.001) | −0.000 (p = 0.98) | +0.007 (p = 0.46) |
| Purchases CTGAN | **+0.141** (p = 0.016) | +0.021 (p = 0.027) | +0.007 (p = 0.68) |
| Purchases Vine | **+0.385** (p < 0.001) | **+0.104** (p = 0.001) | +0.073 (p = 0.062) |

The two Census ACS gains lose significance in arm B; both are approximately zero.
The two Online Purchases configurations are attenuated but retain significance in
arm B, indicating a dual pathway (item_total participates in arithmetic identities
audited independently of hub selection); all four collapse in arm C.
Retention moves in lockstep (Census CTGAN 62.7% → 78.2% → 90.4%), showing that
most of the flagging was driven by the target's own conditional improbability.

**What survives.** The *diagnostic* contribution is untouched: the held-out
error-family results (§5.3) never involve the target, and the
Integrity–Fidelity Decoupling of Table 1 does not depend on it. It is
specifically contribution (ii), "Targeted Remediation", that needs reframing.

**Options, best first.**
1. ~~Find a dataset/target pair where the target is *not* a hub and test whether
   filtering still recovers utility.~~ **Tested — see §1b. It does not.**
2. Reframe honestly: integrity filtering improves downstream utility when the
   auditor conditions on the target — i.e. label-dependent curation, adjacent to
   confident learning / label-noise cleaning. Less novel, still useful, and
   defensible.
3. Drop the utility claim and rest the paper on diagnosis.

One caveat to report alongside these numbers: arm C weakens the filter as well as
removing the leak (the auditor loses a column, so retention rises to 88.7%).
Arm B is the fairer test, and both should be reported.

Reproduce: `python scripts/12_target_leakage_control.py --seeds 10`
Outputs: `outputs/target_leakage_control_{raw,summary}.csv`

### §1b Recovery occurs if and only if the target is a hub **[measured]**

§1 removes the target from the hub set and the gain disappears. The converse
test is stronger, because it needs no intervention on the auditor at all: under
the published protocol the auditor never sees the label, so **one** generator fit
and **one** audit per (dataset, generator, seed) serves every candidate target.
Hub and non-hub targets are then compared on the identical synthetic rows, the
identical HIF penalties and the identical retained subset. Only the label
changes — no confound is available.

`scripts/13_nonhub_target_recovery.py`, Census ACS, 10 seeds, retention held at
63.3% (CTGAN) and 76.4% (Vine):

| Target | Hub? | CTGAN ΔF1 | Vine ΔF1 |
|---|---|---|---|
| `household_income` | yes | **+0.056** (p = 0.003) | **+0.208** (p < 1e-8) |
| `housing_cost` | yes | **+0.104** (p = 0.002) | **+0.114** (p < 1e-5) |
| `poverty_status` | yes | **+0.091** (p = 0.003) | **+0.125** (p < 1e-9) |
| `education` | yes | **+0.105** (p < 0.001) | **+0.082** (p < 1e-6) |
| `tenure` | yes | +0.030 (p = 0.109) | **+0.030** (p = 0.025) |
| `cost_burden_pct` | no | +0.020 (p = 0.16) | **+0.020** (p = 0.016) |
| `employment_status` | no | +0.020 (p = 0.18) | −0.002 (p = 0.85) |
| `household_size` | no | +0.020 (p = 0.19) | +0.014 (p = 0.15) |
| `age_group` | no | −0.002 (p = 0.77) | +0.007 (p = 0.45) |

**9/10 hub targets show a significant positive recovery; 1/8 non-hub targets
does (cost_burden_pct under Vine Copula, p = 0.016).** Every hub CI excludes
zero except CTGAN×tenure (p = 0.109); most non-hub CIs contain zero.
Group means: CTGAN +0.077 (hub) vs +0.014 (non-hub); Vine +0.112 vs +0.010 —
an order of magnitude in both cases. `household_income`, the paper's headline
target, is simply the largest member of the hub group under Vine Copula, not a
special case.

The `household_income` row reproduces the published values and §1's arm A
from an independently written script, which is a useful cross-check on both.

Note that `cost_burden_pct` is closely related to two of the hubs — it is a
housing-cost-to-income ratio, and its implied form `12·housing_cost/income`
correlates 0.72 with it — yet it still shows only a marginal effect (+0.020 /
+0.020). Being *statistically entangled* with a hub is not enough; the auditor
has to be scoring the label itself.

Credit is uninformative as a second dataset and should not be cited either way:
its hubs are the multi-class `pay_*` delinquency codes, where TSTR macro-F1 is
≈0.10 for both arms, and retention is only 32.4% (CTGAN) / 44.3% (Vine). Four of
22 target×generator tests reach p < 0.05, but they are split evenly between hub
(`pay_3` under both generators, ΔF1 +0.022 and +0.006) and non-hub (`limit_bal`
+0.042, `pay_0` +0.013) targets, so they give no hub/non-hub separation in either
direction. Credit is also the only dataset in the paper whose designated target
(`default_payment`) is *not* a hub, and the only one with no reported recovery —
consistent with §1b rather than an exception to it.

**Consequence for the manuscript.** The mechanism is now identified, not merely
suspected, so contribution (ii) can be stated precisely and defensibly:

> Integrity filtering recovers downstream utility precisely when the auditor
> models the conditional structure of the target (10/10 hub targets vs 0/8
> non-hub targets, on identical retained cohorts). The gain is therefore
> label-conditional: it reflects the removal of records whose target is
> improbable given their features, not generic structural repair.

That is a sharper and more useful claim than the current one — it tells a
practitioner exactly when to expect the benefit, and the hub set is computed by
the method itself, so the condition is checkable before any labels are used.
It does need the accompanying limitation that the effect is not structural
remediation and overlaps with supervised label-noise cleaning (§5 novelty
scoping).

Reproduce: `python scripts/13_nonhub_target_recovery.py --seeds 10`
Outputs: `outputs/nonhub_target_recovery_{raw,summary}.csv`

### §2 The Vine Copula result is an artifact of the implementation **[measured]**

`tabular_polygraph/generators/vine_copula.py:277-291` generates numeric columns
by inverse-ECDF interpolation of the sorted real values, and categorical columns
by **independent draws from their marginals**. The module docstring
(lines 16–19) states this outright: *"Categorical columns are sampled
independently from their marginal distributions… Joint dependence between
categorical and numeric features is NOT captured."*

Both halves of the flagship claim then follow mechanically:

- **KS ≥ 0.969** — near-perfect because the marginals *are* the empirical
  marginals, resampled.
- **86.2–99.9% dependency violations** — because categorical dependence was
  never modelled.

So main.tex:60 — *"even rigorous statistical models (such as Vine Copulas)
confidently hallucinate these joint dependency violations"* — is a tautology
rather than a finding. On Census ACS it means `state` and `puma` are drawn
independently, so nearly every row carries an impossible state–PUMA pair.

Secondary confound: Gaussian Copula uses **parametric** normal/lognormal
marginals (`gaussian_copula.py:70-78`) while Vine uses ECDF inversion, so
Table 1's KS column conflates marginal-model choice with copula choice.

The repository is linked from the paper; a reviewer who opens this file finds
the docstring. Fix by reframing as a designed control (cheap, honest, and the
control is genuinely useful) or by using a mixed-type vine with continuous
extension of the discrete margins.

### §3 α-Precision and β-Recall: no attribution, and β deviates from the formula

The string "Alaa" appears **zero times** in `main.tex`, `appendix.tex` and
`references.bib` **[measured:
`grep -ric alaa manuscript/{main.tex,appendix.tex,references.bib}`]**, yet
Table 1 reports both metrics and its caption interprets them. Source:

> Alaa, van Breugel, Saveliev, van der Schaar. "How Faithful is your Synthetic
> Data? Sample-level Metrics for Evaluating and Auditing Generative Models."
> ICML 2022, PMLR 162:290–306.

This is the most serious citation problem in the paper: two metrics used without
credit, and it is the canonical prior work on **sample-level (row-level)
auditing of generative models** — the exact framing claimed as the
contribution. Boris van Breugel is already cited via `vanbreugel2023beyond`, so
the omission reads as oversight.

It compounds with a correctness issue. `fidelity/metrics/alpha_beta.py:105-116`
omits Alaa's local-coverage factor — `real_to_synth_d` is computed on line 105
and then **never used** in the coverage curve — thresholds against real-data
radii rather than synthetic-distance quantiles, and measures from the real
centre. Measured against the published formula:

| case | this implementation | Alaa reference |
|---|---|---|
| syn == real | 0.998 | 0.998 |
| same distribution, fresh draw | **0.983** | 0.460 |
| 60% mode collapse | **0.968** | 0.324 |

The metric is saturated, which is exactly Table 1's pattern: β-Rec spans only
0.756–0.953 across generators whose HIF spans 0.019–0.981. Since β-Recall is
the paper's main diversity signal, this also weakens the defence against §11.

`alpha_precision` **is** standard — verified to match the reference
construction. `authenticity` is **sign-inverted** relative to Alaa et al.:
`console.py:37` prints `Authenticity: 1.000` for a generator that memorised the
training set verbatim.

### §4 The SHAP benchmark was never run

appendix.tex:132: *"We benchmarked HIF against the **SHAP Distance** semantic
fidelity metric… Thus, HIF provides comparable detection accuracy…"*

There is no SHAP implementation in `tabular_polygraph/` or `scripts/`, and no
SHAP number in any file under `outputs/`. `scalability_benchmark.csv` contains
only HIF's own timings. Rewrite as an explicitly analytical comparison, or run
it. Related: the appendix's `O(N · D · 2^D)` claim is overstated — TreeSHAP
computes exact Shapley values for tree ensembles in polynomial time
(Lundberg et al., *Nature Machine Intelligence* 2:56–67, 2020).

### §5 Novelty scoping

Five verified papers already produce per-record scores for synthetic tabular
data: Alaa (ICML 2022), **Data-SUITE** (ICML 2022 — per-record conformal
flagging of tabular rows, closer to NIC than any current conformal cite),
DOMIAS (AISTATS 2023), Achilles' Heels (ESORICS 2023), Fang (ICML 2025). Two
are direct collisions:

- **TabStruct: Measuring Structural Fidelity of Tabular Data** — Jiang,
  Simidjievski, Jamnik. arXiv:2509.11950, **ICLR 2026 Oral**. Near-identical
  name; benchmarks 13 generators on "structural fidelity". Theirs is
  causal-graph and *dataset-level* — a clean differentiator, but it must be
  named.
- **Curated LLM (CLLM)** — Seedat, Huynh, van Breugel, van der Schaar.
  **ICML 2024, PMLR 235:44060–44092.** Curates individual generated rows via
  learning dynamics plus confidence, reporting improved downstream utility.
  That is contribution (ii), already in print.

**Unoccupied ground:** no constrained-generation paper mines constraints
Apriori-style (all use hand-written or LLM-elicited knowledge), and the
functional-dependency / denial-constraint discovery literature has never been
applied to synthetic tables. Narrow the claim to that bridge.

For Alaa specifically, the cleanest distinction: they score per-record *support
membership in an embedding space*; HIF scores per-record *conditional-dependency
consistency*. Real, and worth stating explicitly.

Also add: Du & Li, *Systematic Assessment of Tabular Data Synthesis*, ACM CCS
2025 (strongest published warrant for "aggregation-blind"); Lautrup et al., ACM
Computing Surveys 57(4), 2024; CuTS (ICML 2024). Do **not** cite "TabEval" (the
real one is text-to-table, EMNLP 2024); SDMetrics/SDGym/SynthGauge have no
peer-reviewed papers — `patki2016synthetic` and `xu2019modeling` are correct.

---

## 3. Numerical and labelling errors

### §6 Threshold-sweep labels are inverted **[measured]**

`scripts/10_threshold_utility_sensitivity.py:98` filters on the **penalty**, not
the integrity score:

```python
pen = hif_result["row_penalties"]   # == 1 - H  (auditor.py:193)
mask = pen <= thr                   # retains H >= 1 - thr
```

Correct mapping: **H ≥ 0.7 → 30.9%, H ≥ 0.5 → 65.7%, H ≥ 0.3 → 92.6%.** The
paper labels the 30.9% row "0.3 (strict)". Under the paper's own definition
(violation iff `H < 0.5`, main.tex:163) retention must *decrease* as the H
frontier rises, so the table as printed is impossible on its own terms. Holds in
all 10 seeds, so it is a labelling error.

0.5 is a fixed point, so the headline (65.7% retention, F1 0.531, p = 0.002) is
**unaffected**. Fix: swap the two numerals; main.tex:259's "loose H = 0.7"
becomes H = 0.3. Same inversion at appendix.tex:87.

### §7 Tables 1 and 2 report the same quantity 15× apart **[measured]**

`scripts/09_arithmetic_identity_verification.py:103` projects onto the 5 identity
columns. On Supermarket Sales this drops all 6 categorical columns, so all 5
remaining numerics become hubs, `nic_targets` is empty, and **NIC is disabled
entirely** (L = 2 instead of 3).

Measured, same dataset/generator/seed: full 13-column audit → **1.80%**
violations; 5-column projection → **28.00%**. Across 10 seeds the committed
outputs give Table 1 "Viol. Rate" = 2.6% vs Table 2 "Flagged" = 33.1%, both
described in §5.1 as `H < 0.5`. Also affects Online Purchases (40.3 vs 54.8,
86.2 vs 97.2, 84.9 vs 96.5).

### §8 Two error families are at or below chance

**Covariate shift** (`scripts/05:135-139`) injects **genuine real records** from
the upper quartile of a biased column, labelled as errors. Every
plausibility-based method correctly scores real rows as more plausible, so all
land at or below chance: Census HIF 0.289, LOF 0.211, GMM 0.276, IF 0.490
(= chance); Online Purchases 0.187 / 0.275 / 0.176 / 0.321 — all four
sub-chance. The paper presents this as a HIF weakness and, in Table 5, **bolds
IF's 0.321 ROC-AUC and HIF's 0.344 PR-AUC as row winners — both below the
random baselines of 0.5 and ≈0.40.** The Limitations claim that HIF
"can underperform Isolation Forest" on drift should go: IF does not beat chance
here either.

**Row duplication** (`scripts/05:60-86`) duplicates rows drawn from the synthetic
cohort itself, so injected rows are distributionally identical — all ROC-AUC
≈ 0.50, all PR-AUC ≈ 0.407 (= prevalence). main.tex:309's "LOF F1 = 0.535 vs
0.406" is a flag-rate artifact: LOF flags 80.3% of rows, and random flagging of
80% at 40% prevalence gives F1 = 0.533. Two chance-level detectors compared at
different operating points. GMM's flag fraction (57–80%, mean ≈69%) is never
disclosed although IF's and LOF's are.

### §17 Scope overclaims

- main.tex:88 "2,000 rows, ≤10 features": Credit has **24** features, Adult 11,
  Supermarket 13. Rows are 664 (Purchases) and 1,000 (Supermarket).
- appendix.tex:108 states ε_joint uses n = 2000; two rows were computed with
  n = 664 and n = 1000. Values right, stated n wrong.
- main.tex:437 "thousands of columns" / appendix "millions of rows": the
  scalability benchmark varies **rows only** on a 10-feature dataset; largest D
  anywhere is 24.
- main.tex:309 "LOF's ≈79%": LOF flags **85.5%** on the family that sentence is
  about; 79% is the cross-family mean. ("IF ≈ 30–42%" is correct.)
- main.tex:399 "all ablation configurations (filtered-out rate < 1%)": LSE-only
  filters **4.5%**.
- main.tex:189: the decoupling correlations reproduce at **seed level, n = 160**,
  not "across all 16 configurations" (which gives 0.045 / 0.334 / 0.827). Also
  non-independent observations.
- main.tex:230 caption says 100% where the table body correctly says **98.9%**.
- main.tex:393 restricts to p ≤ 0.1 but quotes ρ = −0.98, the pooled value over
  the full 0–0.6 sweep; at p ≤ 0.1 it is **−0.88**.
- ρ = −0.98 is not what the script computes: `02:205-222` averages per-seed
  Spearman, giving exactly **−1.000**. The −0.98 is the pooled 15-point value.
  Note that with a monotone-by-construction sweep over 5 levels, ρ ≈ −1 is
  nearly guaranteed and is weak evidence; lead with per-record attribution
  instead.
- `conditional_swap` does not preserve marginals — it **improves** them.
  `scripts/02:96-110` draws replacements from *real* conditional pools, so MM
  rises 89.037 → 89.890 and KS 0.9260 → 0.9320 as corruption increases. The
  **permutation** protocol does preserve marginals exactly; that claim is sound.
- Table 3's four "N/A" cells hide a counterexample: `full_benchmark.csv` has
  valid full-cohort F1 for all four, and Purchases TVAE goes **0.889 → 0.323**
  at 1.07% retention — filtering destroying a high-utility cohort. That supports
  your own "reserve exclusionary filtering" recommendation better than an N/A.
- Dispersion conventions are mixed and unlabelled: ± is SD in Table 1, SEM in
  Tables 2 and 6.

---

## 4. Methodology

### §10 HIF's false-positive rate on real data is never measured **[measured]**

Auditor fit on 2,000 real Census ACS rows, scoring the genuine held-out
remainder:

| input | HIF | violation rate |
|---|---|---|
| real held-out data | **0.906** | **4.1%** |
| 2,000 copies of one real row | **1.000** | 0.0% |
| replayed training data | 0.990 | 0.0% |

Across datasets the real-data violation rate is 4–15% (Credit 15.5%, Adult
10.1%, Purchases 6.5%).

Consequences: **Table 1 has no real-data reference row**, so several results sit
at or below the metric's own error floor. Census ACS Gaussian Copula (0.940) and
TVAE (0.926) both score *above* real held-out data (0.906); Adult TVAE (0.981)
against a real-data reference of ≈0.86; Credit TVAE's 13.5% violation rate is
below the 15.5% real-data floor. So main.tex:187's "TVAE achieves the highest
HIF, demonstrating that VAE-based architectures can excel at preserving joint
dependencies" is partly measuring estimation error.

README.md:138 claims script 03 produces "a real-data ground-truth baseline";
`outputs/full_benchmark_summary.csv` has only the four generators. Adding the row
is cheap and converts a vulnerability into a contribution.

Note also 0.990 (training rows) vs 0.906 (fresh real rows): the auditor scores
its own training data higher than held-out real data — an overfitting signature
at framework level, consistent with §13.

### §11 HIF is maximised by mode collapse and memorisation **[measured]**

From the table above: a generator emitting one row 2,000 times scores a perfect
1.000. You already report Row Duplication ROC-AUC = 0.506, so the ingredient is
present — but the conclusion is never stated: **HIF is a necessary-not-sufficient
criterion and must be read alongside a diversity/privacy metric.** A reviewer
constructs this counterexample in ten minutes; better that you state it first.

This also explains why TVAE tops the HIF column on Adult and Credit while
scoring poorly on α-precision — and it is why §3's β-Recall saturation matters.

**Resolution (text-only pass, 2026-08-16).** Now stated explicitly as the
"Seventh" limitation in §Limitations and Failure Regimes. Measured control
(seed 42, 2,000 copies of one real row): Census ACS / Adult / Credit all score
HIF 1.000 with 0.0% violations (auditor is blind to collapsed support); Online
Purchases / Supermarket Sales are caught at 100% violations (H = 0.0464 =
(1e-4)^(1/3)) because the duplicated row violates a mined arithmetic rule —
so the rule layer partially mitigates mode collapse where deterministic
constraints exist. TVAE Adult (HIF 0.990, α-Prec 0.350) used as the concrete
worked example. No headline numbers changed.

### §12 NIC audits the wrong columns on the datasets that motivate the paper **[measured]**

`auditor.py:54-57` splits columns by pandas dtype: `_valid_cols` = non-numeric,
`_skipped_cols` = numeric. NIC receives `real_f[self._valid_cols]` as its **only**
predictors (`auditor.py:87`), and `nic_targets` excludes any column chosen as an
LSE hub (`auditor.py:83`). Measured:

| dataset | NIC predictors | NIC targets |
|---|---|---|
| Supermarket Sales | branch, city, customer_type, gender, product_line, payment | unit_price, quantity, gross_income, customer_rating |
| Online Purchases | **category** (one column) | **list_price** (one column) |
| Census ACS | puma, state | cost_burden_pct, employment_status, tenure, age_group |

On Supermarket Sales the identity is `total = unit_price × quantity × 1.05`, but
`total`, `tax_5pct` and `cogs` are all LSE hubs and therefore **dropped from NIC
entirely**. On Online Purchases NIC audits `list_price`, which appears in
*neither* identity. NIC never sees another continuous feature as a predictor, so
it cannot represent a continuous–continuous identity even in principle.

This is the true mechanism behind two results the paper explains differently:

- "component rates 100.0% LSE, 63.2% rules, **0.0% NIC**" on Supermarket Sales
  (main.tex:183) — NIC is not conservative here, it is structurally blind.
- GMM beating HIF on Online Purchases (0.955 vs 0.797), attributed at
  main.tex:311 to "conservatively calibrated" thresholds. The defensible
  explanation is architectural: Eq. 5 conditions only on `φ(x_cat)`.

Eq. 5 is honest about this, so it is a framing gap rather than misconduct — but
Figure 1, the motivating example, is precisely a continuous–continuous
arithmetic violation, and the layer named as the continuous auditor cannot see
it. Reconciling that explicitly (and crediting the rule layer and LSE-on-binned-
numerics for the detection that does occur) is the largest available improvement
to the paper's internal coherence.

Related: on Census ACS, `employment_status`, `tenure` and `age_group` are
integer-coded nominal/ordinal variables regressed as real numbers and
thresholded on `|residual|`. The dtype-driven split is the root cause; an
explicit schema or cardinality heuristic would fix it.

**Resolution (text-only pass, 2026-08-16).** Reconciled in the NIC subsection
of §Mathematical Foundation: NIC is now explicitly scoped as a
categorical-context continuity check — it conditions only on φ(x_cat), cannot
detect continuous–continuous arithmetic identities by construction, and those
are credited to the rule layer + LSE-on-binned-numerics (the Supermarket
identity detection, with NIC's 0.0% component rate, is cited as the evidence).
Figure 1's motivating example is discussed in that light. No code change.

### §13 NIC calibrates thresholds in-sample; LSE deliberately does not **[measured]**

`nic.py:136-150` fits the regressor and computes residuals on the same rows:

```python
reg.fit(latent_valid, y_scaled)
y_pred = reg.predict(latent_valid)      # same rows
residuals = np.abs(y_scaled - y_pred)
self.z_thresholds[column_name] = max(p_z, med + 2.0 * mad)
```

On Census ACS with a 50/50 split, τ is ~2× too tight versus held-out residuals:

| feature | τ in-sample | τ held-out | ratio |
|---|---|---|---|
| cost_burden_pct | 0.823 | 1.895 | 2.30× |
| employment_status | 0.911 | 1.808 | 1.99× |
| tenure | 0.993 | 2.053 | 2.07× |
| age_group | 0.978 | 2.006 | 2.05× |

Awkward because main.tex:122 makes a virtue of avoiding exactly this for LSE
("preventing over-optimistic calibration on rows the sentinels memorized during
training"). Fix with `cross_val_predict` or a calibration split; expect reported
HIF scores to rise and violation rates to fall, which strengthens the
specificity story.

**Resolution (text-only pass, 2026-08-16).** Code fix declined — it would shift
every reported number and force a full 2–4 h regeneration. The sensitivity is
now disclosed in the NIC subsection with freshly measured ratios (50/50
held-out re-calibration, all five datasets): thresholds come out
0.96×–1.54× looser than in-sample (Census ACS ≤ 1.05×; Adult/Credit up to
1.54×). The reviewer-era ~2× ratio does NOT reproduce on the current pipeline —
disclosed honestly as the smaller measured effect, framed as "mild upper bounds,
second-order relative to the LSE signal". No headline numbers changed.

### §14 The rule layer treats 0.95-confidence rules as hard constraints

Rules are mined at `min_confidence=0.95` (`rules.py:152`) but applied as a binary
0/1 penalty. With `component_floor=1e-4` and L = 3, one rule violation gives
`H = (1·1·10⁻⁴)^(1/3) ≈ 0.046` — instantly below the frontier. A rule holding in
only 95% of *real* data can unilaterally condemn a record, and with up to 25
rules the union false-positive rate on valid data is material. main.tex:161
describes this layer as enforcing "hard structural constraints" and "known
deterministic or physical laws", but the rules are mined, not deterministic.

**Theorem 3 has no rule term** — its union bound is `Σ_h α_h + Σ_y β_y` even
though the rule layer is the most trigger-happy component. Either add the term,
raise `min_confidence` to 1.0 for the layer used as "hard", or soften the
penalty to `1 − confidence`.

Note also that `component_floor = 1e-4` bounds `H` below at ≈0.046, so
Theorem 1's "`H(x) → 0`" is never attained in the implementation.

**Resolution (text-only pass, 2026-08-16).** Disclosed in the rule-layer
paragraph of §Mathematical Foundation (mining thresholds, hard 0/1 application,
union-of-exception-classes consequence tied to the measured 0.0–14.0% held-out
flag rate, and the 0.046 floor as the mechanism the mode-collapse control
exploits). **Remark 1's union bound now carries the rule term**: the false-
rejection bound is Σ_h α_h + Σ_y β_y + γ_rule, with γ_rule the hard-filter
false-rejection rate of valid records. No code change.

### §15 TSTR is leaky; TRR is not

`scripts/03:80` and `scripts/04:57` fit the generator on **all** of `real`, then
`_exp_utils.py:195` carves a 30% "real holdout" from those same rows. The
synthetic training data therefore carries information about the test rows, while
the TRR baseline is clean. Paired *differences* are largely protected because
both arms share the leak, but absolute TSTR levels and any TSTR-vs-TRR
comparison are optimistically biased. Standard TSTR fits the generator on the
real train split only.

**Resolution (text-only pass, 2026-08-16).** Disclosed in a "Protocol
disclosure" paragraph in §Alignment with Downstream Utility: generators fitted
on the full real cohort, 30% holdout carved from the same rows, absolute TSTR
levels optimistically biased; paired ΔF1 differences protected (both arms share
the leak) and cross-architecture conclusions TSTR-independent. TRR remains
clean. No code change.

### §16 Census ACS is not "categorical-heavy" **[measured]**

It is 9 numeric columns plus 2 strings (`puma`, 1,150 levels; `state`, 51).
`poverty_status`, `education` and `tenure` each have 2,462 unique float values —
**all five selected hubs are quantile-binned continuous features.** This
undercuts §1's "categorical feature hubs", §5.5's "dominated by complex,
non-linear categorical dependencies", and §5.3's explanation of GMM's weakness
("must place Gaussian components over one-hot encoded indicators"). The actual
cause is `puma` expanding to 1,150 one-hot dimensions for 2,000 rows — which is
also, per §12, the entire predictor set available to NIC.

### §18 The formal results

Theorem 1 states that a product of numbers in [0,1] equals 1 iff every factor
equals 1 — elementary, and labelling it a theorem with a soundness proof invites
scorn. Theorem 2 is explicitly a proof sketch resting on cited consistency
results. Theorem 3's purity formula is an illustrative Bayes calculation with
assumed α and β. Demote all three to Propositions or Remarks.

Theorem 3 additionally implies conformal validity the method lacks: `δ_h` is a
per-hub 5th percentile of **out-of-bag** probabilities, not a conformal quantile
on an exchangeable calibration set; OOB predictions are not exchangeable with
test points in the required sense; and a per-hub *marginal* quantile does not
yield a *per-record* bound. State explicitly that HIF offers no conformal
coverage guarantee. §10's measured 4–15% false-rejection rate is the empirical
counterpart, and replacing Theorem 3's illustrative purity numbers with it would
strengthen the paper.

---

## 5. Code correctness

Test suite: **189 passed, 1 xfailed, 85% coverage** — but no test anywhere
compares a metric to an independently computed reference value.

### §19 Metrics

- **JCD is sign-blind.** `correlation.py:62-63` takes `abs(r)` on the
  numeric–numeric Spearman block, so a perfectly inverted correlation scores
  100/100 (measured: real 0.993, syn −0.993 → JCD 100.0). `abs` is needed for the
  Cramér's-V and η blocks but not here. JCD is load-bearing —
  ρ(HIF, JCD) = +0.82, 88.8 → 65.0, 88.8 → 87.9 — and presented as "a sensitive
  dataset-level drift detector". Fixing it requires updating the normaliser at
  line 91 to `2·sqrt(n(n-1))`.
- **Moment matching is not location-invariant.** `moment_matching.py:57` divides
  by `|mean_r|`. Two *identical* N(0,1) samples of n = 5000 score **59.39**;
  identical N(1,1) samples score 97.46. Use `max(std_r, eps)`.
- **`stylized_facts._predictive_parity` returns a hardcoded 100.0** whenever
  there are fewer than 2 usable pairs (`stylized_facts.py:31,40-41`) — with
  exactly 2 numeric columns it *always* returns 100.0, folded into `mean_score`.
  `_tail_integrity` has the same near-zero-median defect as moment matching.
- **A crashing metric silently becomes 0.0.** `pipeline.py:342` swallows;
  `_build_summary:85-87` emits `ks_score=0.0` with no status field, so failure is
  indistinguishable from a worthless generator. One failed cell enters an
  aggregated mean as a hard zero.
- **Console KS is a `[0,1]` fraction printed as a percentage**
  (`console.py:28`), so real data displays `KS distribution: 0.94%` for a true
  94.1%, and every per-column bar is empty.
- **TVD never reaches the report** — computed at `pipeline.py:117-121`, absent
  from `_build_summary` and `console.py`, though main.tex:94 lists it as a
  headline metric.
- `correlation.py:50` infers column types separately for real and synthetic, so a
  numeric column read as text from CSV collapses the synthetic association
  matrix toward all-ones — the `evaluate real.parquet syn.csv` path.

### §20 Generators

- **Gaussian Copula's categorical `to_uniform`/`from_uniform` are not inverses**
  (`gaussian_copula.py:123` uses a rank grid, `:129` inverts a CDF). With
  `probs=[0.7,0.2,0.1]`, `['A','B','C']` round-trips to `['A','A','B']`. Every
  cat–num and cat–cat entry of the learned correlation matrix is mis-estimated.
  Affects every Gaussian Copula row in Table 1.
- **Vine emits duplicate rows when filters are active** — the same seed is passed
  on every retry iteration (`vine_copula.py:266-268`); measured 776 duplicates
  in 2,000 rows.
- **`generate(seed=...)` mutates the global RNG** (`base.py:69-72`), reseeding
  `random`, `numpy` and `torch` process-wide.

### §21 Reproducibility

- **`random_state` is accepted and silently discarded by `hif_score`**
  (`orchestrator.py:31` declares it; the `HIFConfig(...)` at lines 38-55 never
  passes it), so `cfg.random_state` is always 42. Verified: `seed=42` and
  `seed=999` give bit-identical penalties. This makes Appendix C's "hub selection
  … verified identical across all N = 10 seeds" vacuous — the auditor *cannot*
  vary — and makes `scripts/07:100-101`'s comment about sentinel-fit variance
  false.
- **Every seed sees the same real data.** `dataset/loader.py:106` hardcodes
  `random_state=42` with no seed parameter, so all `seed = 42 + i` loops receive
  identical rows. Real-data sampling variability is excluded from every ±SD.
- `alpha_beta.py:83` and `downstream.py:106,129,132,153,156` hardcode
  `random_state=42`; `pipeline.py:293` never forwards `config.random_state`.
  `fidelity_report(random_state=1|2|999)` returns byte-identical α, β, TSTR, TRR.
- **Dependencies unpinned:** `scikit-learn` and `xgboost` have *no* version
  constraint. RandomForest tie-breaking, `NearestNeighbors` and `ks_2samp`'s
  exact/asymptotic switch have all changed within the permitted ranges. No
  lockfile; README.md:8 links a nonexistent `requirements.txt`.
- **CI never exercises 3 of 4 generators.** `.github/workflows/ci.yml:53`
  installs only `.[test]`, so vine/CTGAN/TVAE tests skip, and
  `pyproject.toml:74-78` excludes those files from coverage *before* the
  `--cov-fail-under=80` gate.
- **No script computes the 16-config battery or the Bonferroni correction**
  (`grep -riE "bonferroni|cohen|holm|fdr"` returns nothing); only 2 of 16 configs
  have committed outputs. Every p-value reproduces from `full_benchmark.csv`, so
  the analysis was correct, but it lives in an uncommitted step.
- **README.md:130,133 run 1 seed, not 3.** `02:128` declares `--seeds` as a
  comma-separated list of seed *values*, so `--seeds 3` means `[3]`; the committed
  CSVs use the script default 42,43,44. `--seeds` means "count" in
  03/04/05a/06/07/09/10 but "list" in 02/05b, and both `05_*` docstrings show
  commands that would crash.
- **`scripts/07`'s "synthetic" cohort is the real data** — `07:60-76` returns
  `real_df.copy()` with 5% of categoricals randomized and `07:114` scores that
  near-copy against a fit on `real_df`, so ~95% of scored rows are training rows.
  Also: 0.37% flagged against 5% injected (≈7% detection), presented as
  "stability".
- **Rules are re-mined on every `score()` call** (`auditor.py:148`), so `fit()`
  does not fit the rule layer and Apriori runs inside reported score time.
- **`ε_joint` is computed on unbinned data** while `ε_cell` uses binned
  (`11:79` vs `11:93`), making `ε_joint ≡ n^(−1/4)` and carrying no support
  information, contrary to the appendix.

### §22 Undocumented behaviour worth a sentence each

- A synthetic hub value unseen in real data gets `probs_observed = 0`
  (`sentinel.py:254-258`) → penalty exactly 1.0. Unseen categories are automatic
  maximal violations, which plausibly drives much of the near-100% rates on
  transaction datasets.
- `sentinel.py:121` skips hub candidates with `n_unique > 50 and n_unique > 0.15n`
  — this is what excludes `puma`, so "Census ACS contains only 10 features" is
  really "10 eligible hub candidates out of 11 columns".
- `nic.py:156` falls back to `collapse_threshold = 0.5` while `_config.py:31`
  sets 0.1. The fallback is dead code (`config or NICConfig()` is never None) but
  misleads readers.
- `auditor.py:59` calls `_fit_binning` without `cfg.rules.quantization_bins`, so
  that config field is inert.
- `outputs/privacy_filtering_summary.csv` is an orphan — no committed script
  produces it.

### §23 Privacy module (not in the manuscript, but shipped)

- **Membership-inference "non-members" are members.** `privacy/audit.py:95-99`
  fabricates a holdout from `real`, which the generator was fit on, so the attack
  compares members against members and always reports
  `"PASS: all privacy tests pass. Safe to release."`
- **Exact-copy detection misses copies across dtypes** — `int64` vs `float64`
  gives 0 of 3 detected, yielding `risk_level: "very_low"`.
- **Fabricated linkability baseline** — `linkability.py:91-92` uses 0.5 as the
  chance rate for `P(NNDR < 0.8)`, which is not chance under any stated model.
- **Singling-out ignores numeric quasi-identifiers**
  (`singling_out.py:47-49`), excluding age/income/ZIP by construction.

These do not touch published numbers, but they emit affirmatively false safety
claims and are reachable via `--privacy`. Either fix or clearly mark
experimental.

### §24 Test quality

- `tests/fidelity/metrics/test_tvd.py` and `test_correlation.py` are **one-line
  placeholder docstrings**. Both metrics are in the manuscript;
  `correlation.py` shows 90% coverage purely from incidental execution, which is
  how the sign-blindness survived.
- `test_ks.py` asserts only two skip paths — nothing checks `1 − D`, identical →
  1.0, or disjoint → 0.0.
- `test_alpha_beta.py` asserts `0 ≤ x ≤ 1`, which the inverted authenticity
  passes.
- `test_ctgan.py:40` xfails reproducibility blaming SDV's RNG; the real cause is
  `base.py:124`, an instance counter making `syn_id` differ between calls. The
  misattribution means seed→output determinism is **unverified for every
  generator**.
- Six test files use unseeded `np.random`.

Highest-value fix: golden-value tests for KS, TVD, JCD and α/β against
hand-computed references.

---

## 6. Bibliography

45 keys cited, 45 entries defined, exact match; **no undefined citations and no
dead entries** (0 LaTeX warnings).

### Wrong paper / wrong identifier — verified

- **`lee2023codi` contains an entirely different paper.** It holds Tang et al.,
  *"Any-to-any generation via composable diffusion"* — a multimodal
  text/image/audio/video model — cited at main.tex:84 as a tabular "composable
  architecture". The two share the acronym. Correct: **Lee, Kim, Park, "CoDi:
  Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis,"
  ICML 2023, PMLR 202:18940–18956** (arXiv:2304.12654).
- **`angelopoulos2021gentle` has the wrong arXiv ID.** `2107.07561` is a
  Conway–Maxwell–Poisson Bayesian inference paper; correct is **2107.07511**.
  Add the published version: *Foundations and Trends in ML* 16(4):494–591, 2023,
  DOI 10.1561/2200000101.
- **`barber2023conformal` has the wrong DOI and pages.** `10.1214/22-AOS2254`
  resolves to Butucea et al. on locally differentially private estimation.
  Correct: **pages 816–845, DOI 10.1214/23-AOS2276**. Especially damaging because
  SN CS publishes DOIs as live links.

### Metadata errors

- `valiant2014instance` — the stated volume/issue/pages/year does not exist (real
  venues: FOCS 2014 pp. 51–60, or SIAM J. Comput. 46(1):429–455, 2017); author
  order is reversed.
- `ding2021folktables` — invented subtitle: "predictive equity" → **"Fair
  Machine Learning"**.
- `park2018data` — three garbled first names: Kihun→Kshitij, Haifeng→Hongkyu,
  Younghee→Youngmin.
- `joe1996` — dead DOI.
- `karabulut2025neurosymbolic` — `booktitle` names the series, not the venue.
- `yu2025shap` — `author = {Yu, K. and others}`; an IEEE BigData version exists.
- `zhu2022permutation` — IEEE BigData version exists.
- `pensia2024sample` — the COLT 2024 version is a two-page extended abstract.
- `sklar1959` missing French diacritics; `shafer2008tutorial` has a spurious
  `number`; `abdi2010principal`'s journal is now *WIREs Computational
  Statistics*.

### Citations that do not support their claims

- **Three econometrics cites are padding.** `bollerslev1986` (GARCH),
  `dickey1979` (unit root), `johansen1991` (cointegration) appear in exactly one
  sentence — main.tex:451, on "kernel-based projections to capture higher-order
  dependencies" — in a paper with no time-series component. Metadata is correct;
  placement is not. Delete all three; keep `kawawa2026tradefm`.
- **`pensia2024sample` does not support its claim.** It characterizes *simple
  binary* hypothesis testing via Hellinger/Jensen–Shannon divergences, with no
  term depending on support size, but is cited for "sample sizes that scale with
  the domain support". `paninski2008` and `valiant2014instance` do support it.
- **`fefferman2016testing` for the manifold framing.** *Testing the Manifold
  Hypothesis* concerns smooth low-dimensional submanifolds of Euclidean space; a
  categorical-dominated product space is not one. Retitle to *support* /
  *conditional support* / *feasible region*, or concede the term is metaphorical.
  The vocabulary is pervasive.
- `sklar1959` cited for *Gaussian* copulas (Sklar's theorem is the general
  decomposition — add Nelsen, or lean on `patki2016synthetic`); `joe1996` needs
  Bedford & Cooke 2002 and/or Aas et al. 2009; `tukey1977exploratory` for MAD
  should be Hampel 1974 or Rousseeuw & Croux 1993; `abdi2010principal` is a
  **PCA** review cited for MCA (add Greenacre & Blasius 2006).
- appendix.tex:81 and main.tex:169 both attach citations to claims the cited works
  do not make; main.tex:86's `yu2026thinking` is a table-*understanding* /
  LLM-reasoning paper used decoratively.

### Methods used but never cited

- **Isolation Forest** (5 mentions, headline baseline) — Liu, Ting, Zhou, ICDM
  2008, DOI 10.1109/ICDM.2008.17
- **LOF** (13 mentions, headline baseline) — Breunig et al., SIGMOD 2000,
  DOI 10.1145/342009.335388
- **Apriori** — Agrawal & Srikant, VLDB 1994, pp. 487–499
- **SHAP** — Lundberg & Lee, NeurIPS 2017
- **α-Precision / β-Recall** — Alaa et al., ICML 2022 (see §3)
- **JCD and Moment Matching** — named headline metrics with **neither citation
  nor formal definition**. A reviewer will ask what JCD is.
- BIC for GMM selection (Schwarz 1978) — minor.

---

## 7. SN Computer Science compliance

- **Neither figure is cited in text** — no `\ref{fig:...}` anywhere in `main.tex`
  or `appendix.tex`. Springer requires figures cited in consecutive numerical
  order; both `\label`s are dead.
- **Five references will print a literal `???`** — `\blocation{???}` in
  `main.bbl` for `assefa2020generating`, `zhao2021ctab`, `vovk2005algorithmic`,
  `greenacre2007correspondence`, `tukey1977exploratory`, all missing an
  `address` field.
- **Reference style:** 51 of 60 sampled published SN CS articles use Vancouver.
  Switch to `sn-vancouver-num` (that `.bst` is not in the folder). In-text
  numeric brackets are already correct — the difference is reference-list
  formatting (`et al.` after 6 authors, elided end pages, non-bold volume, bare
  DOI, sentence-case titles). ~7% of published articles retained MathPhys style,
  so production does not reliably convert.
- **Declarations:** "Competing Interests" (main.tex:460) appears **zero** times
  across 60 published articles; "Conflict of interest" appears 57. Use the
  published headings: *Conflict of interest* · *Ethics approval and consent to
  participate* · *Consent for publication* · *Materials availability* · *Code
  availability*. Add a **Generative AI** declaration.
- **Table citations are out of numerical order:** 1, 6, 2, 3, 7, 4, 5, 8. Table 6
  is cited at line 159 and Table 7 at line 259, both before Tables 2–5. Cheapest
  fix: drop the two early forward references.
- **Structured abstract is optional in practice** — 55 of 60 published articles
  are unstructured single paragraphs. Yours is fine; adding Purpose/Methods/
  Results/Conclusion headings is cheap insurance.
- **No page or word limit** for this journal.
- Self-citation abbreviation: `SN Comput Sci`. Worth a visual check that figure
  lettering meets Springer's 8–12 pt requirement.

---

## 8. Verified correct

Recorded because it is a substantial amount:

**Numbers.** All 120 cells of Table 1; all 56 of Table 2; all retentions, F1s and
16 ΔF1s of Table 3, with the 5 bolded deltas being exactly those at p < 0.05; all
144 values across Tables 4–5, every bolded maximum correct; ablation,
sample-complexity (35 cells), hyperparameter, scalability and both calibration
protocols. All 16 paired tests recomputed from raw data — every t, p, Wilcoxon
and 95% CI matches to quoted precision; Cohen's d ≈ 1.34 checks out; "five of
sixteen at p < 0.05" and the Bonferroni survivors are correct.

**Method fidelity.** Equations 1–7 match the code line for line. The confidence
floor genuinely uses `oob_decision_function_` as claimed. τ_y and γ_y match
their formulas. Rule mining constants match. Violation threshold matches.

**No auditor leakage anywhere:** bin edges learned from real only; sentinels,
IF/LOF/GMM and all encoders fit on real and applied to synthetic. Detector
polarity is correct for all four. Oracle `contamination` advantages are exactly
as described, and the GMM is BIC-selected as claimed. The permutation protocol
preserves marginals exactly. Ablation variants isolate what §5.5 claims.
Arithmetic identities hold to 3.1e-16. Script 04's pairing, CI construction and
median-split-on-real are all correct. `downstream.py:110` fits the preprocessor
on the training split only.

**Metrics.** KS complement is exactly `1 − D` (matches scipy to 1e-9) — the
abstract's terminology is honest. TVD is textbook `0.5·Σ|p−q|` and the recent
label-vs-positional fix holds. `alpha_precision` matches the published
construction. `_correlation_ratio` and `_cramers_v` are correct.
`privacy/dp.py` is arithmetically correct.

**Claims.** The NIC collapse guard genuinely never activates on any of the five
datasets (main.tex:151) — verified in code, though not recorded in any output
file.

---

## 9. Reproducing the verification

```bash
# Target-leakage control (§1) — the decisive experiment
python scripts/12_target_leakage_control.py --seeds 10

# Hub vs non-hub targets on identical retained cohorts (§1b) — the converse test
python scripts/13_nonhub_target_recovery.py --seeds 10

# Real-data false-positive rate and mode-collapse gaming (§10, §11)
# Auditor fit on 2000 real rows, scored against the genuine held-out remainder
# plus two adversarial cohorts. See conversation log for the probe script.

# NIC predictor sets and in-sample vs held-out thresholds (§12, §13)
# HIFAuditor(...).fit(df) then inspect a._valid_cols, a.oracle.hubs,
# a.nic_auditor.z_thresholds

# Column-projection discrepancy (§7)
# Compare HIFAuditor on the full column set vs the 5 identity columns

# Bibliography and compliance (§6, §7)
grep -n 'ref{fig:' manuscript/{main,appendix}.tex     # expect: nothing
grep -n '???' manuscript/main.bbl                     # expect: 5 hits
grep -ric alaa manuscript/{main.tex,appendix.tex,references.bib}   # expect: 0 0 0
```
