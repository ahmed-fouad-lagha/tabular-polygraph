"""
Example 3: TAMIS Privacy Oracle Walkthrough
--------------------------------------
Demonstrates the TAMIS (Targeted Adversarial Masking and Inference Suite) workflow:
- Membership inference attack
- Singling-out risk
- Linkability risk
- Differential privacy noise addition
- Interpreting TAMIS risk levels and recommendations

Run: python scripts/03_privacy_audit.py
"""

import sys
from pathlib import Path

# ruff: noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


from tabular_polygraph.generators import GaussianCopulaGenerator
from tabular_polygraph.catalog import load_dataset
from tabular_polygraph.privacy import privacy_audit, format_audit
from tabular_polygraph.privacy.dp import PrivacyBudget, laplace_mechanism


def main():
    print("=" * 60)
    print("  TAMIS Privacy Oracle Audit — Census ACS Data")
    print("=" * 60)

    # ── 1. Generate synthetic data ────────────────────────────────────────────
    print("\n[1/4] Generating synthetic census_acs data...")
    seed = load_dataset("census_acs")
    gen = GaussianCopulaGenerator()
    gen.fit(seed)
    syn = gen.generate(1000, seed=42)
    syn_body = syn.drop(columns=["syn_id"])
    print(f"      Real rows: {len(seed):,}  |  Synthetic rows: {len(syn_body):,}")

    # ── 2. Full audit ─────────────────────────────────────────────────────────
    print("\n[2/4] Running full privacy audit (300 attacks per test)...")
    audit = privacy_audit(seed, syn_body, n_attacks=300, seed=42)
    print(format_audit(audit))

    # ── 3. Individual test deep-dives ─────────────────────────────────────────
    print("\n[3/4] Individual test results:")

    # Membership inference
    mi = audit["membership_inference"]
    print("\n  Membership Inference:")
    print(
        f"    Attack AUC     : {mi['attack_auc']}  (0.5 = random, 1.0 = perfect attack)"
    )
    print(f"    Advantage      : {mi['advantage']}  (AUC - 0.5)")
    print(f"    Risk level     : {mi['risk_level']}")
    print(f"    Interpretation : {mi['interpretation']}")

    # Singling-out
    so = audit["singling_out"]
    print("\n  Singling-Out:")
    print(
        f"    Rate           : {so['singling_out_rate']}  (fraction of attacks that uniquely identify)"
    )
    print(f"    Risk level     : {so['risk_level']}")
    print(f"    QI columns used: {so.get('quasi_id_cols', [])}")

    # Linkability
    lk = audit["linkability"]
    print("\n  Linkability:")
    print(f"    Rate           : {lk['linkability_rate']}  (0.5 = random baseline)")
    print(f"    Lift           : {lk['lift_over_baseline_pct']}% over baseline")
    print(f"    Risk level     : {lk['risk_level']}")

    # ── 4. Differential privacy demo ─────────────────────────────────────────
    print("\n[4/4] Differential privacy — protecting aggregate statistics:")
    budget = PrivacyBudget(epsilon=1.0)
    print(f"      Budget: {budget}")

    true_mean_hh_income = float(seed["household_income"].mean())
    true_mean_housing_units = float(seed["total_housing_units"].mean())

    noisy_hh_income = laplace_mechanism(
        true_mean_hh_income,
        sensitivity=500_000,
        epsilon=0.3,
        budget=budget,
        label="mean_household_income",
    )
    noisy_housing_units = laplace_mechanism(
        true_mean_housing_units,
        sensitivity=200_000,
        epsilon=0.3,
        budget=budget,
        label="mean_total_housing_units",
    )

    print(
        f"\n      Household income:  true={true_mean_hh_income:>12,.0f}  noisy={noisy_hh_income:>12,.0f}"
    )
    print(
        f"      Housing units:    true={true_mean_housing_units:>12,.0f}  noisy={noisy_housing_units:>12,.0f}"
    )
    print("\n      Budget log:")
    for entry in budget.log:
        print(f"        ε={entry['epsilon']}  [{entry['label']}]")
    print(f"      Remaining ε: {budget.remaining_epsilon:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
