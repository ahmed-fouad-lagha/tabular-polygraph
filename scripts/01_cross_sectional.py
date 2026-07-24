"""
Example 1: Cross-Sectional Model Training
--------------------------------------
Generate synthetic census-like tabular data and run downstream evaluation.

Run: python scripts/01_cross_sectional.py
"""

from pathlib import Path

# ruff: noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


from tabular_polygraph.dataset import load_dataset
from tabular_polygraph.fidelity import fidelity_report
from tabular_polygraph.generators import GaussianCopulaGenerator


def main():
    print("=" * 60)
    print("  Census ACS — Synthetic Model Training")
    print("=" * 60)

    # ── 1. Load seed and fit generator ────────────────────────────────────────
    print("\n[1/4] Fitting generator on census_acs seed data...")
    seed = load_dataset("census_acs")

    gen = GaussianCopulaGenerator()
    gen.fit(seed)
    print(f"      {gen}")
    print(f"      Marginal kinds: {gen.marginal_kinds}")

    # ── 2. Generate training data ─────────────────────────────────────────────
    print("\n[2/4] Generating 10,000 synthetic training records...")
    train = gen.generate(10_000, seed=42)
    print(f"      Shape: {train.shape}")
    emp_col = "employment_status"
    if emp_col in train.columns:
        print(
            f"      Employment rate: {(train[emp_col].astype(str).str.lower() == 'employed').mean():.2%}"
        )

    # ── 3. Generate evaluation sample ─────────────────────────────────────────
    print("\n[3/4] Generating 2,000 synthetic evaluation records...")
    eval_sample = gen.generate(2_000, seed=99)
    print(f"      Shape: {eval_sample.shape}")

    # ── 4. Full fidelity report ───────────────────────────────────────────────
    print("\n[4/4] Running fidelity report...")
    syn_body = train.drop(columns=["syn_id"])
    report = fidelity_report(
        seed, syn_body, target_col="employment_status", include_downstream=True
    )
    s = report["summary"]
    print(f"      Hybrid Integrity : {s['hybrid_integrity']}%")
    print(f"      Moment matching  : {s['moment_matching_score']}%")
    print(f"      KS distribution  : {s['ks_score']}%")
    print(f"      Joint score      : {s['joint_score']}%")
    if "downstream" in report:
        d = report["downstream"]
        print(
            f"      TSTR score       : {d['tstr_score']}  (TRR: {d['trr_score']})  ratio: {d['ratio']}"
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    from tabular_polygraph.io import write

    write(train, "results/_census_train.csv")
    write(eval_sample, "results/_census_eval.csv")
    print(f"\n  Saved: results/_census_train.csv   ({len(train):,} rows)")
    print(f"         results/_census_eval.csv    ({len(eval_sample):,} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
