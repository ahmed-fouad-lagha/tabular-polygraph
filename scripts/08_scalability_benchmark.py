"""
Experiment: HIF Auditing Scalability Benchmark.

Measures wall-clock time of the HIF scoring (auditing) phase over synthetic
cohorts of increasing size, on a single CPU core. Supports the scalability
claims in the manuscript (e.g., auditing 10k rows in seconds).

The auditor is fitted once on a fixed real cohort, then ``score`` is timed
for each cohort size (the cost that grows with the number of synthetic rows).

Run:
    python scripts/08_scalability_benchmark.py --sizes 10000,100000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ruff: noqa: E402
from _exp_utils import generate, load_real
from tabular_polygraph._config import HIFConfig, RulesConfig
from tabular_polygraph.fidelity.hif.auditor import HIFAuditor

pd.options.mode.copy_on_write = True


def main() -> None:
    parser = argparse.ArgumentParser(description="HIF scalability benchmark")
    parser.add_argument("--dataset", type=str, default="census_acs")
    parser.add_argument("--train-records", type=int, default=2000)
    parser.add_argument("--generator", type=str, default="gaussian")
    parser.add_argument(
        "--sizes", type=str, default="10000,100000", help="Comma-separated cohort sizes"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    real = load_real(args.dataset, n=args.train_records)

    print(f"Fitting {args.generator} on {args.dataset} ({len(real)} train rows)...")
    syn_full = generate(real, max(sizes), args.seed, args.generator)
    print(f"Generated {len(syn_full)} synthetic rows.")

    columns = real.columns.intersection(syn_full.columns).tolist()

    # Fit the HIF auditor (sentinels + rules) once on the real cohort; time
    # only the scoring phase per cohort size, since that cost grows with N.
    config = HIFConfig(
        epochs=10,
        hubs=5,
        depth=12,
        confidence_percentile=5.0,
        violation_threshold=0.5,
        rules=RulesConfig(
            min_confidence=0.95,
            min_support=0.005,
            max_rules=25,
            min_lift=1.0,
            max_antecedents=2,
        ),
    )
    auditor = HIFAuditor(config)
    fit_t0 = time.perf_counter()
    auditor.fit(real, columns=columns)
    fit_time = time.perf_counter() - fit_t0
    print(f"Fitted HIF auditor once ({fit_time:.1f}s).")

    rows = []
    for size in sizes:
        cohort = syn_full.head(size)
        timings = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            auditor.score(cohort)
            timings.append(time.perf_counter() - t0)
        rows.append(
            {
                "dataset": args.dataset,
                "generator": args.generator,
                "train_records": len(real),
                "audit_records": size,
                "fit_seconds": round(fit_time, 3),
                "score_seconds_mean": round(float(pd.Series(timings).mean()), 3),
                "score_seconds_sd": round(float(pd.Series(timings).std(ddof=1)), 3),
                "seconds_per_10k": round(
                    10_000 * float(pd.Series(timings).mean()) / size, 4
                ),
            }
        )
        print(
            f"  size={size:>7d}: score {rows[-1]['score_seconds_mean']:.2f}s "
            f"(+/- {rows[-1]['score_seconds_sd']:.2f}) per run"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scalability_benchmark.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved scalability results to {out_path}")


if __name__ == "__main__":
    main()
