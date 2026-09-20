# Output layout

## Held-out error-detection experiments

- heldout_census_acs/ contains the Census ACS benchmark used for the main
  held-out error-detection tables.
- heldout_online_purchases/ contains the separate Online Purchases
  replication used in the appendix.

Each directory contains the same four artifacts:

- heldout_errors_raw.csv: per-seed, per-error-family baseline results.
- heldout_errors_summary.csv: aggregate baseline metrics.
- heldout_errors_matched.csv: fixed and matched-operating-point results.
- heldout_errors_matched_summary.csv: aggregate matched-threshold metrics.

To regenerate a dataset's set, pass its directory to both held-out scripts.
For Census ACS, use:

    python scripts/05_heldout_error_baselines.py --dataset census_acs --output-dir outputs/heldout_census_acs
    python scripts/05_heldout_matched_threshold.py --dataset census_acs --output-dir outputs/heldout_census_acs

Replace census_acs and the output directory with online_purchases and
outputs/heldout_online_purchases for the replication.
