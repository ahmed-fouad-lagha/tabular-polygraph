# AGENTS.md

## Commands

```bash
# Tests (must all pass before commit)
python -m pytest tests/ -v

# Lint + format + typecheck (pre-commit runs these in order)
python -m ruff check . --fix
python -m ruff format .
python -m mypy tabular_polygraph --ignore-missing-imports

# Single test file
python -m pytest tests/test_fidelity.py -v
```

## Key Facts

- **Python 3.10+** required (type hints use `X | Y` union syntax)
- **Generator deps are optional**: `ctgan`, `torch`, `sdv`, `pyvinecopulib` — core must work without them. Tests for these generators may be skipped.
- **Scripts use `sys.path.insert`**: files in `scripts/` need `# noqa: E402` on imports after the path hack.
- **Datasets cached at** `~/.tabular_polygraph/cache/`. Use `tabular_polygraph download <name>` or `load_dataset()`. Fallback synthetic data in `tests/conftest.py` when download fails.
- **Available datasets**: `adult`, `bls`, `census_acs`, `credit`, `online_purchases`, `supermarket_sales`

## Architecture

- `tabular_polygraph/fidelity/logical.py` — `hif_score()` entry point (the main function)
- `tabular_polygraph/fidelity/report.py` — `fidelity_report()` for full scorecard
- `tabular_polygraph/generators/base.py` — `BaseGenerator` ABC; all generators inherit from it
- `tabular_polygraph/dataset/loader.py` — `load_dataset()`, `DATASETS` registry
- `manuscript/` — LaTeX paper (Springer `sn-jnl.cls`); do not restructure without asking

## Conventions

- Line length 88, ruff handles isort (`tabular_polygraph` is first-party)
- Type hints on all public functions
- No comments unless explaining non-obvious logic
- No emojis in code or output
- Google-style docstrings for public APIs

## Gotchas

- `alpha_precision_beta_recall(real, syn)` — does NOT take a columns argument; third param is `n_steps: int`
- `hif_score()` skips numeric columns for LSE (sentinel ensemble); numeric analysis goes through NIC
- Generator `.generate()` returns a DataFrame with a `syn_id` column — drop it with `errors="ignore"`
- Pre-commit mypy hook runs on the whole `tabular_polygraph` package, not individual files
