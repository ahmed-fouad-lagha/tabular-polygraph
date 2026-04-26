# Examples and Validation Scripts

| File | Dataset | Topics |
|---|---|---|
| `01_cross_sectional.py` | `census_acs` | Generator fitting, priors, TSTR downstream evaluation, privacy audit |
| `02_macro_scenarios.py` | `fred_macro` | VAR generator, all 5 scenarios, temporal fidelity report |
| `03_privacy_audit.py` | `census_acs` | Full privacy audit, membership inference, singling-out, linkability, differential privacy budget |
| `04_hif_validation.py` | `census_acs` | Full HIF validation suite: Monotonicity, External Validity, Seed Stability, Separability |
| `05_cross_domain_audit.py` | `census_acs`, `world_bank` | Cross-Architecture audit for manuscript Table 2 & 3 (Gaussian, Vine, CTGAN) |
| `06_sensitivity_benchmark.py` | `census_acs` | 'Logical Virus' sensitivity benchmark vs distributional metrics |
