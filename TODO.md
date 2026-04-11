# TODO

## Paper Drafting
- [ ] Write abstract (problem, method, key result, contribution)
- [ ] Write introduction (fidelity-validity gap + motivation)
- [ ] Write methods section for LCV (theory + implementation details)
- [ ] Draft experimental protocol (datasets, generators, tasks, thresholds)
- [ ] Draft results section template with placeholders for tables/figures

## Experiments
- [ ] Run multi-dataset benchmark: life_insurance, census_acs, commercial_real_estate
- [ ] Run multi-generator benchmark: GaussianCopula, VineCopula, CTGAN/TVAE
- [ ] Compute and record fidelity vs logical validity for all runs
- [ ] Run downstream tasks on full synthetic vs clean synthetic partitions
- [ ] Quantify F1/accuracy deltas and retention-corruption tradeoff

## Analysis and Ablations
- [ ] Threshold sensitivity study (60/70/75/80/90 percentiles)
- [ ] Ablation: remove bottleneck in autoencoder and compare
- [ ] Ablation: random/noise baseline penalties vs LCV penalties
- [ ] Correlation analysis between fidelity score and logical validity score


## Figures and Tables
- [ ] Figure 1: motivating gap (high fidelity vs logical violations)
- [ ] Figure 2: LCV architecture + CSSP mechanism
- [ ] Figure 3: fidelity vs logical validity scatter across runs
- [ ] Figure 4: downstream performance heatmap (full vs clean synthetic)
- [ ] Figure 5: semantic filtration pipeline diagram
- [ ] Table 1: consolidated benchmark metrics per dataset/generator

## Reproducibility
- [ ] Save all experiment outputs to versioned CSV/JSON files
- [ ] Fix random seeds and log config for every run
- [ ] Add one command/script to reproduce all paper tables
- [ ] Verify environment and dependency versions in README/docs

## Finalization
- [ ] Write discussion (implications, limitations, future work)
- [ ] Write conclusion and key takeaways
- [ ] Perform internal consistency check across text, figures, and numbers
- [ ] Final pass for submission formatting and references
