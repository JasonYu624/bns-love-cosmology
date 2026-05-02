# bns-love-cosmology

Bayesian BNS population and parameter-estimation workflows for Cosmic Explorer,
including EOS-fit and Traditional PE pipelines plus posterior reweighting.

## What This Workspace Contains

This workspace is organized around one end-to-end flow:

1. Generate / load injections and detector data products.
2. Run event-level PE with either:
   - EOS-fit (UR parameterization), or
   - Traditional tidal parameterization.
3. Reweight PE posteriors (current recommended mode: `weighted`).
4. Compare priors/posteriors and EOS-fit vs Traditional posteriors in notebooks.

## Main Scripts

- `Injection_population_eosfit.py`, `Injection_population.ipynb`
  Injection/population setup and examples.

- `PE_eosfit_reweight.py`
  EOS-fit PE + reweight pipeline.

- `PE_Traditional.py`
  Traditional PE + reweight pipeline.

- `PE_eosfit_plot.ipynb`
  Plotting/debug notebook for prior/posterior and EOS-fit vs Traditional comparisons.

- `hier_eosfit_hyper.py`, `hier_mass_bias*.py`
  Hierarchical inference utilities.

- `PE_eosfit.sh`, `PE_Tradition.sh`, `PE_reweight_eos.sh`, `*.sh`
  Cluster/job submission helpers.

## Output Directories

- `outdir_population_exactfd`
  Injection-level metadata and exact-signal products.

- `outdir_population_run_test`
  EOS-fit run outputs.

- `outdir_population_run_traditional`
  Traditional run outputs.

Typical per-run products include:

- `*_posterior_augmented.csv`
- `*_reweighted_posterior_augmented.csv`
- `*_reweighted_summary.json`
- `*_all_params_corner.png`
- `*_reweighted_all_params_corner.png`
- `*_result.(hdf5|json|pkl)`

## Reweighting Notes

Current runs are using weighted posterior sampling (`rw_method=weighted`).
This reduces sample count to roughly the effective sample size (`ESS`) and can
introduce repeated rows because weighted resampling is performed with
replacement.

## License

This repository is released under the MIT License. See `LICENSE`.
