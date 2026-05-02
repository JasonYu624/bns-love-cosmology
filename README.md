# bns-love-cosmology

Bayesian BNS population & parameter-estimation reweighting tools for Cosmic Explorer.

This repository contains scripts, notebooks and helper utilities used to run
binary neutron-star (BNS) parameter-estimation (PE) tasks and population-level
reweighting for Cosmic Explorer studies. Large data products and runtime
outputs are intentionally excluded from the public repository; see **Not
Included** below.

## Contents

- `PE_eosfit_reweight.py` — main reweighting script used to compute reweighted posteriors and summaries.
- `Injection_population_eosfit.py` / `Injection_population.ipynb` — injection / population scripts and examples.
- `hier_eosfit_hyper.py`, `hier_mass_bias*.py` — hierarchical inference helper scripts.
- `*.sh` — job submission / helper bash scripts used to run PE and population workflows.
- `*.ipynb` — notebooks demonstrating plotting and example runs (small data only).
- `working_note_pe_to_hier.md` — notes and operational details about transforming PE outputs into hierarchical inputs.
- `.gitignore` — patterns used to keep large outputs out of the repository.

## Quick start

Clone the repository and create a Python environment (recommended: `venv` or `conda`):

```bash
git clone https://github.com/JasonYu624/bns-love-cosmology.git
cd bns-love-cosmology
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # optional if provided
```

Minimal runtime dependencies (examples):

- Python 3.8+
- numpy, scipy, pandas
- matplotlib
- h5py (for reading bilby results)
- bilby

Install an example set with pip if you need:

```bash
pip install numpy scipy pandas matplotlib h5py bilby
```

## Typical workflow

1. Prepare configuration and/or injection catalogs (outside repository; see `.gitignore`).
2. Run PE scripts (e.g. `PE_eosfit.sh` / `PE_eosfit_reweight.py`) on a compute node or locally.
3. Outputs (results, plots, checkpoint files) are written into `outdir_*` folders and are excluded from git.
4. Use `hier_eosfit_hyper.py` and related utilities to ingest PE results for hierarchical runs.

## Files included vs excluded

- Included: scripts, notebooks, small configuration files, README, LICENSE.
- Excluded (not pushed): `outdir*` directories, large binary result files (`*.npz`, `*.h5`, `*.pkl`), logs (`*.out`, `*.err`, `*.log`). See `.gitignore` for full patterns.

If you need to share specific results, either provide a small example fixture, export a minimal CSV, or provide a download link hosted elsewhere (e.g. Zenodo / Google Drive).

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature`.
3. Add tests or a short example demonstrating the change.
4. Open a pull request describing the change.

## License

This repository is released under the MIT License — see `LICENSE` for details.

## Contact

Maintainer: JasonYu624 (github.com/JasonYu624)

If you'd like, I can also:
- add a `requirements.txt` or `pyproject.toml` with pinned versions,
- add CI checks (GitHub Actions) to run lightweight linting/tests on pushes,
- generate smaller example datasets and a demo notebook for newcomers.
# bns-love-cosmology
