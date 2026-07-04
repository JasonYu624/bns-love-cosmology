#!/usr/bin/env python3
"""HyperPE EOS-fit pipeline with backend-agnostic array operations."""

import argparse
import glob
import json
import math
import os
import random
import re
import time

import bilby as bb
import gwpopulation as gwpop
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.core.utils.random import seed as bilby_seed
from bilby.hyper.model import Model
from gwpopulation.models.redshift import PowerLawRedshift
from wcosmo.astropy import FlatwCDM, Planck18
from wcosmo.utils import disable_units

disable_units()


def set_random_seeds(seed: int | None):
    if seed is None:
        print("Seed: None; stochastic run is not fully reproducible.")
        return

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    bilby_seed(seed)
    print(f"Seed: {seed} (numpy, Python random, bilby/dynesty)")


class SampleH0ConversionModel(Model):
    """Custom model with vectorized sample-level H0 conversion."""

    H0_FID = 70.0
    Z_MIN = 1e-5
    Z_MAX = 1.5
    NZ = 3000

    def __init__(self, model_functions=None, Om0=0.30966, w0=-1.0):
        super().__init__(model_functions=model_functions, cache=False)
        from gwpopulation.utils import xp

        self._cosmo_fid = FlatwCDM(H0=self.H0_FID, Om0=Om0, w0=w0)
        self._z_grid = xp.logspace(xp.log10(self.Z_MIN), xp.log10(self.Z_MAX), self.NZ)
        self._dL_fid_grid = xp.asarray(self._cosmo_fid.luminosity_distance(self._z_grid))
        self.model_data = None
        self.jacobian_det_to_src = None

    def prepare_data(self, data):
        from gwpopulation.utils import xp

        dL = xp.asarray(data["luminosity_distance"])
        H0 = xp.asarray(data["H0"])
        m1_det = xp.asarray(data["mass_1_detector"])
        m2_det = xp.asarray(data["mass_2_detector"])

        dL_scaled = dL * H0 / self.H0_FID
        redshift = xp.interp(dL_scaled, self._dL_fid_grid, self._z_grid)

        mass_1 = m1_det / (1.0 + redshift)
        mass_2 = m2_det / (1.0 + redshift)

        dDL_dz_fid = xp.asarray(self._cosmo_fid.dDLdz(redshift))
        dDL_dz = dDL_dz_fid * self.H0_FID / H0
        jac = 1.0 / ((1.0 + redshift) * (1.0 + redshift) * dDL_dz)

        self.redshift = redshift
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.jacobian_det_to_src = jac
        self.model_data = dict(data)
        self.model_data["redshift"] = self.redshift
        self.model_data["mass_1"] = self.mass_1
        self.model_data["mass_2"] = self.mass_2

    def prob(self, data, **kwargs):
        if self.model_data is None or self.jacobian_det_to_src is None:
            raise RuntimeError("Call prepare_data(data) before prob().")
        prob = super().prob(self.model_data, **kwargs)
        return prob * self.jacobian_det_to_src


def make_fixed_h0_redshift_population(z_max=0.3, cosmo_model="FlatwCDM", H0_fixed=None):
    """Return redshift population with fixed H0 in dVc/dz."""
    base = PowerLawRedshift(z_max=z_max, cosmo_model=cosmo_model)
    h0 = float(Planck18.H0.value if H0_fixed is None else H0_fixed)

    def redshift_population_fixed(dataset, lamb, Om0, w0):
        return base(dataset, lamb=lamb, Om0=Om0, w0=w0, H0=h0)

    return redshift_population_fixed


def ordered_bns_mass_population(dataset, mu_m, sigma_m):
    from gwpopulation.utils import xp

    m1 = dataset["mass_1"]
    m2 = dataset["mass_2"]
    p1 = gwpop.utils.truncnorm(m1, mu=mu_m, sigma=sigma_m, low=1.1, high=1.8)
    p2 = gwpop.utils.truncnorm(m2, mu=mu_m, sigma=sigma_m, low=1.1, high=1.8)
    return 2.0 * p1 * p2 * xp.asarray(m1 >= m2)


def H0_population(dataset, mu_H, sigma_H):
    return gwpop.utils.truncnorm(dataset["H0"], mu=mu_H, sigma=sigma_H, low=10.0, high=150.0)


def eos_population(dataset, mu_a0, sigma_a0, mu_a1, sigma_a1, mu_a2, sigma_a2):
    p0 = gwpop.utils.truncnorm(dataset["delta_a0"], mu=mu_a0, sigma=sigma_a0, low=-5.0, high=5.0)
    p1 = gwpop.utils.truncnorm(dataset["delta_a1"], mu=mu_a1, sigma=sigma_a1, low=-5.0, high=5.0)
    p2 = gwpop.utils.truncnorm(dataset["delta_a2"], mu=mu_a2, sigma=sigma_a2, low=-5.0, high=5.0)
    return p0 * p1 * p2


def get_args():
    p = argparse.ArgumentParser(description="HyperPE EOS-fit pipeline")
    p.add_argument("--posterior-glob", type=str, default="outputs/outdir_population_run/*_reweighted_posterior_augmented.csv")
    p.add_argument("--vt-glob", type=str, default="outputs/outdir_selection_calib_eosfit/worker_*/detected_chunk*.npz")
    p.add_argument("--outdir", type=str, default="outputs/outdir_hyperpe_eosfit")
    p.add_argument("--label", type=str, default="hyperpe_eosfit")
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--seed", type=int, default=12345, help="Seed for numpy, Python random, bilby, and dynesty.")
    p.add_argument("--backend", choices=["jax", "numpy"], default="jax")
    p.add_argument("--fix-lamb-zero", action="store_true")
    p.add_argument("--jit-likelihood", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument(
        "--postprocess-max-samples",
        type=int,
        default=5000,
        help="Maximum posterior samples used only for scatter-matrix plots. Use <=0 to plot all samples.",
    )
    return p.parse_args()

def load_event_posteriors(posterior_glob: str, max_events: int = 0):
    paths = sorted(glob.glob(posterior_glob))
    if not paths:
        raise FileNotFoundError(f"No posterior files matched: {posterior_glob}")

    if max_events is not None and max_events > 0:
        if len(paths) < max_events:
            raise ValueError(
                f"Requested max_events={max_events}, but only found {len(paths)} posterior files."
            )
        paths = paths[:max_events]

    required = [
        "mass_1_detector",
        "mass_2_detector",
        "luminosity_distance",
        "H0",
        "H0_sample",
        "delta_a0",
        "delta_a1",
        "delta_a2",
        "prior",
    ]
    posteriors = []
    for path in paths:
        df = pd.read_csv(path, usecols=required).replace([math.inf, -math.inf], math.nan).dropna()
        df["H0"] = df["H0_sample"]
        df = df.astype(
            {
                "mass_1_detector": float,
                "mass_2_detector": float,
                "luminosity_distance": float,
                "H0": float,
                "H0_sample": float,
                "delta_a0": float,
                "delta_a1": float,
                "delta_a2": float,
                "prior": float,
            }
        )
        posteriors.append(df.reset_index(drop=True))

    return posteriors, paths


def _map_vt_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"m1_det": "mass_1_detector", "m2_det": "mass_2_detector", "dL_mpc": "luminosity_distance"}
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _read_total_generated(summary_path: str) -> int:
    """Read total_generated from one VT summary file."""
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if "total_generated" not in summary:
        raise KeyError(f"{summary_path} does not contain total_generated")
    return int(summary["total_generated"])


def load_vt_injections(vt_glob: str):
    paths = sorted(glob.glob(vt_glob))
    if not paths:
        raise FileNotFoundError(f"No VT files matched: {vt_glob}")

    required = ["mass_1_detector", "mass_2_detector", "luminosity_distance", "H0", "prior"]
    frames = []

    # A worker can write many detected_chunk*.npz files but only one summary.json.
    # Count each unique summary.json once; otherwise total_generated is over-counted
    # by the number of chunks in that worker directory.
    summary_totals: dict[str, int] = {}
    total_generated_naive_per_chunk_sum = 0

    for path in paths:
        with np.load(path) as npz:
            df = pd.DataFrame({k: npz[k] for k in npz.files})
        df = _map_vt_columns(df)

        summary_path = os.path.abspath(os.path.join(os.path.dirname(path), "summary.json"))
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Missing VT summary file for {path}: {summary_path}")
        summary_total = _read_total_generated(summary_path)
        total_generated_naive_per_chunk_sum += summary_total
        summary_totals.setdefault(summary_path, summary_total)

        df = df[required].replace([math.inf, -math.inf], math.nan).dropna()
        df = df.astype(
            {
                "mass_1_detector": float,
                "mass_2_detector": float,
                "luminosity_distance": float,
                "H0": float,
                "prior": float,
            }
        )
        frames.append(df.reset_index(drop=True))

    vt_df = pd.concat(frames, ignore_index=True)
    n_found_after = len(vt_df)
    vt_summary_files = sorted(summary_totals)
    total_generated = int(sum(summary_totals.values()))
    if total_generated <= 0:
        raise ValueError("Corrected VT total_generated is non-positive.")

    if total_generated_naive_per_chunk_sum != total_generated:
        factor = total_generated_naive_per_chunk_sum / total_generated
        print(
            "Corrected VT total_generated over-count: "
            f"naive_per_chunk_sum={total_generated_naive_per_chunk_sum}, "
            f"unique_summary_sum={total_generated}, factor={factor:.6g}",
            flush=True,
        )

    from gwpopulation.utils import xp

    injections = {
        "mass_1_detector": xp.asarray(vt_df["mass_1_detector"].to_numpy(dtype=float)),
        "mass_2_detector": xp.asarray(vt_df["mass_2_detector"].to_numpy(dtype=float)),
        "luminosity_distance": xp.asarray(vt_df["luminosity_distance"].to_numpy(dtype=float)),
        "H0": xp.asarray(vt_df["H0"].to_numpy(dtype=float)),
        "prior": xp.asarray(vt_df["prior"].to_numpy(dtype=float)),
        "total_generated": int(total_generated),
    }
    return (
        injections,
        paths,
        total_generated,
        n_found_after,
        vt_summary_files,
        int(total_generated_naive_per_chunk_sum),
    )


def make_models():
    redshift_population = make_fixed_h0_redshift_population(z_max=0.3, cosmo_model="FlatwCDM")
    event_model = SampleH0ConversionModel(
        [ordered_bns_mass_population, redshift_population, H0_population, eos_population],
        Om0=float(Planck18.Om0),
        w0=-1.0,
    )
    vt_model = SampleH0ConversionModel(
        [ordered_bns_mass_population, redshift_population, H0_population],
        Om0=float(Planck18.Om0),
        w0=-1.0,
    )
    return event_model, vt_model


def make_priors(fix_lamb_zero: bool):
    priors = PriorDict()
    priors["Om0"] = DeltaFunction(peak=float(Planck18.Om0), name="Om0")
    priors["w0"] = DeltaFunction(peak=-1.0, name="w0")
    priors["lamb"] = DeltaFunction(peak=0.0, name="lamb") if fix_lamb_zero else Uniform(-10.0, 10.0, name="lamb")
    priors["mu_m"] = Uniform(0.8, 1.8, name="mu_m")
    priors["sigma_m"] = Uniform(0.005, 0.5, name="sigma_m")
    priors["mu_H"] = Uniform(10.0, 150.0, name="mu_H")
    priors["sigma_H"] = Uniform(0.1, 10.0, name="sigma_H")
    priors["mu_a0"] = Uniform(-5.0, 5.0, name="mu_a0")
    priors["sigma_a0"] = Uniform(0.01, 5.0, name="sigma_a0")
    priors["mu_a1"] = Uniform(-5.0, 5.0, name="mu_a1")
    priors["sigma_a1"] = Uniform(0.01, 5.0, name="sigma_a1")
    priors["mu_a2"] = Uniform(-5.0, 5.0, name="mu_a2")
    priors["sigma_a2"] = Uniform(0.01, 5.0, name="sigma_a2")
    return priors


def write_diagnostics(
    path,
    args,
    posterior_files,
    vt_files,
    posteriors,
    found_vt,
    total_generated,
    vt_summary_files=None,
    total_generated_naive_per_chunk_sum=None,
):
    h0_all = pd.concat([df["H0"] for df in posteriors], ignore_index=True).astype(float)
    prior_all = pd.concat([df["prior"] for df in posteriors], ignore_index=True).astype(float)
    vt_summary_files = list(vt_summary_files or [])
    overcount_factor = math.nan
    if total_generated and total_generated_naive_per_chunk_sum is not None:
        overcount_factor = float(total_generated_naive_per_chunk_sum / total_generated)

    diag = {
        "args": vars(args),
        "posterior_files_used": posterior_files,
        "vt_files_used": vt_files,
        "vt_summary_files_used": vt_summary_files,
        "n_vt_files": int(len(vt_files)),
        "n_vt_summary_files": int(len(vt_summary_files)),
        "n_events": len(posteriors),
        "samples_per_event": [int(len(df)) for df in posteriors],
        "n_found_vt_injections": int(found_vt),
        "total_generated": int(total_generated),
        "total_generated_unique_summary_sum": int(total_generated),
        "total_generated_naive_per_chunk_sum": (
            int(total_generated_naive_per_chunk_sum)
            if total_generated_naive_per_chunk_sum is not None
            else None
        ),
        "total_generated_overcount_factor": overcount_factor,
        "found_fraction": float(found_vt / total_generated) if total_generated > 0 else math.nan,
        "H0_min": float(h0_all.min()),
        "H0_max": float(h0_all.max()),
        "prior_min": float(prior_all.min()),
        "prior_max": float(prior_all.max()),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, sort_keys=True)




KNOWN_HYPERPARAMETERS = [
    "lamb",
    "mu_m",
    "sigma_m",
    "mu_H",
    "sigma_H",
    "mu_a0",
    "sigma_a0",
    "mu_a1",
    "sigma_a1",
    "mu_a2",
    "sigma_a2",
    "Om0",
    "w0",
]


VARIANCE_TARGET_COLUMNS = [
    "log_likelihood",
    "variance",
    "selection_variance",
    "selection_relative_variance",
    "selection_raw_variance",
    "selection_loglike_variance",
    "selection_neff",
    "selection_neff_over_4n",
    "selection_converged",
    "event_variance_sum",
    "selection_variance_fraction",
    "event_variance_fraction",
    "variance_reconstruction_residual",
]


def _as_float_if_scalar(value):
    """Convert numpy/jax scalar-like values to Python floats."""
    try:
        array_value = np.asarray(value)
        if array_value.shape == ():
            return float(array_value)
    except Exception:
        pass
    return value


def _materialize_stats(stats):
    return {key: _as_float_if_scalar(value) for key, value in dict(stats).items()}


def _safe_corr(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    cols = df[[col_a, col_b]].replace([math.inf, -math.inf], math.nan).dropna()
    if len(cols) < 2:
        return math.nan
    if cols[col_a].nunique(dropna=True) < 2 or cols[col_b].nunique(dropna=True) < 2:
        return math.nan
    return float(cols[col_a].corr(cols[col_b]))


def get_hyperparameter_columns(result, full_posterior: pd.DataFrame) -> list[str]:
    """Return hyperparameter columns present in the post-processed posterior."""
    columns = list(full_posterior.columns)
    search_keys = list(getattr(result, "search_parameter_keys", []))
    candidates = search_keys + KNOWN_HYPERPARAMETERS
    hyper_cols = []
    for col in candidates:
        if col in columns and col not in hyper_cols:
            hyper_cols.append(col)
    return hyper_cols


def detect_event_variance_columns(columns, n_events: int) -> list[str]:
    """Detect per-event variance columns from gwpopulation extra statistics.

    gwpopulation versions have used names such as variance_0, var_0, and
    event_variance_0. This function is intentionally conservative: it only
    accepts variance-like column names ending in a valid event index.
    """
    columns = list(columns)
    excluded = {
        "variance",
        "selection_variance",
        "selection_loglike_variance",
        "event_variance_sum",
        "selection_variance_fraction",
        "event_variance_fraction",
        "variance_reconstruction_residual",
    }

    ordered = []
    exact_name_sets = [
        [f"variance_{i}" for i in range(n_events)],
        [f"var_{i}" for i in range(n_events)],
        [f"event_variance_{i}" for i in range(n_events)],
        [f"posterior_variance_{i}" for i in range(n_events)],
        [f"event_{i}_variance" for i in range(n_events)],
    ]
    for names in exact_name_sets:
        present = [name for name in names if name in columns]
        if len(present) == n_events:
            return present
        ordered.extend(present)

    pattern_list = [
        re.compile(r"^(?:var|variance|event_variance|posterior_variance)[_-](\d+)$"),
        re.compile(r"^(?:event|posterior)[_-](\d+)[_-]variance$"),
    ]
    indexed = []
    for col in columns:
        if col in excluded:
            continue
        for pattern in pattern_list:
            match = pattern.match(col)
            if match is None:
                continue
            idx = int(match.group(1))
            if 0 <= idx < n_events:
                indexed.append((idx, col))
            break

    for _, col in sorted(indexed):
        if col not in ordered:
            ordered.append(col)
    return ordered


def add_variance_components(full_posterior: pd.DataFrame, n_events: int) -> tuple[pd.DataFrame, list[str]]:
    """Add event-vs-selection variance decomposition columns.

    selection_variance from gwpopulation is Var(Pdet) / Pdet^2. Since the
    selection term enters the log likelihood as -N_events log(Pdet), its
    contribution to the log-likelihood variance is N_events^2 times this
    stored quantity.
    """
    full_posterior = full_posterior.copy()
    event_variance_cols = detect_event_variance_columns(full_posterior.columns, n_events)

    if event_variance_cols:
        full_posterior["event_variance_sum"] = full_posterior[event_variance_cols].sum(axis=1)
    else:
        full_posterior["event_variance_sum"] = np.nan

    if "selection_variance" in full_posterior.columns:
        # Current gwpopulation.generate_extra_statistics stores selection_variance
        # after dividing the raw Var(Pdet) by selection**2.  It is therefore the
        # relative variance Var(Pdet) / Pdet**2.  The contribution of the
        # selection term -N_events log(Pdet) to the log-likelihood variance is
        # N_events**2 times this relative variance.
        full_posterior["selection_relative_variance"] = full_posterior["selection_variance"]
        full_posterior["selection_loglike_variance"] = (
            float(n_events) ** 2
        ) * full_posterior["selection_relative_variance"]
        full_posterior["selection_neff"] = 1.0 / full_posterior[
            "selection_relative_variance"
        ].replace(0.0, np.nan)
        full_posterior["selection_neff_threshold"] = float(4 * n_events)
        full_posterior["selection_neff_over_4n"] = (
            full_posterior["selection_neff"] / float(4 * n_events)
        )
        full_posterior["selection_converged"] = (
            full_posterior["selection_neff"] > float(4 * n_events)
        ).astype(float)
        if "selection" in full_posterior.columns:
            full_posterior["selection_raw_variance"] = (
                full_posterior["selection_relative_variance"]
                * full_posterior["selection"] ** 2
            )
        else:
            full_posterior["selection_raw_variance"] = np.nan
    else:
        full_posterior["selection_relative_variance"] = np.nan
        full_posterior["selection_loglike_variance"] = np.nan
        full_posterior["selection_neff"] = np.nan
        full_posterior["selection_neff_threshold"] = float(4 * n_events)
        full_posterior["selection_neff_over_4n"] = np.nan
        full_posterior["selection_converged"] = np.nan
        full_posterior["selection_raw_variance"] = np.nan

    if "variance" in full_posterior.columns:
        denominator = full_posterior["variance"].replace(0.0, np.nan)
        full_posterior["event_variance_fraction"] = full_posterior["event_variance_sum"] / denominator
        full_posterior["selection_variance_fraction"] = full_posterior["selection_loglike_variance"] / denominator
        full_posterior["variance_reconstructed"] = (
            full_posterior["event_variance_sum"] + full_posterior["selection_loglike_variance"]
        )
        full_posterior["variance_reconstruction_residual"] = (
            full_posterior["variance"] - full_posterior["variance_reconstructed"]
        )
    else:
        full_posterior["event_variance_fraction"] = np.nan
        full_posterior["selection_variance_fraction"] = np.nan
        full_posterior["variance_reconstructed"] = np.nan
        full_posterior["variance_reconstruction_residual"] = np.nan

    return full_posterior, event_variance_cols


def write_variance_component_summary(full_posterior: pd.DataFrame, path: str):
    cols = [
        "variance",
        "event_variance_sum",
        "selection_loglike_variance",
        "selection_variance",
        "selection_relative_variance",
        "selection_raw_variance",
        "selection_neff",
        "selection_neff_over_4n",
        "selection_converged",
        "event_variance_fraction",
        "selection_variance_fraction",
        "variance_reconstruction_residual",
    ]
    cols = [col for col in cols if col in full_posterior.columns]
    rows = []
    for col in cols:
        series = full_posterior[col].replace([math.inf, -math.inf], math.nan).dropna()
        if len(series) == 0:
            continue
        rows.append(
            {
                "quantity": col,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "q05": float(series.quantile(0.05)),
                "q10": float(series.quantile(0.10)),
                "q90": float(series.quantile(0.90)),
                "q95": float(series.quantile(0.95)),
                "q99": float(series.quantile(0.99)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_parameter_correlation_table(full_posterior: pd.DataFrame, hyper_cols: list[str], path: str):
    target_cols = [col for col in VARIANCE_TARGET_COLUMNS if col in full_posterior.columns]
    rows = []
    for param in hyper_cols:
        row = {"parameter": param}
        for target in target_cols:
            if target == param:
                continue
            row[f"corr_with_{target}"] = _safe_corr(full_posterior, param, target)
        rows.append(row)
    corr_df = pd.DataFrame(rows)
    if "corr_with_variance" in corr_df.columns:
        corr_df["abs_corr_with_variance"] = corr_df["corr_with_variance"].abs()
        corr_df = corr_df.sort_values("abs_corr_with_variance", ascending=False)
    corr_df.to_csv(path, index=False)


def sample_for_plot(full_posterior: pd.DataFrame, max_samples: int, seed: int | None) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(full_posterior) <= max_samples:
        return full_posterior
    return full_posterior.sample(n=max_samples, random_state=seed)


def save_scatter_matrix(df: pd.DataFrame, cols: list[str], path: str, alpha: float = 0.1):
    cols = [col for col in cols if col in df.columns]
    cols = list(dict.fromkeys(cols))
    if len(cols) < 2:
        print(f"Not enough columns found for scatter matrix: {path}")
        return
    plot_df = df[cols].replace([math.inf, -math.inf], math.nan).dropna()
    if len(plot_df) < 2:
        print(f"Not enough finite rows found for scatter matrix: {path}")
        return

    figsize = (max(10, 2.6 * len(cols)), max(10, 2.6 * len(cols)))
    axes = pd.plotting.scatter_matrix(
        plot_df,
        alpha=alpha,
        diagonal="hist",
        figsize=figsize,
    )
    for ax in np.ravel(axes):
        ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"Scatter matrix written to {path}")


def print_variance_summary(full_posterior: pd.DataFrame):
    if "variance" in full_posterior.columns:
        variance = full_posterior["variance"].replace([math.inf, -math.inf], math.nan).dropna()
        if len(variance) > 0:
            print(
                "Variance summary: "
                f"median={variance.median():.6g}, "
                f"90%={variance.quantile(0.90):.6g}, "
                f"95%={variance.quantile(0.95):.6g}, "
                f"99%={variance.quantile(0.99):.6g}, "
                f"max={variance.max():.6g}"
            )

    for col in [
        "event_variance_fraction",
        "selection_variance_fraction",
        "selection_neff",
        "selection_neff_over_4n",
        "selection_converged",
    ]:
        if col in full_posterior.columns:
            series = full_posterior[col].replace([math.inf, -math.inf], math.nan).dropna()
            if len(series) > 0:
                print(
                    f"{col}: "
                    f"median={series.median():.6g}, "
                    f"90%={series.quantile(0.90):.6g}, "
                    f"95%={series.quantile(0.95):.6g}, "
                    f"max={series.max():.6g}"
                )


def write_postprocessing_checks(likelihood, result, args, n_events: int):
    """Generate post-processing checks for Monte Carlo variance diagnostics."""
    print("=== Post-processing checks ===")
    print(f"Detected n_events for variance decomposition: {n_events}")

    stats_func = likelihood.generate_extra_statistics
    if args.backend == "jax":
        try:
            import jax

            stats_func = jax.jit(stats_func)
            print("Using jax.jit for generate_extra_statistics.")
        except Exception as exc:
            print(f"Could not JIT generate_extra_statistics; using plain function. Reason: {exc}")

    posterior_records = result.posterior.to_dict(orient="records")
    full_posterior = pd.DataFrame([_materialize_stats(stats_func(params)) for params in posterior_records]).astype(float)
    full_posterior, event_variance_cols = add_variance_components(full_posterior, n_events=n_events)
    hyper_cols = get_hyperparameter_columns(result, full_posterior)
    sampled_cols = [
        col for col in getattr(result, "search_parameter_keys", [])
        if col in full_posterior.columns
    ]
    if not sampled_cols:
        # Fallback for older bilby result objects.  hyper_cols may include fixed
        # DeltaFunction parameters, but this is preferable to silently skipping
        # the advisor-requested compact scatter matrix.
        sampled_cols = hyper_cols

    full_path = os.path.join(args.outdir, f"{args.label}_full_posterior_extra_statistics.csv")
    desc_path = os.path.join(args.outdir, f"{args.label}_extra_statistics_describe.csv")
    corr_path = os.path.join(args.outdir, f"{args.label}_extra_statistics_corr_all.csv")
    component_summary_path = os.path.join(args.outdir, f"{args.label}_variance_components_summary.csv")
    parameter_corr_path = os.path.join(args.outdir, f"{args.label}_variance_correlations_by_parameter.csv")
    metadata_path = os.path.join(args.outdir, f"{args.label}_variance_decomposition_metadata.json")
    scatter_h0_path = os.path.join(args.outdir, f"scatter_matrix_{args.label}_variance_sigma_H.png")
    scatter_components_path = os.path.join(args.outdir, f"scatter_matrix_{args.label}_variance_components.png")
    scatter_all_path = os.path.join(args.outdir, f"scatter_matrix_{args.label}_all_hyperparams_variance.png")
    scatter_sampled_path = os.path.join(args.outdir, f"scatter_matrix_{args.label}_sampled_params_loglike_variance.png")

    full_posterior.to_csv(full_path, index=False)
    full_posterior.describe().to_csv(desc_path)
    write_variance_component_summary(full_posterior, component_summary_path)
    write_parameter_correlation_table(full_posterior, hyper_cols, parameter_corr_path)

    corr_cols = hyper_cols + [col for col in VARIANCE_TARGET_COLUMNS if col in full_posterior.columns]
    corr_cols += event_variance_cols
    corr_cols = [col for col in dict.fromkeys(corr_cols) if col in full_posterior.columns]
    if len(corr_cols) >= 2:
        full_posterior[corr_cols].corr().to_csv(corr_path)
        print(f"Full correlation matrix written to {corr_path}")
    else:
        print("Not enough columns found for a correlation matrix.")

    metadata = {
        "n_events": int(n_events),
        "event_variance_columns": event_variance_cols,
        "hyperparameter_columns": hyper_cols,
        "sampled_parameter_columns": sampled_cols,
        "selection_variance_column_present": "selection_variance" in full_posterior.columns,
        "notes": [
            "current gwpopulation.generate_extra_statistics stores selection_variance as Var(Pdet) / Pdet**2",
            "selection_relative_variance is an explicit copy of selection_variance for clarity",
            "selection_loglike_variance = n_events**2 * selection_relative_variance",
            "selection_neff = 1 / selection_relative_variance",
            "selection_converged uses the ResamplingVT criterion selection_neff > 4 * n_events",
            "event_variance_sum is the sum of detected per-event variance columns",
            "selection_variance_fraction = selection_loglike_variance / variance",
            "event_variance_fraction = event_variance_sum / variance",
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if not event_variance_cols:
        print("WARNING: no per-event variance columns were detected in generate_extra_statistics output.")
    else:
        print(f"Detected per-event variance columns: {event_variance_cols}")
    if "selection_variance" not in full_posterior.columns:
        print("WARNING: selection_variance was not found in generate_extra_statistics output.")

    plot_df = sample_for_plot(full_posterior, args.postprocess_max_samples, args.seed)
    if len(plot_df) < len(full_posterior):
        print(f"Scatter matrices use {len(plot_df)} randomly selected samples out of {len(full_posterior)}.")

    save_scatter_matrix(
        plot_df,
        ["mu_H", "sigma_H", "log_likelihood", "variance", "selection_loglike_variance", "event_variance_sum"],
        scatter_h0_path,
    )
    save_scatter_matrix(
        plot_df,
        [
            "log_likelihood",
            "variance",
            "selection_variance",
            "selection_relative_variance",
            "selection_raw_variance",
            "selection_loglike_variance",
            "selection_neff",
            "selection_neff_over_4n",
            "event_variance_sum",
            "selection_variance_fraction",
            "event_variance_fraction",
        ],
        scatter_components_path,
    )
    save_scatter_matrix(
        plot_df,
        hyper_cols
        + [
            "log_likelihood",
            "variance",
            "selection_loglike_variance",
            "selection_neff",
            "selection_neff_over_4n",
            "event_variance_sum",
            "selection_variance_fraction",
        ],
        scatter_all_path,
    )
    save_scatter_matrix(
        plot_df,
        sampled_cols + ["log_likelihood", "variance"],
        scatter_sampled_path,
    )

    print_variance_summary(full_posterior)
    print(f"Full extra-statistics posterior written to {full_path}")
    print(f"Extra-statistics summary written to {desc_path}")
    print(f"Variance component summary written to {component_summary_path}")
    print(f"Parameter-variance correlations written to {parameter_corr_path}")
    print(f"Variance decomposition metadata written to {metadata_path}")

def main():
    args = get_args()
    set_random_seeds(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    bb.core.utils.setup_logger(outdir=args.outdir, label=args.label)

    gwpop.set_backend(args.backend)
    print(f"Backend: {args.backend}")
    if args.backend == "jax":
        import jax

        print("JAX devices:", jax.devices())

    posteriors, posterior_files = load_event_posteriors(args.posterior_glob, max_events=args.max_events)
    (
        injections,
        vt_files,
        total_generated,
        n_found_vt,
        vt_summary_files,
        total_generated_naive_per_chunk_sum,
    ) = load_vt_injections(args.vt_glob)

    n_samples_total = sum(len(p) for p in posteriors)
    print(f"Loaded {len(posteriors)} events, total posterior samples = {n_samples_total}")
    print(
        f"Loaded {n_found_vt} found VT injections "
        f"(corrected total_generated = {total_generated}, "
        f"naive per-chunk sum = {total_generated_naive_per_chunk_sum}, "
        f"unique summary files = {len(vt_summary_files)})"
    )

    event_model, vt_model = make_models()
    vt_model.prepare_data(injections)
    vt = gwpop.vt.ResamplingVT(
        model=vt_model,
        data=injections,
        n_events=len(posteriors),
        marginalize_uncertainty=False,
        enforce_convergence=False,
    )
    likelihood = gwpop.hyperpe.HyperparameterLikelihood(
        posteriors=posteriors,
        hyper_prior=event_model,
        selection_function=vt,
    )
    event_model.prepare_data(likelihood.data)
    priors = make_priors(args.fix_lamb_zero)

    print("=== Timing diagnostics ===")
    test_params = {
        "mu_m": 1.33,
        "sigma_m": 0.09,
        "mu_H": float(Planck18.H0.value),
        "sigma_H": 5.0,
        "mu_a0": 0.0,
        "sigma_a0": 1.0,
        "mu_a1": 0.0,
        "sigma_a1": 1.0,
        "mu_a2": 0.0,
        "sigma_a2": 1.0,
        "lamb": 0.0,
        "Om0": float(Planck18.Om0),
        "w0": -1.0,
    }

    t0 = time.perf_counter()
    _ = event_model.prob(likelihood.data, **test_params)
    t_event = time.perf_counter() - t0
    print(f"event_model.prob  : {t_event:.4f} s")

    t0 = time.perf_counter()
    _ = vt_model.prob(injections, **test_params)
    t_vt = time.perf_counter() - t0
    print(f"vt_model.prob     : {t_vt:.4f} s")

    t0 = time.perf_counter()
    _ = likelihood.log_likelihood_ratio(test_params)
    t_like = time.perf_counter() - t0
    print(f"One likelihood    : {t_like:.4f} s")

    mu, var = vt.detection_efficiency(test_params)
    mu = float(mu)
    var = float(var)
    n_eff = float(mu**2 / var) if var > 0 else math.inf
    selection_relative_variance = float(var / mu**2) if mu > 0 and var > 0 else math.inf
    selection_loglike_variance = float(len(posteriors) ** 2 * selection_relative_variance)
    print(f"VT Pdet = {mu:.6g}")
    print(f"VT raw Var(Pdet) = {var:.6g}")
    print(f"VT relative Var(Pdet)/Pdet^2 = {selection_relative_variance:.6g}")
    print(f"VT selection loglike variance = {selection_loglike_variance:.6g}")
    print(f"VT N_eff = {n_eff:.1f}  (threshold 4*N_events = {4 * len(posteriors)})")
    print(f"Estimated time for 10k likelihood evals: {t_like * 10000 / 60:.1f} min")

    diag_path = os.path.join(args.outdir, f"{args.label}_diagnostics.json")
    write_diagnostics(
        diag_path,
        args,
        posterior_files,
        vt_files,
        posteriors,
        n_found_vt,
        total_generated,
        vt_summary_files=vt_summary_files,
        total_generated_naive_per_chunk_sum=total_generated_naive_per_chunk_sum,
    )

    if args.dry_run:
        print("=== Dry run complete, exiting before sampler. ===")
        print(f"Diagnostics written to {diag_path}")
        return

    run_likelihood = likelihood
    if args.backend == "jax" and args.jit_likelihood:
        from gwpopulation.experimental.jax import JittedLikelihood

        parameters = priors.sample()
        _ = likelihood.log_likelihood_ratio(parameters)
        run_likelihood = JittedLikelihood(likelihood)
        _ = run_likelihood.log_likelihood_ratio(parameters)
        print("Using JittedLikelihood.")
    elif args.backend == "jax":
        print("Using non-JIT likelihood on jax backend.")
    else:
        print("Using numpy backend.")

    result = bb.run_sampler(
        likelihood=run_likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=args.nlive,
        naccept=5,
        walks=20,
        check_point_plot=True,
        check_point_delta_t=1800,
        sample="acceptance-walk",
        outdir=args.outdir,
        label=args.label,
        resume=args.resume,
        seed=args.seed,
        print_method="interval-60",
        save="hdf5",
    )
    plot_parameters = [
        "lamb",
        "mu_m",
        "sigma_m",
        "mu_H",
        "sigma_H",
        "mu_a0",
        "sigma_a0",
        "mu_a1",
        "sigma_a1",
        "mu_a2",
        "sigma_a2",
    ]
    plot_parameters = [p for p in plot_parameters if p in result.posterior.columns]

    corner_truth = {
        "lamb": 0.0,
        "mu_m": 1.33,
        "sigma_m": 0.09,
        "mu_H": float(Planck18.H0.value),
        "sigma_H": 0.0,
        "mu_a0": 0.0,
        "sigma_a0": 0.0,
        "mu_a1": 0.0,
        "sigma_a1": 0.0,
        "mu_a2": 0.0,
        "sigma_a2": 0.0,
    }
    truths = [corner_truth.get(p, np.nan) for p in plot_parameters]
    result.plot_corner(
        parameters=plot_parameters,
        save=True,
        filename=os.path.join(args.outdir, f"corner_{args.label}.png"),
        truths=truths,
        truth_color="red",
    )

    write_postprocessing_checks(likelihood, result, args, n_events=len(posteriors))


if __name__ == "__main__":
    main()
