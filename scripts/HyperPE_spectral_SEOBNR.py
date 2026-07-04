# gwtc3_cosmo_seob_spectral.py
import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import bilby as bb
import gwpopulation as gwpop
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.core.utils.random import seed as bilby_seed
from gwpopulation.experimental.cosmo_models import CosmoModel
from gwpopulation.experimental.jax import JittedLikelihood
from gwpopulation.models.redshift import PowerLawRedshift
from wcosmo.astropy import Planck18
from wcosmo.utils import disable_units


disable_units()
gwpop.set_backend("jax")

# Edit these paths if needed.
RUNDIR = "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
POSTERIOR_GLOB = os.path.join(
    RUNDIR, "outputs/outdir_population_run_SEOBNR", "*_reweighted_posterior_augmented.csv"
)
VT_GLOB = os.path.join(
    RUNDIR, "outputs/outdir_selection_calib_SEOBNR", "worker_*", "detected_chunk*.npz"
)
OUTDIR = os.path.join(RUNDIR, "outputs/outdir_spectral_siren_seobnr")
LABEL = "spectral_siren_seobnr"
NLIVE = 500
SEED = 12345
PE_DELTA_SIGMA = 1.0 
POSTPROCESS_MAX_SAMPLES = 5000
REDSHIFT_Z_MAX = 0.3


def get_args():
    p = argparse.ArgumentParser(description="Spectral-siren-only HyperPE pipeline")
    p.add_argument("--posterior-glob", type=str, default=POSTERIOR_GLOB)
    p.add_argument("--vt-glob", type=str, default=VT_GLOB)
    p.add_argument("--outdir", type=str, default=OUTDIR)
    p.add_argument("--label", type=str, default=LABEL)
    p.add_argument("--nlive", type=int, default=NLIVE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--redshift-z-max",
        type=float,
        default=REDSHIFT_Z_MAX,
        help="Maximum redshift support for the PowerLawRedshift population model.",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Use only the first N posterior files after sorting. Use <=0 for all events.",
    )
    p.add_argument(
        "--postprocess-max-samples",
        type=int,
        default=POSTPROCESS_MAX_SAMPLES,
        help="Maximum posterior samples used for scatter-matrix plots. Use <=0 for all samples.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume dynesty run if possible.",
    )
    return p.parse_args()

def set_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    bilby_seed(seed)
    print(f"Seed: {seed}")


def ordered_bns_mass_population(dataset, mu_m, sigma_m):
    from gwpopulation.utils import xp

    m1 = dataset["mass_1"]
    m2 = dataset["mass_2"]
    p1 = gwpop.utils.truncnorm(m1, mu=mu_m, sigma=sigma_m, low=1.1, high=1.8)
    p2 = gwpop.utils.truncnorm(m2, mu=mu_m, sigma=sigma_m, low=1.1, high=1.8)
    return 2.0 * p1 * p2 * xp.asarray(m1 >= m2)


def _map_common_columns(df):
    mapping = {
        "m1_det": "mass_1_detector",
        "m2_det": "mass_2_detector",
        "dL_mpc": "luminosity_distance",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _keep_spectral_siren_columns(df, source, remove_eos_prior=False):
    df = _map_common_columns(df)
    required = [
        "mass_1_detector",
        "mass_2_detector",
        "luminosity_distance",
        "prior",
    ]

    if remove_eos_prior:
        required += ["delta_a0", "delta_a1", "delta_a2"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")

    df = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    df = df.astype({c: float for c in required})

    if remove_eos_prior:
        delta_shape = (
            np.exp(-0.5 * (df["delta_a0"] / PE_DELTA_SIGMA) ** 2)
            * np.exp(-0.5 * (df["delta_a1"] / PE_DELTA_SIGMA) ** 2)
            * np.exp(-0.5 * (df["delta_a2"] / PE_DELTA_SIGMA) ** 2)
        )
        if np.any(delta_shape <= 0) or not np.all(np.isfinite(delta_shape)):
            raise ValueError(f"Invalid delta prior factor in {source}")

        # PE_SEOBNR_reweight.py writes prior = dL^2 * delta_shape * support.
        # For spectral-siren-only inference we drop H0_sample and delta_a*, so
        # the recycling prior should be the marginal PE prior for the retained
        # detector-frame masses and luminosity distance. H0_sample was uniform
        # and absent up to a constant; the non-constant delta_a* factor is
        # removed here.
        df["prior"] = df["prior"] / delta_shape
        df = df.drop(columns=["delta_a0", "delta_a1", "delta_a2"])

    df = df[df["prior"] > 0]
    return df.reset_index(drop=True)


def load_seob_posteriors(posterior_glob, max_events=0):
    paths = sorted(glob.glob(posterior_glob))
    if not paths:
        raise FileNotFoundError(f"No posterior files matched: {posterior_glob}")

    if max_events is not None and max_events > 0:
        if len(paths) < max_events:
            raise ValueError(
                f"Requested max_events={max_events}, but only found {len(paths)} posterior files."
            )
        paths = paths[:max_events]

    posteriors = []
    for path in paths:
        df = pd.read_csv(path)
        posteriors.append(_keep_spectral_siren_columns(df, path, remove_eos_prior=True))

    return posteriors, paths


def load_seob_vt_injections(vt_glob):
    paths = sorted(glob.glob(vt_glob))
    if not paths:
        raise FileNotFoundError(f"No VT files matched: {vt_glob}")

    frames = []
    total_generated_naive = 0
    summary_generated = {}

    for path in paths:
        with np.load(path) as npz:
            df = pd.DataFrame({k: npz[k] for k in npz.files})
        frames.append(_keep_spectral_siren_columns(df, path))

        summary_path = os.path.join(os.path.dirname(path), "summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Missing VT summary file for {path}: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            generated = int(json.load(f)["total_generated"])

        # This naive value reproduces the old bug: the same worker-level
        # summary.json is counted once per detected_chunk*.npz file. Keep it
        # only for diagnostics.
        total_generated_naive += generated

        # Correct behavior: each unique summary.json contributes once.
        summary_generated.setdefault(summary_path, generated)

    vt_df = pd.concat(frames, ignore_index=True)
    total_generated = int(sum(summary_generated.values()))
    summary_paths = sorted(summary_generated)

    if total_generated <= 0:
        raise ValueError("Corrected VT total_generated is non-positive.")

    if total_generated_naive != total_generated:
        factor = float(total_generated_naive / total_generated)
        print(
            "Corrected VT total_generated over-count: "
            f"naive_per_chunk_sum={total_generated_naive}, "
            f"unique_summary_sum={total_generated}, factor={factor:.6g}",
            flush=True,
        )

    from gwpopulation.utils import xp

    injections = {
        "mass_1_detector": xp.asarray(vt_df["mass_1_detector"].to_numpy(dtype=float)),
        "mass_2_detector": xp.asarray(vt_df["mass_2_detector"].to_numpy(dtype=float)),
        "luminosity_distance": xp.asarray(vt_df["luminosity_distance"].to_numpy(dtype=float)),
        "prior": xp.asarray(vt_df["prior"].to_numpy(dtype=float)),
        "total_generated": int(total_generated),
    }
    return injections, paths, total_generated, len(vt_df), summary_paths, int(total_generated_naive)



def _finite_numeric_series(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _write_quantile_summary(df, columns, path):
    rows = []
    for col in columns:
        s = _finite_numeric_series(df, col)
        if len(s) == 0:
            continue
        rows.append(
            {
                "quantity": col,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "q05": float(s.quantile(0.05)),
                "q10": float(s.quantile(0.10)),
                "q90": float(s.quantile(0.90)),
                "q95": float(s.quantile(0.95)),
                "q99": float(s.quantile(0.99)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_scatter_matrix(df, columns, path, max_samples=POSTPROCESS_MAX_SAMPLES, seed=SEED):
    columns = [c for c in columns if c in df.columns]
    if len(columns) < 2:
        print(f"Skipping scatter matrix {path}: fewer than two available columns.")
        return

    plot_df = df[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(plot_df) == 0:
        print(f"Skipping scatter matrix {path}: no finite rows.")
        return

    kept_columns = []
    for col in plot_df.columns:
        values = plot_df[col].to_numpy(dtype=float)
        spread = np.nanmax(values) - np.nanmin(values)
        scale = max(1.0, float(np.nanmax(np.abs(values))))
        if np.isfinite(spread) and spread > 1e-12 * scale:
            kept_columns.append(col)
    plot_df = plot_df[kept_columns]
    if len(kept_columns) < 2:
        print(f"Skipping scatter matrix {path}: fewer than two non-constant columns.")
        return

    if max_samples and max_samples > 0 and len(plot_df) > max_samples:
        plot_df = plot_df.sample(n=max_samples, random_state=seed)

    figsize = max(8.0, 2.15 * len(kept_columns))
    try:
        axes = pd.plotting.scatter_matrix(
            plot_df,
            alpha=0.1,
            diagonal="hist",
            figsize=(figsize, figsize),
        )
    except ValueError as exc:
        print(f"Skipping scatter matrix {path}: {exc}")
        plt.close("all")
        return
    for ax in np.ravel(axes):
        ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print("Wrote:", path)


def _add_variance_decomposition(full_posterior, n_events):
    df = full_posterior.copy()
    var_cols = sorted(
        [c for c in df.columns if c.startswith("var_") and c.split("var_", 1)[1].isdigit()],
        key=lambda x: int(x.split("var_", 1)[1]),
    )

    if var_cols:
        df["event_variance_sum"] = df[var_cols].sum(axis=1)
    else:
        df["event_variance_sum"] = np.nan

    if "selection_variance" in df.columns:
        # gwpopulation.generate_extra_statistics stores selection_variance as
        # Var(Pdet) / Pdet**2. The contribution to the log-likelihood variance
        # is N_events**2 times this relative variance because the selection
        # term is -N_events * log(Pdet).
        df["selection_relative_variance"] = df["selection_variance"]
        df["selection_loglike_variance"] = float(n_events) ** 2 * df["selection_relative_variance"]

        positive = df["selection_relative_variance"].replace(0.0, np.nan)
        df["selection_neff"] = 1.0 / positive
        df["selection_neff_threshold"] = float(4 * n_events)
        df["selection_neff_over_4n"] = df["selection_neff"] / df["selection_neff_threshold"]
        df["selection_converged"] = (df["selection_neff"] > df["selection_neff_threshold"]).astype(float)

        if "selection" in df.columns:
            df["selection_raw_variance"] = df["selection_relative_variance"] * df["selection"] ** 2
        else:
            df["selection_raw_variance"] = np.nan
    else:
        df["selection_relative_variance"] = np.nan
        df["selection_loglike_variance"] = np.nan
        df["selection_neff"] = np.nan
        df["selection_neff_threshold"] = float(4 * n_events)
        df["selection_neff_over_4n"] = np.nan
        df["selection_converged"] = np.nan
        df["selection_raw_variance"] = np.nan

    if "variance" in df.columns:
        denom = df["variance"].replace(0.0, np.nan)
        df["event_variance_fraction"] = df["event_variance_sum"] / denom
        df["selection_variance_fraction"] = df["selection_loglike_variance"] / denom
        df["variance_reconstruction_residual"] = (
            df["variance"] - df["event_variance_sum"] - df["selection_loglike_variance"]
        )
    else:
        df["event_variance_fraction"] = np.nan
        df["selection_variance_fraction"] = np.nan
        df["variance_reconstruction_residual"] = np.nan

    return df, var_cols


def write_postprocessing_checks(
    likelihood,
    result,
    outdir,
    label,
    n_events,
    posterior_files,
    vt_files,
    vt_summary_files,
    total_generated,
    total_generated_naive,
    n_found_vt,
    postprocess_max_samples,
    seed,
    redshift_z_max,
):
    print("=== Post-processing variance checks ===")
    func = jax.jit(likelihood.generate_extra_statistics)
    full_posterior = pd.DataFrame(
        [func(p) for p in result.posterior.to_dict(orient="records")]
    ).astype(float)
    full_posterior, var_cols = _add_variance_decomposition(full_posterior, n_events)

    posterior_path = os.path.join(outdir, f"{label}_posterior_with_variance.csv")
    full_extra_path = os.path.join(outdir, f"{label}_full_posterior_extra_statistics.csv")
    describe_path = os.path.join(outdir, f"{label}_extra_statistics_describe.csv")
    corr_all_path = os.path.join(outdir, f"{label}_extra_statistics_corr_all.csv")
    variance_summary_path = os.path.join(outdir, f"{label}_variance_components_summary.csv")
    corr_by_param_path = os.path.join(outdir, f"{label}_variance_correlations_by_parameter.csv")
    metadata_path = os.path.join(outdir, f"{label}_variance_decomposition_metadata.json")

    # Keep the original spectral-siren output name and also write the full HyperPE-style name.
    full_posterior.to_csv(posterior_path, index=False)
    full_posterior.to_csv(full_extra_path, index=False)
    full_posterior.describe().to_csv(describe_path)

    numeric_cols = full_posterior.select_dtypes(include=[np.number]).columns.tolist()
    full_posterior[numeric_cols].corr().to_csv(corr_all_path)

    variance_cols = [
        "variance",
        "event_variance_sum",
        "selection_loglike_variance",
        "selection_relative_variance",
        "selection_raw_variance",
        "selection_variance",
        "selection_neff",
        "selection_neff_over_4n",
        "event_variance_fraction",
        "selection_variance_fraction",
        "variance_reconstruction_residual",
    ]
    _write_quantile_summary(full_posterior, variance_cols, variance_summary_path)

    hyper_cols = ["H0", "mu_m", "sigma_m", "lamb"]
    sampled_cols = [
        col for col in getattr(result, "search_parameter_keys", [])
        if col in full_posterior.columns
    ]
    if not sampled_cols:
        # Fallback for older bilby result objects.
        sampled_cols = [col for col in hyper_cols if col in full_posterior.columns]

    target_cols = [
        "log_likelihood",
        "variance",
        "event_variance_sum",
        "selection_loglike_variance",
        "selection_variance",
        "selection_relative_variance",
        "selection_raw_variance",
        "selection_neff",
        "selection_neff_over_4n",
        "event_variance_fraction",
        "selection_variance_fraction",
        "variance_reconstruction_residual",
    ]
    available_hyper = [c for c in hyper_cols if c in full_posterior.columns]
    available_targets = [c for c in target_cols if c in full_posterior.columns]

    rows = []
    corr_df = full_posterior[available_hyper + available_targets].corr()
    for param in available_hyper:
        row = {"parameter": param}
        for target in available_targets:
            row[f"corr_with_{target}"] = float(corr_df.loc[param, target])
        rows.append(row)
    pd.DataFrame(rows).to_csv(corr_by_param_path, index=False)

    scatter_basic_path = os.path.join(outdir, f"scatter_matrix_{label}_H0_variance.png")
    scatter_components_path = os.path.join(outdir, f"scatter_matrix_{label}_variance_components.png")
    scatter_all_path = os.path.join(outdir, f"scatter_matrix_{label}_all_hyperparams_variance.png")
    scatter_sampled_path = os.path.join(outdir, f"scatter_matrix_{label}_sampled_params_loglike_variance.png")

    _plot_scatter_matrix(
        full_posterior,
        ["H0", "mu_m", "sigma_m", "lamb", "log_likelihood", "variance"],
        scatter_basic_path,
        max_samples=postprocess_max_samples,
        seed=seed,
    )
    _plot_scatter_matrix(
        full_posterior,
        [
            "log_likelihood",
            "variance",
            "event_variance_sum",
            "selection_loglike_variance",
            "selection_relative_variance",
            "selection_neff_over_4n",
            "selection_variance_fraction",
            "event_variance_fraction",
        ],
        scatter_components_path,
        max_samples=postprocess_max_samples,
        seed=seed,
    )
    _plot_scatter_matrix(
        full_posterior,
        [
            "H0",
            "mu_m",
            "sigma_m",
            "lamb",
            "log_likelihood",
            "variance",
            "event_variance_sum",
            "selection_loglike_variance",
            "selection_relative_variance",
            "selection_neff_over_4n",
            "selection_variance_fraction",
        ],
        scatter_all_path,
        max_samples=postprocess_max_samples,
        seed=seed,
    )
    _plot_scatter_matrix(
        full_posterior,
        sampled_cols + ["log_likelihood", "variance"],
        scatter_sampled_path,
        max_samples=postprocess_max_samples,
        seed=seed,
    )

    overcount_factor = (
        float(total_generated_naive / total_generated)
        if total_generated and total_generated > 0
        else np.nan
    )
    metadata = {
        "n_events": int(n_events),
        "n_posterior_samples_hyper": int(len(full_posterior)),
        "var_columns": var_cols,
        "sampled_parameter_columns": sampled_cols,
        "posterior_files_used": list(posterior_files),
        "vt_files_used": list(vt_files),
        "vt_summary_files_used": list(vt_summary_files),
        "n_vt_files": int(len(vt_files)),
        "n_vt_summary_files": int(len(vt_summary_files)),
        "n_found_vt_injections": int(n_found_vt),
        "total_generated_unique_summary_sum": int(total_generated),
        "total_generated_naive_per_chunk_sum": int(total_generated_naive),
        "total_generated_overcount_factor": overcount_factor,
        "found_fraction": float(n_found_vt / total_generated) if total_generated > 0 else np.nan,
        "selection_variance_scaling": "selection_loglike_variance = n_events**2 * selection_relative_variance",
        "selection_relative_variance_note": "gwpopulation extra-statistics selection_variance = Var(Pdet) / Pdet**2",
        "postprocess_max_samples_for_scatter": int(postprocess_max_samples),
        "redshift_z_max": float(redshift_z_max),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if "variance" in full_posterior.columns:
        variance = _finite_numeric_series(full_posterior, "variance")
        if len(variance) > 0:
            print(
                "Variance summary: "
                f"median={variance.median():.6g}, "
                f"90%={variance.quantile(0.90):.6g}, "
                f"95%={variance.quantile(0.95):.6g}, "
                f"99%={variance.quantile(0.99):.6g}, "
                f"max={variance.max():.6g}"
            )

    print("Wrote:", posterior_path)
    print("Wrote:", full_extra_path)
    print("Wrote:", describe_path)
    print("Wrote:", corr_all_path)
    print("Wrote:", variance_summary_path)
    print("Wrote:", corr_by_param_path)
    print("Wrote:", metadata_path)


def main():
    args = get_args()
    set_random_seeds(args.seed)
    print("JAX devices:", jax.devices())
    print(f"POSTERIOR_GLOB={args.posterior_glob}")
    print(f"VT_GLOB={args.vt_glob}")
    print(f"OUTDIR={args.outdir}")
    print(f"LABEL={args.label}")
    print(f"NLIVE={args.nlive}")
    print(f"SEED={args.seed}")
    print(f"REDSHIFT_Z_MAX={args.redshift_z_max}")
    print(f"MAX_EVENTS={args.max_events}")
    print(f"POSTPROCESS_MAX_SAMPLES={args.postprocess_max_samples}")
    print(f"RESUME={args.resume}")

    # -------------------------
    # Load posteriors (SEOB samples)
    # -------------------------
    posteriors, posterior_files = load_seob_posteriors(
        args.posterior_glob,
        max_events=args.max_events,
    )
    print(f"Loaded {len(posteriors)} posterior files")
    print(f"Total posterior samples: {sum(len(p) for p in posteriors)}")
    for path, post in zip(posterior_files, posteriors):
        print(f"  {os.path.basename(path)}: {len(post)} samples")

    # -------------------------
    # Load injections (sensitivity)
    # -------------------------
    (
        injections,
        vt_files,
        total_generated,
        n_found_vt,
        vt_summary_files,
        total_generated_naive,
    ) = load_seob_vt_injections(args.vt_glob)
    print(f"Loaded {len(vt_files)} VT files")
    print(f"Loaded {len(vt_summary_files)} unique VT summary files")
    print(f"Found VT injections: {n_found_vt}")
    print(f"Corrected total generated VT injections: {total_generated}")
    print(f"Naive per-chunk total generated VT injections: {total_generated_naive}")
    if total_generated > 0:
        print(f"VT found fraction: {n_found_vt / total_generated:.8g}")

    # -------------------------
    # Model + likelihood
    # -------------------------
    model = CosmoModel(
        model_functions=[
            ordered_bns_mass_population,
            PowerLawRedshift(z_max=args.redshift_z_max, cosmo_model="FlatwCDM"),
        ],
        cosmo_model="FlatwCDM",
    )

    vt = gwpop.vt.ResamplingVT(
        model=model,
        data=injections,
        n_events=len(posteriors),
        enforce_convergence=False,
    )
    likelihood = gwpop.hyperpe.HyperparameterLikelihood(
        posteriors=posteriors,
        hyper_prior=model,
        selection_function=vt,
    )

    # -------------------------
    # Priors
    # -------------------------
    priors = PriorDict()
    priors["mu_m"] = Uniform(minimum=0.8, maximum=1.8, latex_label=r"$\mu_m$")
    priors["sigma_m"] = Uniform(minimum=0.005, maximum=0.5, latex_label=r"$\sigma_m$")
    priors["H0"] = Uniform(minimum=10, maximum=150, latex_label=r"$H_0$")
    priors["Om0"] = DeltaFunction(peak=float(Planck18.Om0), name="Om0")
    priors["w0"] = DeltaFunction(peak=-1.0, name="w0")
    priors["lamb"] = Uniform(minimum=-10, maximum=10, latex_label=r"$\lambda_z$")

    # -------------------------
    # JIT compile
    # -------------------------
    parameters = priors.sample()
    _ = likelihood.log_likelihood_ratio(parameters)  # warmup
    jit_likelihood = JittedLikelihood(likelihood)
    _ = jit_likelihood.log_likelihood_ratio(parameters)  # compile

    # -------------------------
    # Run dynesty via bilby
    # -------------------------
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    result = bb.run_sampler(
        likelihood=jit_likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=args.nlive,
        label=args.label,
        sample="acceptance-walk",
        naccept=5,
        walks=20,
        print_method="interval-60",
        save="hdf5",
        resume=args.resume,
        outdir=outdir,
        seed=args.seed,
    )

    # -------------------------
    # Plot corner (save to file)
    # -------------------------
    fig = result.plot_corner(
        save=False,
        parameters=["H0", "mu_m", "sigma_m", "lamb"],
        truths=[float(Planck18.H0.value), 1.33, 0.09, 0.0],
    )
    fig.savefig(os.path.join(outdir, f"corner_{args.label}.png"), dpi=200, bbox_inches="tight")

    # -------------------------
    # Post-processing variance checks
    # -------------------------
    write_postprocessing_checks(
        likelihood=likelihood,
        result=result,
        outdir=outdir,
        label=args.label,
        n_events=len(posteriors),
        posterior_files=posterior_files,
        vt_files=vt_files,
        vt_summary_files=vt_summary_files,
        total_generated=total_generated,
        total_generated_naive=total_generated_naive,
        n_found_vt=n_found_vt,
        postprocess_max_samples=args.postprocess_max_samples,
        seed=args.seed,
        redshift_z_max=args.redshift_z_max,
    )

    print("Wrote:", os.path.join(outdir, f"corner_{args.label}.png"))


if __name__ == "__main__":
    main()
