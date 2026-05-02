#!/usr/bin/env python3
import glob
import json
import os

import bilby as bb
import gwpopulation as gwpop
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from bilby.core.prior import Gaussian, PriorDict, Uniform
from gwpopulation.experimental.jax import JittedLikelihood
from jax.scipy.special import erf
from wcosmo import wcosmo as wc
from wcosmo.astropy import Planck18
from wcosmo.utils import disable_units

jax.config.update("jax_enable_x64", True)

disable_units()
gwpop.set_backend("jax")
xp = gwpop.utils.xp

# =========================================================
# Paths
# =========================================================
PE_POST_GLOB = os.environ.get(
    "PE_POST_GLOB",
    "outdir_population_run_test/*_reweighted_posterior_augmented.csv",
)
SEL_MERGED_NPZ = os.environ.get(
    "SEL_MERGED_NPZ",
    "outdir_selection_calib_eosfit/merged_detected.npz",
)
SEL_MERGED_SUMMARY = os.environ.get(
    "SEL_MERGED_SUMMARY",
    "outdir_selection_calib_eosfit/merged_summary.json",
)
OUTDIR = os.environ.get("OUTDIR", "outdir_hier_eosfit_hyperdist_gpu")
LABEL = os.environ.get("LABEL", "hier_eosfit_hyperdist_gpu")

# =========================================================
# Cosmology / population setup
# =========================================================
OM0 = float(os.environ.get("OM0", str(float(Planck18.Om0))))
W0 = float(os.environ.get("W0", "-1.0"))
H0_POP = float(os.environ.get("H0_POP", str(float(Planck18.H0.value))))
Z_MAX = float(os.environ.get("Z_MAX", "0.3"))
GAMMA = float(os.environ.get("GAMMA", "0.0"))
Z_GRID_SIZE = int(os.environ.get("Z_GRID_SIZE", "4096"))

# Population mass model in source frame
M_POP_MIN = float(os.environ.get("M_POP_MIN", "1.1"))
M_POP_MAX = float(os.environ.get("M_POP_MAX", "2.25"))
SIG_M_MIN = float(os.environ.get("SIG_M_MIN", "0.01"))
SIG_M_MAX = float(os.environ.get("SIG_M_MAX", "0.5"))

# Event-level H0 support / hierarchical priors
H0_MIN = float(os.environ.get("H0_MIN", "10.0"))
H0_MAX = float(os.environ.get("H0_MAX", "150.0"))
SIG_H_MIN = float(os.environ.get("SIG_H_MIN", "0.1"))
SIG_H_MAX = float(os.environ.get("SIG_H_MAX", "20.0"))

# delta_a mean / spread hyper-priors
DELTA_MU_SIGMA = float(os.environ.get("DELTA_MU_SIGMA", "2.0"))
SIG_DA_MIN = float(os.environ.get("SIG_DA_MIN", "0.01"))
SIG_DA_MAX = float(os.environ.get("SIG_DA_MAX", "2.0"))

# Selection proposal used in population_selection_bias_eosfit.py
SEL_MASS_DIST = os.environ.get("SEL_MASS_DIST", "flat").lower()
M_SEL_MIN = float(os.environ.get("M_SEL_MIN", "1.0"))
M_SEL_MAX = float(os.environ.get("M_SEL_MAX", "1.7"))
M_SEL_MU = float(os.environ.get("M_SEL_MU", "1.33"))
M_SEL_SIGMA = float(os.environ.get("M_SEL_SIGMA", "0.09"))
SEL_Z_MAX = float(os.environ.get("SEL_Z_MAX", str(Z_MAX)))
SEL_GAMMA = float(os.environ.get("SEL_GAMMA", str(GAMMA)))
SEL_H0 = float(os.environ.get("SEL_H0", str(H0_POP)))

# Runtime
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "2000"))
RNG_SEED = int(os.environ.get("RNG_SEED", "1234"))
NLIVE = int(os.environ.get("NLIVE", "400"))
DLOGZ = float(os.environ.get("DLOGZ", "0.1"))
MAXIMUM_UNCERTAINTY = float(os.environ.get("MAXIMUM_UNCERTAINTY", "1000000000.0"))
RESUME = os.environ.get("RESUME", "1") == "1"

os.makedirs(OUTDIR, exist_ok=True)
bb.core.utils.setup_logger(outdir=OUTDIR, label=LABEL)

# =========================================================
# Helper PDFs
# =========================================================
SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def gaussian_np(x, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * SQRT_2PI)


def trunc_gaussian_np(x, mu, sigma, xmin, xmax):
    sigma = max(float(sigma), 1e-12)
    x = np.asarray(x, dtype=float)
    a = (xmin - mu) / (sigma * np.sqrt(2.0))
    b = (xmax - mu) / (sigma * np.sqrt(2.0))
    z = 0.5 * (math_erf(b) - math_erf(a))
    z = max(float(z), 1e-300)
    p = gaussian_np(x, mu, sigma) / z
    return np.where((x >= xmin) & (x <= xmax), p, 0.0)


def ordered_uniform_mass_pdf_np(m1, m2, mmin, mmax):
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    dm = float(mmax - mmin)
    ok = (m1 >= m2) & (m2 >= mmin) & (m1 <= mmax)
    return np.where(ok, 2.0 / (dm * dm), 0.0)


def ordered_trunc_gaussian_mass_pdf_np(m1, m2, mu, sigma, mmin, mmax):
    p1 = trunc_gaussian_np(m1, mu, sigma, mmin, mmax)
    p2 = trunc_gaussian_np(m2, mu, sigma, mmin, mmax)
    return np.where(np.asarray(m1) >= np.asarray(m2), 2.0 * p1 * p2, 0.0)


def math_erf(x):
    from math import erf as _erf

    if np.isscalar(x):
        return _erf(x)
    return np.vectorize(_erf)(x)


def trunc_gaussian_jax(x, mu, sigma, xmin, xmax):
    x = xp.asarray(x)
    mu = xp.asarray(mu)
    sigma = xp.maximum(xp.asarray(sigma), 1e-12)
    xmin = xp.asarray(xmin)
    xmax = xp.asarray(xmax)
    p = xp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * xp.sqrt(2.0 * xp.pi))
    a = (xmin - mu) / (sigma * xp.sqrt(2.0))
    b = (xmax - mu) / (sigma * xp.sqrt(2.0))
    z = 0.5 * (erf(b) - erf(a))
    z = xp.maximum(z, 1e-300)
    p = p / z
    return xp.where((x >= xmin) & (x <= xmax), p, 0.0)


def gaussian_jax(x, mu, sigma):
    x = xp.asarray(x)
    mu = xp.asarray(mu)
    sigma = xp.maximum(xp.asarray(sigma), 1e-12)
    return xp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * xp.sqrt(2.0 * xp.pi))


def ordered_trunc_gaussian_mass_pdf_jax(m1, m2, mu, sigma, mmin, mmax):
    p1 = trunc_gaussian_jax(m1, mu, sigma, mmin, mmax)
    p2 = trunc_gaussian_jax(m2, mu, sigma, mmin, mmax)
    return xp.where(xp.asarray(m1) >= xp.asarray(m2), 2.0 * p1 * p2, 0.0)


# =========================================================
# Normalized p(z) used by population and VT
# =========================================================
# With fixed Om0 and w0, varying H0 only changes dVc/dz by an overall H0^-3,
# which cancels after normalization. So the normalized redshift PDF can be
# precomputed at any convenient fiducial H0.
_z_grid = xp.linspace(0.0, Z_MAX, Z_GRID_SIZE)
_pz_unnorm_grid = (
    4.0
    * xp.pi
    * wc.differential_comoving_volume(_z_grid, H0=H0_POP, Om0=OM0, w0=W0)
    * (1.0 + _z_grid) ** (GAMMA - 1.0)
)
_pz_norm = xp.trapezoid(_pz_unnorm_grid, _z_grid)


def redshift_pdf_jax(z):
    z = xp.asarray(z)
    p = (
        4.0
        * xp.pi
        * wc.differential_comoving_volume(z, H0=H0_POP, Om0=OM0, w0=W0)
        * (1.0 + z) ** (GAMMA - 1.0)
    ) / xp.maximum(_pz_norm, 1e-300)
    return xp.where((z >= 0.0) & (z <= Z_MAX), p, 0.0)


def redshift_pdf_np(z, zmax, gamma, h0, om0, w0):
    z = np.asarray(z, dtype=float)
    z_grid = np.linspace(0.0, zmax, 6000)
    p_grid = (
        4.0
        * np.pi
        * np.asarray(wc.differential_comoving_volume(z_grid, H0=h0, Om0=om0, w0=w0), dtype=float)
        * (1.0 + z_grid) ** (gamma - 1.0)
    )
    norm = float(np.trapezoid(p_grid, z_grid))
    p = (
        4.0
        * np.pi
        * np.asarray(wc.differential_comoving_volume(z, H0=h0, Om0=om0, w0=w0), dtype=float)
        * (1.0 + z) ** (gamma - 1.0)
    ) / max(norm, 1e-300)
    return np.where((z >= 0.0) & (z <= zmax), p, 0.0)


# =========================================================
# Event / VT models
# =========================================================
def event_model_density(
    dataset,
    mu_H,
    sig_H,
    mu_a0,
    sig_a0,
    mu_a1,
    sig_a1,
    mu_a2,
    sig_a2,
    mu_m,
    sig_m,
):
    z = dataset["redshift_sample"]
    m1s = dataset["mass_1_source"]
    m2s = dataset["mass_2_source"]
    h0_obs = dataset["H0_sample"]

    p_h0 = trunc_gaussian_jax(h0_obs, mu_H, sig_H, H0_MIN, H0_MAX)
    p_a0 = gaussian_jax(dataset["delta_a0"], mu_a0, sig_a0)
    p_a1 = gaussian_jax(dataset["delta_a1"], mu_a1, sig_a1)
    p_a2 = gaussian_jax(dataset["delta_a2"], mu_a2, sig_a2)

    p_m = ordered_trunc_gaussian_mass_pdf_jax(m1s, m2s, mu_m, sig_m, M_POP_MIN, M_POP_MAX)
    p_z = redshift_pdf_jax(z)

    dz_ddL = 1.0 / wc.dDLdz(z, H0=h0_obs, Om0=OM0, w0=W0)
    jac_mass = 1.0 / (1.0 + z) ** 2

    out = p_h0 * p_a0 * p_a1 * p_a2 * p_m * p_z * dz_ddL * jac_mass
    return xp.maximum(out, 1e-300)


# Keep VT dependent only on source-frame population variables. This matches the
# current selection-generation scripts and treats the H0 / delta_a hyper-
# distributions as a recycling-level surrogate rather than an explicit driver of
# detectability.
def vt_model_density(dataset, mu_H, sig_H, mu_a0, sig_a0, mu_a1, sig_a1, mu_a2, sig_a2, mu_m, sig_m):
    m1s = dataset["mass_1_source"]
    m2s = dataset["mass_2_source"]
    z = dataset["redshift"]
    p_m = ordered_trunc_gaussian_mass_pdf_jax(m1s, m2s, mu_m, sig_m, M_POP_MIN, M_POP_MAX)
    p_z = redshift_pdf_jax(z)
    return xp.maximum(p_m * p_z, 1e-300)


# =========================================================
# Data loading
# =========================================================
def load_event_posteriors():
    paths = sorted(glob.glob(PE_POST_GLOB))
    if len(paths) == 0:
        raise FileNotFoundError(f"No posterior CSVs matched PE_POST_GLOB={PE_POST_GLOB}")

    rng = np.random.default_rng(RNG_SEED)
    posteriors = []
    info = []

    required = [
        "luminosity_distance",
        "H0_sample",
        "delta_a0",
        "delta_a1",
        "delta_a2",
        "redshift_sample",
        "mass_1_source",
        "mass_2_source",
        "log_prior",
    ]

    for path in paths:
        df = pd.read_csv(path)
        missing = [k for k in required if k not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
        df["prior"] = np.exp(np.clip(df["log_prior"].to_numpy(dtype=float), -745.0, 300.0))
        df = df[np.isfinite(df["prior"]) & (df["prior"] > 0.0)].copy()

        keep = [
            "luminosity_distance",
            "H0_sample",
            "delta_a0",
            "delta_a1",
            "delta_a2",
            "redshift_sample",
            "mass_1_source",
            "mass_2_source",
            "prior",
        ]

        if MAX_SAMPLES > 0 and len(df) > MAX_SAMPLES:
            idx = rng.choice(len(df), size=MAX_SAMPLES, replace=False)
            df = df.iloc[np.sort(idx)].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        posteriors.append(df[keep].copy())
        info.append({"path": path, "nsamples": int(len(df))})

    return posteriors, info


def selection_mass_pdf_np(m1, m2):
    if SEL_MASS_DIST == "flat":
        return ordered_uniform_mass_pdf_np(m1, m2, M_SEL_MIN, M_SEL_MAX)
    if SEL_MASS_DIST in {"gaussian", "trunc_gaussian", "truncated_gaussian"}:
        return ordered_trunc_gaussian_mass_pdf_np(m1, m2, M_SEL_MU, M_SEL_SIGMA, M_SEL_MIN, M_SEL_MAX)
    raise ValueError(f"Unknown SEL_MASS_DIST={SEL_MASS_DIST}")


def load_selection_injections():
    with open(SEL_MERGED_SUMMARY, "r", encoding="utf-8") as f:
        summary = json.load(f)
    total_generated = int(summary["n_trials_total"])
    data = np.load(SEL_MERGED_NPZ, allow_pickle=False)

    m1s = np.asarray(data["m1_src"], dtype=float)
    m2s = np.asarray(data["m2_src"], dtype=float)
    z = np.asarray(data["z"], dtype=float)

    swap = m2s > m1s
    if np.any(swap):
        m1s[swap], m2s[swap] = m2s[swap], m1s[swap]

    pz = redshift_pdf_np(z, SEL_Z_MAX, SEL_GAMMA, SEL_H0, OM0, W0)
    pm = selection_mass_pdf_np(m1s, m2s)
    prior = np.maximum(pz * pm, 1e-300)

    injections = dict(
        mass_1_source=m1s,
        mass_2_source=m2s,
        redshift=z,
        prior=prior,
        total_generated=total_generated,
    )
    return injections, {
        "n_detected": int(len(z)),
        "total_generated": total_generated,
        "selection_mass_dist": SEL_MASS_DIST,
    }


# =========================================================
# Main
# =========================================================
def main():
    posteriors, event_info = load_event_posteriors()
    vt_injections, vt_info = load_selection_injections()

    event_model = bb.hyper.model.Model(model_functions=[event_model_density], cache=False)
    vt_model = bb.hyper.model.Model(model_functions=[vt_model_density], cache=False)

    vt = gwpop.vt.ResamplingVT(model=vt_model, data=vt_injections, n_events=len(posteriors))
    likelihood = gwpop.hyperpe.HyperparameterLikelihood(
        posteriors=posteriors,
        hyper_prior=event_model,
        selection_function=vt,
        max_samples=MAX_SAMPLES,
        maximum_uncertainty=MAXIMUM_UNCERTAINTY,
    )

    priors = PriorDict()
    priors["mu_H"] = Uniform(H0_MIN, H0_MAX, latex_label=r"$\mu_{H_0}$")
    priors["sig_H"] = Uniform(SIG_H_MIN, SIG_H_MAX, latex_label=r"$\sigma_{H_0}$")

    priors["mu_a0"] = Gaussian(mu=0.0, sigma=DELTA_MU_SIGMA, latex_label=r"$\mu_{\delta a_0}$")
    priors["sig_a0"] = Uniform(SIG_DA_MIN, SIG_DA_MAX, latex_label=r"$\sigma_{\delta a_0}$")

    priors["mu_a1"] = Gaussian(mu=0.0, sigma=DELTA_MU_SIGMA, latex_label=r"$\mu_{\delta a_1}$")
    priors["sig_a1"] = Uniform(SIG_DA_MIN, SIG_DA_MAX, latex_label=r"$\sigma_{\delta a_1}$")

    priors["mu_a2"] = Gaussian(mu=0.0, sigma=DELTA_MU_SIGMA, latex_label=r"$\mu_{\delta a_2}$")
    priors["sig_a2"] = Uniform(SIG_DA_MIN, SIG_DA_MAX, latex_label=r"$\sigma_{\delta a_2}$")

    priors["mu_m"] = Uniform(M_POP_MIN, M_POP_MAX, latex_label=r"$\mu_m$")
    priors["sig_m"] = Uniform(SIG_M_MIN, SIG_M_MAX, latex_label=r"$\sigma_m$")

    test_params = priors.sample()
    _ = likelihood.log_likelihood_ratio(test_params)
    jit_like = JittedLikelihood(likelihood)
    _ = jit_like.log_likelihood_ratio(test_params)

    meta = {
        "label": LABEL,
        "pe_post_glob": PE_POST_GLOB,
        "n_events": len(posteriors),
        "event_info": event_info,
        "selection_info": vt_info,
        "cosmology": {"H0_fid_for_pz": H0_POP, "Om0": OM0, "w0": W0},
        "population": {"z_max": Z_MAX, "gamma": GAMMA, "mmin": M_POP_MIN, "mmax": M_POP_MAX},
        "notes": [
            "Event recycling denominator is read directly from each PE posterior log_prior column.",
            "H0 and delta_a_i are modeled at the recycling step by hyper-distributions with means and spreads.",
            "Selection function is evaluated only in source-frame proposal variables (m1_source, m2_source, z).",
            "This matches the current selection-generation scripts and leaves VT independent of the H0 / delta_a_i hyper-distributions.",
        ],
    }
    with open(os.path.join(OUTDIR, f"{LABEL}_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    result = bb.run_sampler(
        likelihood=jit_like,
        priors=priors,
        sampler="dynesty",
        nlive=NLIVE,
        dlogz=DLOGZ,
        sample="acceptance-walk",
        naccept=5,
        outdir=OUTDIR,
        label=LABEL,
        resume=RESUME,
        save="hdf5",
        check_point=True,
        check_point_plot=False,
    )

    truth = {
        "mu_H": H0_POP,
        "mu_a0": 0.0,
        "mu_a1": 0.0,
        "mu_a2": 0.0,
    }
    fig = result.plot_corner(save=False, truth=truth, truth_color="orange")
    fig.savefig(os.path.join(OUTDIR, f"corner_{LABEL}.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
