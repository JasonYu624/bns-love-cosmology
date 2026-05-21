#!/usr/bin/env python3
"""HyperPE EOS-fit pipeline with backend-agnostic array operations."""

import argparse
import glob
import json
import math
import os
import time

import bilby as bb
import gwpopulation as gwpop
import numpy as np
import pandas as pd
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.hyper.model import Model
from gwpopulation.models.redshift import PowerLawRedshift
from wcosmo.astropy import FlatwCDM, Planck18
from wcosmo.utils import disable_units

disable_units()

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
    p1 = gwpop.utils.truncnorm(m1, mu=mu_m, sigma=sigma_m, low=0.8, high=1.8)
    p2 = gwpop.utils.truncnorm(m2, mu=mu_m, sigma=sigma_m, low=0.8, high=1.8)
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
    p.add_argument("--posterior-glob", type=str, default="outdir_population_run/*_reweighted_posterior_augmented.csv")
    p.add_argument("--vt-glob", type=str, default="outdir_selection_calib_eosfit/worker_*/detected_chunk*.npz")
    p.add_argument("--outdir", type=str, default="outdir_hyperpe_eosfit")
    p.add_argument("--label", type=str, default="hyperpe_eosfit")
    p.add_argument("--nlive", type=int, default=500)
    p.add_argument("--backend", choices=["jax", "numpy"], default="jax")
    p.add_argument("--fix-lamb-zero", action="store_true")
    p.add_argument("--jit-likelihood", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def load_event_posteriors(posterior_glob: str):
    paths = sorted(glob.glob(posterior_glob))
    if not paths:
        raise FileNotFoundError(f"No posterior files matched: {posterior_glob}")

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


def load_vt_injections(vt_glob: str):
    paths = sorted(glob.glob(vt_glob))
    if not paths:
        raise FileNotFoundError(f"No VT files matched: {vt_glob}")

    required = ["mass_1_detector", "mass_2_detector", "luminosity_distance", "H0", "prior"]
    frames = []
    total_generated = 0

    for path in paths:
        with np.load(path) as npz:
            df = pd.DataFrame({k: npz[k] for k in npz.files})
        df = _map_vt_columns(df)

        summary_path = os.path.join(os.path.dirname(path), "summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            total_generated += int(json.load(f)["total_generated"])

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

    from gwpopulation.utils import xp

    injections = {
        "mass_1_detector": xp.asarray(vt_df["mass_1_detector"].to_numpy(dtype=float)),
        "mass_2_detector": xp.asarray(vt_df["mass_2_detector"].to_numpy(dtype=float)),
        "luminosity_distance": xp.asarray(vt_df["luminosity_distance"].to_numpy(dtype=float)),
        "H0": xp.asarray(vt_df["H0"].to_numpy(dtype=float)),
        "prior": xp.asarray(vt_df["prior"].to_numpy(dtype=float)),
        "total_generated": int(total_generated),
    }
    return injections, paths, total_generated, n_found_after


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
    priors["lamb"] = DeltaFunction(peak=0.0, name="lamb") if fix_lamb_zero else Uniform(-1.0, 10.0, name="lamb")
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


def write_diagnostics(path, args, posterior_files, vt_files, posteriors, found_vt, total_generated):
    h0_all = pd.concat([df["H0"] for df in posteriors], ignore_index=True).astype(float)
    prior_all = pd.concat([df["prior"] for df in posteriors], ignore_index=True).astype(float)
    diag = {
        "args": vars(args),
        "posterior_files_used": posterior_files,
        "vt_files_used": vt_files,
        "n_events": len(posteriors),
        "samples_per_event": [int(len(df)) for df in posteriors],
        "n_found_vt_injections": int(found_vt),
        "total_generated": int(total_generated),
        "found_fraction": float(found_vt / total_generated) if total_generated > 0 else math.nan,
        "H0_min": float(h0_all.min()),
        "H0_max": float(h0_all.max()),
        "prior_min": float(prior_all.min()),
        "prior_max": float(prior_all.max()),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, sort_keys=True)


def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    bb.core.utils.setup_logger(outdir=args.outdir, label=args.label)

    gwpop.set_backend(args.backend)
    print(f"Backend: {args.backend}")
    if args.backend == "jax":
        import jax

        print("JAX devices:", jax.devices())

    posteriors, posterior_files = load_event_posteriors(args.posterior_glob)
    injections, vt_files, total_generated, n_found_vt = load_vt_injections(args.vt_glob)

    n_samples_total = sum(len(p) for p in posteriors)
    print(f"Loaded {len(posteriors)} events, total posterior samples = {n_samples_total}")
    print(f"Loaded {n_found_vt} found VT injections (total_generated = {total_generated})")

    event_model, vt_model = make_models()
    vt_model.prepare_data(injections)
    vt = gwpop.vt.ResamplingVT(
        model=vt_model,
        data=injections,
        n_events=len(posteriors),
        marginalize_uncertainty=False,
        enforce_convergence=True,
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
    n_eff = float(mu**2 / var) if var > 0 else math.inf
    print(f"VT N_eff = {n_eff:.1f}  (threshold 4*N_events = {4 * len(posteriors)})")
    print(f"Estimated time for 10k likelihood evals: {t_like * 10000 / 60:.1f} min")

    diag_path = os.path.join(args.outdir, f"{args.label}_diagnostics.json")
    write_diagnostics(diag_path, args, posterior_files, vt_files, posteriors, n_found_vt, total_generated)

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
        resume=True,
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


if __name__ == "__main__":
    main()
