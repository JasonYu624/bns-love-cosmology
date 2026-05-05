#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys

import bilby
import numpy as np
import pandas as pd
from bilby.core.utils.random import seed as bilby_seed
from bilby_cython.geometry import rotation_matrix_from_delta
from bilby_cython.time import greenwich_mean_sidereal_time
from scipy.special import logsumexp

for _k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ.setdefault(_k, "1")


class UniformReflected(bilby.core.prior.analytical.Uniform):
    def rescale(self, val):
        u = 2 * np.minimum(val, 1 - val)
        return super().rescale(u)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-outdir", default="outdir_population_exactfd")
    parser.add_argument("--event-index", type=int, default=1)
    parser.add_argument("--outdir", default="outdir_population_run_test")
    parser.add_argument("--label")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--widen-mc", type=float, default=0.1)
    parser.add_argument("--nlive", type=int, default=1000)
    parser.add_argument("--naccept", type=int, default=60)
    parser.add_argument("--npool", type=int, default=1)
    parser.add_argument(
        "--sky-frame",
        choices=["detector", "sky"],
        default="detector",
        help="Sky parameterization: detector uses zenith/azimuth, sky uses ra/dec.",
    )
    parser.add_argument("--lambda-tilde-min", type=float, default=10.0)
    parser.add_argument("--lambda-tilde-max", type=float, default=8000.0)
    parser.add_argument("--delta-lambda-tilde-max", type=float, default=8000.0)
    parser.add_argument("--fix-tidal", action="store_true")
    parser.add_argument("--update-fiducial", action="store_true")
    parser.add_argument("--rw-npool", type=int, default=4)
    parser.add_argument("--rw-resume-file")
    parser.add_argument("--rw-checkpoint", type=int, default=2000)
    parser.add_argument("--resume-reweight", action="store_true")
    parser.add_argument("--skip-reweight", action="store_true")
    # Keep this flag only for backward compatibility with existing submit scripts.
    parser.add_argument(
        "--rw-method",
        choices=["weighted"],
        default="weighted",
        help="Reweight resampling method (weighted only).",
    )
    parser.add_argument(
        "--rw-use-nested-samples",
        action="store_true",
        help="Use nested samples in bilby.reweight (can improve support coverage)",
    )
    return parser.parse_args()


args = get_args()

POP_OUTDIR = args.pop_outdir
EVENT_INDEX = args.event_index
EVENT_NAME = f"event_{EVENT_INDEX:04d}"
META_PATH = os.path.join(POP_OUTDIR, EVENT_NAME, "meta.json")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

inj = dict(meta["injection_parameters"])
seg = meta["data_segment"]
noise_seed = int(meta["noise_seed"])
exact_info = meta.get("exact_signal_file")
if exact_info is None:
    raise ValueError("meta.json does not contain exact_signal_file")

SIGNAL_PATH = os.path.join(POP_OUTDIR, EVENT_NAME, exact_info["filename"])
duration = int(seg["duration"])
start_time = float(seg["start_time"])
sampling_frequency = float(seg["sampling_frequency"])
fmin = float(seg["fmin"])
ifo_files = list(meta["reproduction"]["ifo_files"])

waveform_arguments_full = dict(meta["reproduction"]["waveform_arguments"])
waveform_arguments_full.pop("frequency_bin_edges", None)
waveform_arguments_full.pop("fiducial", None)
waveform_arguments_rb = dict(waveform_arguments_full)
waveform_arguments_rb["fiducial"] = 1

outdir = args.outdir
label = args.label or f"bns_{EVENT_NAME}_traditional"
bilby.core.utils.setup_logger(outdir=outdir, label=label)
RUN_REWEIGHT = not args.skip_reweight
RW_LABEL = label + "_reweighted"
RW_CLEAN_RESUME = not args.resume_reweight
RW_N_CHECKPOINT = args.rw_checkpoint
RW_NPOOL = args.rw_npool
RW_METHOD = args.rw_method
RW_USE_NESTED = args.rw_use_nested_samples
SKY_FRAME = args.sky_frame
RESULT_EXTENSION = "hdf5"
resume_dir = os.path.dirname(args.rw_resume_file) if args.rw_resume_file else outdir
os.makedirs(resume_dir, exist_ok=True)
RESUME_FILE = args.rw_resume_file or os.path.join(resume_dir, f"{RW_LABEL}_weights_resume.npz")


def load_exact_detector_signals(signal_path, info):
    with np.load(signal_path) as data:
        frequency_array = np.asarray(data[info["frequency_array_key"]])
        detector_signals = {
            ifo_name: np.asarray(data[key])
            for ifo_name, key in info["detector_signal_keys"].items()
        }
    return frequency_array, detector_signals


EXACT_FREQUENCY_ARRAY, EXACT_DETECTOR_SIGNALS = load_exact_detector_signals(
    SIGNAL_PATH, exact_info
)


def build_interferometers_with_exact_data(zero_noise=False):
    ifos = bilby.gw.detector.InterferometerList(
        [bilby.gw.detector.load_interferometer(f) for f in ifo_files]
    )
    for ifo in ifos:
        ifo.minimum_frequency = fmin

    if zero_noise:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )
    else:
        bilby_seed(noise_seed)
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )

    for ifo in ifos:
        ifo.minimum_frequency = fmin
        ifo_freqs = np.asarray(ifo.frequency_array)
        if ifo_freqs.shape != EXACT_FREQUENCY_ARRAY.shape or not np.allclose(
            ifo_freqs, EXACT_FREQUENCY_ARRAY, rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"Frequency grid mismatch for {ifo.name}")
        ifo.strain_data.frequency_domain_strain += EXACT_DETECTOR_SIGNALS[ifo.name]
    return ifos


def theta_phi_to_zenith_azimuth(theta, phi, ifos):
    """Inverse map for detector-frame sampling (theta/phi -> zenith/azimuth)."""
    rot = rotation_matrix_from_delta(ifos[0].vertex - ifos[1].vertex)
    rotated = rot.T @ np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    return np.arccos(rotated[2]), np.mod(np.arctan2(rotated[1], rotated[0]), 2.0 * np.pi)


def convert_traditional_bns(parameters, *args, **kwargs):
    is_df = isinstance(parameters, pd.DataFrame)
    p = (
        {k: parameters[k].to_numpy() for k in parameters.columns}
        if is_df
        else dict(parameters)
    )
    for idx in ("1", "2"):
        p.pop(f"chi_{idx}_in_plane", None)
        if f"chi_{idx}" in p and f"a_{idx}" in p:
            try:
                if float(p[f"a_{idx}"]) == 0.0:
                    p.pop(f"a_{idx}", None)
                    p.pop(f"cos_tilt_{idx}", None)
                    p.pop(f"tilt_{idx}", None)
            except Exception:
                pass

    converted, _ = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters(p)
    converted = bilby.gw.conversion.generate_mass_parameters(converted)
    converted = bilby.gw.conversion.generate_tidal_parameters(converted)
    if is_df:
        return pd.DataFrame(converted, index=parameters.index)
    return converted


def make_priors():
    p = bilby.core.prior.PriorDict(conversion_function=convert_traditional_bns)
    if SKY_FRAME == "detector":
        p["zenith"] = bilby.core.prior.Sine(name="zenith", latex_label="$\\kappa$")
        p["azimuth"] = bilby.core.prior.Uniform(
            minimum=0.0, maximum=2.0 * np.pi, boundary="periodic", name="azimuth"
        )
    else:
        p["ra"] = bilby.core.prior.Uniform(
            minimum=0.0, maximum=2.0 * np.pi, boundary="periodic", name="ra"
        )
        p["dec"] = bilby.core.prior.Cosine(name="dec")
    p["theta_jn"] = bilby.core.prior.Sine(name="theta_jn", latex_label="$\\theta_{JN}$")
    p["psi"] = bilby.core.prior.Uniform(
        minimum=0.0, maximum=np.pi, boundary="periodic", name="psi"
    )
    p["phase"] = bilby.core.prior.Uniform(
        minimum=0.0, maximum=2.0 * np.pi, boundary="periodic", name="phase"
    )
    p["luminosity_distance"] = bilby.core.prior.PowerLaw(
        alpha=2.0,
        minimum=10.0,
        maximum=2500.0,
        name="luminosity_distance",
        latex_label="$d_L$",
        unit="Mpc",
    )
    p["chi_1"] = bilby.core.prior.Uniform(
        minimum=-0.05, maximum=0.05, name="chi_1", latex_label="$\\chi_1$"
    )
    p["chi_2"] = bilby.core.prior.Uniform(
        minimum=-0.05, maximum=0.05, name="chi_2", latex_label="$\\chi_2$"
    )
    p["chirp_mass"] = bilby.core.prior.Uniform(
        minimum=Mc_inj - args.widen_mc,
        maximum=Mc_inj + args.widen_mc,
        name="chirp_mass",
        latex_label="$\\mathcal{M}^{\\rm det}$",
    )
    p["mass_ratio"] = UniformReflected(
        minimum=0.5, maximum=1.0, name="mass_ratio", latex_label="$q$"
    )
    p["geocent_time"] = bilby.core.prior.Uniform(
        minimum=inj["geocent_time"] - 0.05,
        maximum=inj["geocent_time"] + 0.05,
        name="geocent_time",
    )

    if args.fix_tidal:
        p["lambda_tilde"] = bilby.core.prior.DeltaFunction(
            peak=lambda_tilde_inj, name="lambda_tilde", latex_label="$\\tilde{\\Lambda}$"
        )
        p["delta_lambda_tilde"] = bilby.core.prior.DeltaFunction(
            peak=delta_lambda_tilde_inj,
            name="delta_lambda_tilde",
            latex_label="$\\delta\\tilde{\\Lambda}$",
        )
    else:
        p["lambda_tilde"] = bilby.core.prior.Uniform(
            minimum=args.lambda_tilde_min,
            maximum=args.lambda_tilde_max,
            name="lambda_tilde",
            latex_label="$\\tilde{\\Lambda}$",
        )
        p["delta_lambda_tilde"] = bilby.core.prior.Uniform(
            minimum=-args.delta_lambda_tilde_max,
            maximum=args.delta_lambda_tilde_max,
            name="delta_lambda_tilde",
            latex_label="$\\delta\\tilde{\\Lambda}$",
        )

    p["lambda_1"] = bilby.core.prior.Constraint(minimum=10, maximum=1e4, name="lambda_1")
    p["lambda_2"] = bilby.core.prior.Constraint(minimum=10, maximum=1e4, name="lambda_2")
    return p


interferometers = build_interferometers_with_exact_data(zero_noise=args.zero_noise)
REFERENCE_FRAME = interferometers if SKY_FRAME == "detector" else "sky"

wg_rb = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star_relative_binning,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments_rb,
)
wg_full = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments_full,
)

m1_det = float(meta["mass_1_detector"])
m2_det = float(meta["mass_2_detector"])
Mc_inj = float(bilby.gw.conversion.component_masses_to_chirp_mass(m1_det, m2_det))
q_inj = min(m1_det, m2_det) / max(m1_det, m2_det)
lambda_tilde_inj = float(
    bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(
        float(inj["lambda_1"]), float(inj["lambda_2"]), m1_det, m2_det
    )
)
delta_lambda_tilde_inj = float(
    bilby.gw.conversion.lambda_1_lambda_2_to_delta_lambda_tilde(
        float(inj["lambda_1"]), float(inj["lambda_2"]), m1_det, m2_det
    )
)

priors = make_priors()

gmst = greenwich_mean_sidereal_time(inj["geocent_time"])
theta, phi = bilby.core.utils.conversion.ra_dec_to_theta_phi(inj["ra"], inj["dec"], gmst)
if SKY_FRAME == "detector":
    zenith, azimuth = theta_phi_to_zenith_azimuth(theta, phi, interferometers)
else:
    zenith, azimuth = None, None

fiducial_parameters = dict(
    chirp_mass=Mc_inj,
    mass_ratio=q_inj,
    luminosity_distance=float(inj["luminosity_distance"]),
    lambda_tilde=lambda_tilde_inj,
    delta_lambda_tilde=delta_lambda_tilde_inj,
    chi_1=float(inj.get("chi_1", 0.0)),
    chi_2=float(inj.get("chi_2", 0.0)),
    theta_jn=float(inj["theta_jn"]),
    psi=float(inj["psi"]),
    phase=float(inj.get("phase", 0.0)),
    geocent_time=float(inj["geocent_time"]),
    ra=float(inj["ra"]),
    dec=float(inj["dec"]),
)
if SKY_FRAME == "detector":
    fiducial_parameters["zenith"] = float(zenith)
    fiducial_parameters["azimuth"] = float(azimuth)
    fiducial_parameters.pop("ra", None)
    fiducial_parameters.pop("dec", None)
else:
    fiducial_parameters.pop("zenith", None)
    fiducial_parameters.pop("azimuth", None)


likelihood = bilby.gw.likelihood.relative.RelativeBinningGravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=wg_rb,
    fiducial_parameters=fiducial_parameters,
    update_fiducial_parameters=args.update_fiducial,
    reference_frame=REFERENCE_FRAME,
    time_reference="geocenter",
    distance_marginalization=False,
    phase_marginalization=True,
    time_marginalization=False,
    priors=priors,
    chi=20.0,
    epsilon=0.01,
)

print(f"=== Starting Dynesty Run for {EVENT_NAME} (traditional tides) ===", flush=True)
print(
    f"ZERO_NOISE={args.zero_noise}, WIDEN_MC={args.widen_mc}, "
    f"NLIVE={args.nlive}, FIX_TIDAL={args.fix_tidal}, SKY_FRAME={SKY_FRAME}, "
    f"lambda_tilde_inj={lambda_tilde_inj:.6g}, "
    f"delta_lambda_tilde_inj={delta_lambda_tilde_inj:.6g}",
    flush=True,
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=args.nlive,
    naccept=args.naccept,
    check_point_plot=True,
    check_point_delta_t=1800,
    npool=args.npool,
    outdir=outdir,
    label=label,
    resume=True,
    print_method="interval-60",
    sample="acceptance-walk",
    injection_parameters=fiducial_parameters,
    conversion_function=convert_traditional_bns,
    result_class=bilby.gw.result.CBCResult,
    save=RESULT_EXTENSION,
)

def ensure_dataframe(x):
    return pd.DataFrame(x) if isinstance(x, dict) else x.copy()


result.save_to_file(overwrite=True, extension=RESULT_EXTENSION, outdir=outdir)
print(
    f"Saved Result with nested samples to: "
    f"{bilby.core.result.result_file_name(outdir, label, extension=RESULT_EXTENSION)}",
    flush=True,
)


def ensure_traditional_augmented(df):
    df = ensure_dataframe(df)
    needed = {"mass_1", "mass_2", "lambda_1", "lambda_2"}
    if needed.issubset(df.columns):
        return df.copy()
    return convert_traditional_bns(df)


def save_corner_and_csv(res, out_label):
    post_df = ensure_traditional_augmented(res.posterior)
    post_csv = os.path.join(outdir, f"{out_label}_posterior_augmented.csv")
    post_df.to_csv(post_csv, index=False)

    truth = convert_traditional_bns(dict(fiducial_parameters))
    truth.update(mass_1=m1_det, mass_2=m2_det)
    plot_params = [
        "mass_ratio",
        "chirp_mass",
        "luminosity_distance",
        "theta_jn",
        "psi",
        "chi_1",
        "chi_2",
        "zenith",
        "azimuth",
        "ra",
        "dec",
        "geocent_time",
        "lambda_tilde",
        "delta_lambda_tilde",
        "lambda_1",
        "lambda_2",
        "mass_1",
        "mass_2",
    ]
    plot_params = [p for p in plot_params if p in post_df.columns]
    label_map = {
        "mass_ratio": r"$q$",
        "chirp_mass": r"$\mathcal{M}^{\rm det}$",
        "luminosity_distance": r"$d_L$",
        "theta_jn": r"$\theta_{JN}$",
        "psi": r"$\psi$",
        "chi_1": r"$\chi_1$",
        "chi_2": r"$\chi_2$",
        "zenith": r"$\kappa$",
        "azimuth": r"$\mathrm{azimuth}$",
        "ra": r"$\alpha$",
        "dec": r"$\delta$",
        "geocent_time": r"$t_c$",
        "lambda_tilde": r"$\tilde{\Lambda}$",
        "delta_lambda_tilde": r"$\delta\tilde{\Lambda}$",
        "lambda_1": r"$\Lambda_1$",
        "lambda_2": r"$\Lambda_2$",
        "mass_1": r"$m_1^{\rm det}$",
        "mass_2": r"$m_2^{\rm det}$",
    }
    plot_df = post_df[plot_params].replace([np.inf, -np.inf], np.nan).dropna().copy()
    plot_res = copy.copy(res)
    plot_res.posterior = plot_df
    fig = plot_res.plot_corner(
        parameters=plot_params,
        labels=[label_map[p] for p in plot_params],
        truths=[truth.get(p, np.nan) for p in plot_params],
        save=False,
        truth_color="red",
        quantiles=[0.16, 0.84],
    )
    corner_path = os.path.join(outdir, f"{out_label}_all_params_corner.png")
    fig.savefig(corner_path, dpi=300, bbox_inches="tight")
    print(f"[*] Saved augmented posterior to: {post_csv}", flush=True)
    print(f"[*] Saved corner plot to: {corner_path}", flush=True)


save_corner_and_csv(result, label)

if not RUN_REWEIGHT:
    print("--skip-reweight -> skip reweight.", flush=True)
    sys.exit(0)


def maybe_remove(path, do_remove):
    if do_remove and os.path.exists(path):
        os.remove(path)
        print(f"Removed stale resume: {path}", flush=True)


def build_new_likelihood():
    return bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=wg_full,
        reference_frame=REFERENCE_FRAME,
        time_reference="geocenter",
        distance_marginalization=False,
        phase_marginalization=True,
        time_marginalization=False,
        priors=make_priors(),
    )


def summarize_log_weights(lnw):
    finite = np.isfinite(lnw)
    finite_fraction = float(np.mean(finite))
    if not finite.any():
        return None, {
            "ok": False,
            "reason": "no finite log-weights",
            "finite_fraction": finite_fraction,
            "num_finite_weights": 0,
            "ess_fraction": 0.0,
            "max_normalized_weight": np.inf,
        }

    lnw_f = lnw[finite]
    w_rel = np.exp(np.clip(lnw_f - np.max(lnw_f), -745.0, 0.0))
    sw = np.sum(w_rel)
    if sw == 0.0 or not np.isfinite(sw):
        return None, {
            "ok": False,
            "reason": "all relative weights underflowed or are non-finite",
            "finite_fraction": finite_fraction,
            "num_finite_weights": int(finite.sum()),
            "ess_fraction": 0.0,
            "max_normalized_weight": np.inf,
        }

    w_norm = w_rel / sw
    ess = 1.0 / np.sum(w_norm ** 2)
    ess_fraction = float(ess / len(w_norm))
    max_normalized_weight = float(np.max(w_norm))
    ok = finite_fraction == 1.0 and ess_fraction > 1.0e-3 and max_normalized_weight < 0.99

    weights = np.zeros_like(lnw, dtype=float)
    weights[finite] = w_rel
    return weights, {
        "ok": bool(ok),
        "reason": "ok" if ok else "pathological importance weights",
        "finite_fraction": finite_fraction,
        "num_finite_weights": int(finite.sum()),
        "ess": float(ess),
        "ess_fraction": ess_fraction,
        "max_normalized_weight": max_normalized_weight,
    }


def weighted_resample_posterior(posterior, weights, n_samples):
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    n_samples = int(max(1, n_samples))
    return posterior.sample(
        n=n_samples,
        replace=True,
        weights=w,
        random_state=12345 + EVENT_INDEX,
    ).reset_index(drop=True)


def compute_reweight_arrays(result_obj, old_likelihood, new_likelihood, use_nested):
    res = copy.copy(result_obj)
    if use_nested:
        nested_df = ensure_dataframe(result_obj.nested_samples)
        res.posterior = nested_df.copy()
    else:
        res.posterior = ensure_dataframe(result_obj.posterior)
    maybe_remove(RESUME_FILE, RW_CLEAN_RESUME)

    lnw, new_ll, new_lp, old_ll, old_lp = bilby.core.result.get_weights_for_reweighting(
        result=res,
        new_likelihood=new_likelihood,
        old_likelihood=old_likelihood,
        old_prior=None,
        new_prior=None,
        resume_file=RESUME_FILE,
        n_checkpoint=RW_N_CHECKPOINT,
        npool=RW_NPOOL,
    )
    if use_nested:
        lnw = np.asarray(lnw, dtype=float) + np.log(np.asarray(res.posterior["weights"], dtype=float))

    posterior = res.posterior.copy()
    posterior["log_likelihood"] = new_ll
    posterior["log_prior"] = new_lp
    return res, posterior, lnw


def reweight_posterior_weighted(result_obj, old_likelihood, new_likelihood):
    res, posterior, lnw = compute_reweight_arrays(
        result_obj, old_likelihood, new_likelihood, use_nested=bool(RW_USE_NESTED)
    )
    weights, diag = summarize_log_weights(lnw)
    diag["resume_file"] = RESUME_FILE
    if not diag["ok"]:
        return None, diag

    n_eff = int(max(1, round(diag["ess"])))
    res.posterior = weighted_resample_posterior(posterior, weights, n_samples=n_eff)
    if len(res.posterior) == 0:
        diag.update(ok=False, reason="weighted resampling kept no samples")
        return None, diag
    res.posterior = convert_traditional_bns(res.posterior)
    res.label = RW_LABEL
    if RW_USE_NESTED:
        res.log_evidence += float(logsumexp(lnw))
    else:
        res.log_evidence += float(logsumexp(lnw) - np.log(len(posterior)))
    res.meta_data["reweighted_using_importance_resampling"] = True
    diag["resampling_method"] = "weighted_posterior_sampling"
    diag["n_eff_target"] = int(n_eff)
    diag["n_resampled"] = int(len(res.posterior))
    diag["n_unique_rows"] = int(
        len(
            ensure_dataframe(res.posterior)
            .drop(columns=["log_likelihood", "log_prior"], errors="ignore")
            .drop_duplicates()
        )
    )
    return res, diag


def resolve_nested_sampling_flag(result_obj, requested_use_nested):
    if not requested_use_nested:
        return False, "disabled_by_user"
    nested = getattr(result_obj, "nested_samples", None)
    if nested is None:
        return False, "nested_samples_missing"
    nested_df = ensure_dataframe(nested)
    if len(nested_df) == 0:
        return False, "nested_samples_empty"
    if "weights" not in nested_df.columns:
        return False, "nested_samples_missing_weights"
    return True, "ok"


print(f"=== Starting bilby reweight for {EVENT_NAME} ===", flush=True)
print(f"RW_LABEL={RW_LABEL}", flush=True)
result_rw = copy.copy(result)
result_rw.posterior = ensure_dataframe(result.posterior)
rw_use_nested_effective, nested_reason = resolve_nested_sampling_flag(result_rw, RW_USE_NESTED)
if RW_USE_NESTED and not rw_use_nested_effective:
    print(
        f"RW_USE_NESTED requested but falling back to posterior samples: {nested_reason}",
        flush=True,
    )
print(
    f"ZERO_NOISE={args.zero_noise}, WIDEN_MC={args.widen_mc}, FIX_TIDAL={args.fix_tidal}, "
    f"SKY_FRAME={SKY_FRAME}, "
    f"RW_METHOD={RW_METHOD}, RW_USE_NESTED={rw_use_nested_effective}",
    flush=True,
)
print("Using nested samples." if rw_use_nested_effective else "Using posterior samples only.", flush=True)
RW_USE_NESTED = rw_use_nested_effective
rw_result, diag = reweight_posterior_weighted(result_rw, likelihood, build_new_likelihood())
diag["nested_samples_reason"] = nested_reason
print("Reweight diagnostics:", json.dumps(diag, indent=2), flush=True)

if not diag["ok"]:
    summary_path = os.path.join(outdir, f"{RW_LABEL}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    raise RuntimeError(
        "Reweight aborted before importance resampling: "
        f"reason={diag['reason']}, "
        f"finite_fraction={diag['finite_fraction']}, "
        f"ess_fraction={diag['ess_fraction']}, "
        f"max_normalized_weight={diag['max_normalized_weight']}"
    )

summary = {
    "result_file": bilby.core.result.result_file_name(outdir, label, extension=RESULT_EXTENSION),
    "result_json": bilby.core.result.result_file_name(outdir, label, extension=RESULT_EXTENSION),
    "result_extension": RESULT_EXTENSION,
    "rw_label": RW_LABEL,
    "rw_method": RW_METHOD,
    "use_nested_samples": bool(RW_USE_NESTED),
    "n_rb_posterior": int(len(result_rw.posterior)),
    "n_rw_posterior": int(len(rw_result.posterior)),
    **diag,
}
summary_path = os.path.join(outdir, f"{RW_LABEL}_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"[*] Saved reweight summary to: {summary_path}", flush=True)

save_corner_and_csv(rw_result, RW_LABEL)
print("Done.", flush=True)
