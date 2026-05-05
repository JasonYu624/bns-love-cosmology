#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys

import bilby
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, Planck18
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
    parser.add_argument("--delta-sigma", type=float, default=1.0)
    parser.add_argument("--npool", type=int, default=1)
    parser.add_argument(
        "--sky-frame",
        choices=["detector", "sky"],
        default="detector",
        help="Sky parameterization: detector uses zenith/azimuth, sky uses ra/dec.",
    )
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


# =========================================================
# 0. Read meta information
# =========================================================
POP_OUTDIR = args.pop_outdir
EVENT_INDEX = args.event_index
EVENT_NAME = f"event_{EVENT_INDEX:04d}"

META_PATH = os.path.join(POP_OUTDIR, EVENT_NAME, "meta.json")
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Cannot find {META_PATH}.")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

inj = dict(meta["injection_parameters"])
seg = meta["data_segment"]
noise_seed = int(meta["noise_seed"])

exact_info = meta.get("exact_signal_file")
if exact_info is None:
    raise ValueError("meta.json does not contain exact_signal_file")

SIGNAL_PATH = os.path.join(POP_OUTDIR, EVENT_NAME, exact_info["filename"])
if not os.path.exists(SIGNAL_PATH):
    raise FileNotFoundError(f"Cannot find exact signal file: {SIGNAL_PATH}")


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
label = args.label or f"bns_{EVENT_NAME}_eosfit"
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# =========================================================
# Analysis settings
# =========================================================
ZERO_NOISE = args.zero_noise
WIDEN_MC = args.widen_mc
NLIVE = args.nlive
DELTA_SIGMA = args.delta_sigma
NPOOL = args.npool
SKY_FRAME = args.sky_frame

H0_TRUE = float(Planck18.H0.value)
RUN_REWEIGHT = not args.skip_reweight
RW_LABEL = label + "_reweighted"
RW_CLEAN_RESUME = not args.resume_reweight
RW_N_CHECKPOINT = args.rw_checkpoint
RW_NPOOL = args.rw_npool
RW_METHOD = args.rw_method
RW_USE_NESTED = args.rw_use_nested_samples
RESULT_EXTENSION = "hdf5"

resume_dir = os.path.dirname(args.rw_resume_file) if args.rw_resume_file else outdir
os.makedirs(resume_dir, exist_ok=True)
RESUME_FILE = args.rw_resume_file or os.path.join(resume_dir, f"{RW_LABEL}_weights_resume.npz")


# =========================================================
# 1. EOS-motivated UR
# =========================================================
LAMBDA_FIT_NORM = 3500.0
A0_FIT = -0.30781804
A1_FIT = 0.79244108
A2_FIT = -0.51480556
DELTA_TRUE = 0.0

cosmo_fid = FlatLambdaCDM(H0=70.0, Om0=float(Planck18.Om0), Tcmb0=Planck18.Tcmb0)
_z_grid = np.linspace(0.0, 2.0, 20000)
_dL_grid = np.asarray(cosmo_fid.luminosity_distance(_z_grid).value, dtype=float)


def z_from_dL_H0_vec(dL_mpc, H0):
    dL_scaled = np.asarray(dL_mpc, dtype=float) * (np.asarray(H0, dtype=float) / 70.0)
    return np.interp(np.clip(dL_scaled, _dL_grid[0], _dL_grid[-1]), _dL_grid, _z_grid)


def lambda_of_mbar_vec(mbar, delta_a0, delta_a1, delta_a2):
    mbar = np.asarray(mbar, dtype=float)
    poly = (
        1.0
        + A0_FIT * (1.0 + np.asarray(delta_a0, dtype=float))
        + A1_FIT * (1.0 + np.asarray(delta_a1, dtype=float)) * mbar
        + A2_FIT * (1.0 + np.asarray(delta_a2, dtype=float)) * mbar**2
    )
    lam = LAMBDA_FIT_NORM * poly / mbar**5
    lam = np.where(np.isfinite(lam), lam, 0.0)
    return np.maximum(lam, 1e-8)


EOSFIT_DERIVED_KEYS = {
    "mass_1", "mass_2", "mass_1_source", "mass_2_source", "redshift_sample",
    "lambda_1", "lambda_2", "lambda_tilde", "delta_lambda_tilde",
}
LAL_DERIVED_KEYS = {"a_1", "a_2", "tilt_1", "tilt_2", "cos_tilt_1", "cos_tilt_2", "phi_12", "phi_jl"}
DERIVED_KEYS = EOSFIT_DERIVED_KEYS | LAL_DERIVED_KEYS


def _drop_derived_parameters(parameters):
    p = parameters.copy()
    for key in DERIVED_KEYS.intersection(p):
        del p[key]
    return p


def add_ur_derived_parameters(parameters):
    p = dict(parameters)

    dL = np.asarray(p["luminosity_distance"], dtype=float)
    H0 = np.asarray(p.get("H0_sample", H0_TRUE), dtype=float)
    q = np.asarray(p["mass_ratio"], dtype=float)
    Mc = np.asarray(p["chirp_mass"], dtype=float)
    m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(Mc, q)
    z = z_from_dL_H0_vec(dL, H0)

    m1_src, m2_src = np.asarray(m1, dtype=float) / (1.0 + z), np.asarray(m2, dtype=float) / (1.0 + z)
    delta_a0 = np.asarray(p.get("delta_a0", DELTA_TRUE), dtype=float)
    delta_a1 = np.asarray(p.get("delta_a1", DELTA_TRUE), dtype=float)
    delta_a2 = np.asarray(p.get("delta_a2", DELTA_TRUE), dtype=float)
    lam1, lam2 = (
        lambda_of_mbar_vec(m1_src, delta_a0, delta_a1, delta_a2),
        lambda_of_mbar_vec(m2_src, delta_a0, delta_a1, delta_a2),
    )

    p.update(
        mass_1=m1, mass_2=m2, mass_1_source=m1_src, mass_2_source=m2_src,
        redshift_sample=z, lambda_1=lam1, lambda_2=lam2,
        lambda_tilde=bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(lam1, lam2, m1, m2),
        delta_lambda_tilde=bilby.gw.conversion.lambda_1_lambda_2_to_delta_lambda_tilde(lam1, lam2, m1, m2),
    )
    return p


def _convert_eosfit_to_lal_bns(parameters):
    sampling_parameters = _drop_derived_parameters(parameters)
    eosfit_parameters = add_ur_derived_parameters(sampling_parameters)

    lal_input = dict(sampling_parameters)
    for key in ("mass_1", "mass_2", "lambda_1", "lambda_2"):
        lal_input[key] = eosfit_parameters[key]

    converted, _ = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters(lal_input)
    converted.update({key: eosfit_parameters[key] for key in EOSFIT_DERIVED_KEYS})

    added_keys = [key for key in converted if key not in sampling_parameters]
    return converted, added_keys


def convert_eosfit_to_lal_bns(parameters, *args, **kwargs):
    converted, _ = _convert_eosfit_to_lal_bns(parameters)
    if isinstance(parameters, pd.DataFrame):
        return pd.DataFrame(converted, index=parameters.index)
    return converted


def convert_eosfit_to_lal_bns_for_waveform(parameters, *args, **kwargs):
    return _convert_eosfit_to_lal_bns(parameters)


# =========================================================
# 2. Exact-signal data loading
# =========================================================
def load_exact_detector_signals(signal_path, exact_info):
    with np.load(signal_path) as data:
        freq_key = exact_info["frequency_array_key"]
        frequency_array = np.asarray(data[freq_key])
        detector_signals = {
            ifo_name: np.asarray(data[key])
            for ifo_name, key in exact_info["detector_signal_keys"].items()
        }
    return frequency_array, detector_signals


EXACT_FREQUENCY_ARRAY, EXACT_DETECTOR_SIGNALS = load_exact_detector_signals(SIGNAL_PATH, exact_info)


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


# =========================================================
# 3. Detectors and waveform generators
# =========================================================
interferometers = build_interferometers_with_exact_data(zero_noise=ZERO_NOISE)
REFERENCE_FRAME = interferometers if SKY_FRAME == "detector" else "sky"

wg_rb = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star_relative_binning,
    parameter_conversion=convert_eosfit_to_lal_bns_for_waveform,
    waveform_arguments=waveform_arguments_rb,
)

wg_full = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=convert_eosfit_to_lal_bns_for_waveform,
    waveform_arguments=waveform_arguments_full,
)

# =========================================================
# 4. Priors
# =========================================================
def make_priors():
    p = bilby.core.prior.PriorDict(conversion_function=convert_eosfit_to_lal_bns)
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
    p["psi"] = bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic", name="psi")
    p["phase"] = bilby.core.prior.Uniform(minimum=0.0, maximum=2.0*np.pi, boundary="periodic", name="phase")

    p["luminosity_distance"] = bilby.core.prior.PowerLaw(alpha=2.0, minimum=10.0, maximum=2500.0, name="luminosity_distance", latex_label="$d_L$", unit="Mpc")

    p["chi_1"] = bilby.core.prior.Uniform(minimum=-0.05, maximum=0.05, name="chi_1", latex_label="$\\chi_1$")
    p["chi_2"] = bilby.core.prior.Uniform(minimum=-0.05, maximum=0.05, name="chi_2", latex_label="$\\chi_2$")

    p["chirp_mass"] = bilby.core.prior.Uniform(minimum=Mc_inj - WIDEN_MC, maximum=Mc_inj + WIDEN_MC, name="chirp_mass", latex_label="$\\mathcal{M}$")

    p["mass_ratio"] = UniformReflected(minimum=0.5, maximum=1.0, name="mass_ratio", latex_label="$q$")

    p["geocent_time"] = bilby.core.prior.Uniform(minimum=inj["geocent_time"] - 0.05, maximum=inj["geocent_time"] + 0.05, name="geocent_time")
    p["H0_sample"] = bilby.core.prior.Uniform(minimum=10, maximum=150, name="H0_sample", latex_label="$H_0$")
    p["delta_a0"] = bilby.core.prior.Gaussian(mu=0.0, sigma=DELTA_SIGMA, name="delta_a0", latex_label="$\\delta a_0$")
    p["delta_a1"] = bilby.core.prior.Gaussian(mu=0.0, sigma=DELTA_SIGMA, name="delta_a1", latex_label="$\\delta a_1$")
    p["delta_a2"] = bilby.core.prior.Gaussian(mu=0.0, sigma=DELTA_SIGMA, name="delta_a2", latex_label="$\\delta a_2$")

    p["mass_1_source"] = bilby.core.prior.Constraint(minimum=0.8, maximum=1.8, name="mass_1_source")
    p["mass_2_source"] = bilby.core.prior.Constraint(minimum=0.8, maximum=1.8, name="mass_2_source")
    p["lambda_1"] = bilby.core.prior.Constraint(minimum=10, maximum=1e4, name="lambda_1")
    p["lambda_2"] = bilby.core.prior.Constraint(minimum=10, maximum=1e4, name="lambda_2")

    return p

m1_det = float(meta["mass_1_detector"])
m2_det = float(meta["mass_2_detector"])
Mc_inj = float(bilby.gw.conversion.component_masses_to_chirp_mass(m1_det, m2_det))
q_inj = min(m1_det, m2_det) / max(m1_det, m2_det)

priors = make_priors()


# =========================================================
# 5. Fiducial and likelihood
# =========================================================
def theta_phi_to_zenith_azimuth(theta, phi, ifos):
    """Inverse map for detector-frame sampling (theta/phi -> zenith/azimuth)."""
    rot = rotation_matrix_from_delta(ifos[0].vertex - ifos[1].vertex)
    rotated = rot.T @ np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    zen = np.arccos(rotated[2])
    az = np.arctan2(rotated[1], rotated[0])
    return zen, np.mod(az, 2.0 * np.pi)


gmst = greenwich_mean_sidereal_time(inj["geocent_time"])
theta, phi = bilby.core.utils.conversion.ra_dec_to_theta_phi(inj["ra"], inj["dec"], gmst)
if SKY_FRAME == "detector":
    zenith, azimuth = theta_phi_to_zenith_azimuth(theta, phi, interferometers)
else:
    zenith, azimuth = None, None

fiducial_aug = add_ur_derived_parameters(
    dict(
        chirp_mass=Mc_inj,
        mass_ratio=q_inj,
        luminosity_distance=float(inj["luminosity_distance"]),
        H0_sample=H0_TRUE,
        delta_a0=DELTA_TRUE,
        delta_a1=DELTA_TRUE,
        delta_a2=DELTA_TRUE,
        chi_1=float(inj.get("chi_1", 0.0)),
        chi_2=float(inj.get("chi_2", 0.0)),
        theta_jn=float(inj["theta_jn"]),
        psi=float(inj["psi"]),
        phase=float(inj.get("phase", 0.0)),
        geocent_time=float(inj["geocent_time"]),
        zenith=float(zenith) if zenith is not None else None,
        azimuth=float(azimuth) if azimuth is not None else None,
        ra=float(inj["ra"]),
        dec=float(inj["dec"]),
    )
)

fiducial_parameters = dict(
    chirp_mass=float(fiducial_aug["chirp_mass"]),
    mass_ratio=float(fiducial_aug["mass_ratio"]),
    luminosity_distance=float(fiducial_aug["luminosity_distance"]),
    H0_sample=float(fiducial_aug["H0_sample"]),
    delta_a0=float(fiducial_aug["delta_a0"]),
    delta_a1=float(fiducial_aug["delta_a1"]),
    delta_a2=float(fiducial_aug["delta_a2"]),
    lambda_1=float(np.asarray(fiducial_aug["lambda_1"])),
    lambda_2=float(np.asarray(fiducial_aug["lambda_2"])),
    lambda_tilde=float(np.asarray(fiducial_aug["lambda_tilde"])),
    delta_lambda_tilde=float(np.asarray(fiducial_aug["delta_lambda_tilde"])),
    chi_1=float(fiducial_aug.get("chi_1", 0.0)),
    chi_2=float(fiducial_aug.get("chi_2", 0.0)),
    theta_jn=float(fiducial_aug["theta_jn"]),
    psi=float(fiducial_aug["psi"]),
    phase=float(fiducial_aug.get("phase", 0.0)),
    geocent_time=float(fiducial_aug["geocent_time"]),
    ra=float(fiducial_aug["ra"]),
    dec=float(fiducial_aug["dec"]),
)
if SKY_FRAME == "detector":
    fiducial_parameters["zenith"] = float(fiducial_aug["zenith"])
    fiducial_parameters["azimuth"] = float(fiducial_aug["azimuth"])
    fiducial_parameters.pop("ra", None)
    fiducial_parameters.pop("dec", None)
else:
    fiducial_parameters.pop("zenith", None)
    fiducial_parameters.pop("azimuth", None)

likelihood = bilby.gw.likelihood.relative.RelativeBinningGravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=wg_rb,
    fiducial_parameters=fiducial_parameters,
    update_fiducial_parameters=False,
    reference_frame=REFERENCE_FRAME,
    time_reference="geocenter",
    distance_marginalization=False,
    phase_marginalization=True,
    time_marginalization=False,
    priors=priors,
    chi=20.0,
    epsilon=0.01,
)


# =========================================================
# 6. Dynesty run
# =========================================================
print(f"=== Starting Dynesty Run for {EVENT_NAME} ===")
print(
    f"ZERO_NOISE={ZERO_NOISE}, WIDEN_MC={WIDEN_MC}, "
    f"DELTA_SIGMA={DELTA_SIGMA}, H0_TRUE={H0_TRUE}, SKY_FRAME={SKY_FRAME}",
    flush=True,
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=NLIVE,
    naccept=60,
    check_point_plot=True,
    check_point_delta_t=1800,
    npool=NPOOL,
    outdir=outdir,
    label=label,
    resume=True,
    print_method="interval-60",
    sample="acceptance-walk",
    injection_parameters=fiducial_parameters,
    conversion_function=convert_eosfit_to_lal_bns,
    result_class=bilby.gw.result.CBCResult,
    save=RESULT_EXTENSION,
)


# =========================================================
# 7. Post-processing after PE
# =========================================================
def ensure_dataframe(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, dict):
        return pd.DataFrame(x)
    return pd.DataFrame(x)


result.save_to_file(overwrite=True, extension=RESULT_EXTENSION, outdir=outdir)
print(
    f"Saved Result with nested samples to: "
    f"{bilby.core.result.result_file_name(outdir, label, extension=RESULT_EXTENSION)}",
    flush=True,
)


def save_corner_and_csv(res, out_label, include_redshift=True):
    post_df = ensure_dataframe(res.posterior)
    post_csv = os.path.join(outdir, f"{out_label}_posterior_augmented.csv")
    post_df.to_csv(post_csv, index=False)

    truth_aug = convert_eosfit_to_lal_bns(fiducial_parameters)

    plot_params = [
        "mass_ratio",
        "chirp_mass",
        "luminosity_distance",
        "redshift_sample",
        "H0_sample",
        "delta_a0",
        "delta_a1",
        "delta_a2",
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
        "mass_1",
        "mass_2",
        "log_likelihood",
    ]
    if not include_redshift and "redshift_sample" in plot_params:
        plot_params.remove("redshift_sample")
    plot_params = [p for p in plot_params if p in post_df.columns]

    label_map = {
        "mass_ratio": r"$q$",
        "chirp_mass": r"$\\mathcal{M}$",
        "luminosity_distance": r"$d_L$",
        "redshift_sample": r"$z$",
        "H0_sample": r"$H_0$",
        "delta_a0": r"$\\delta a_0$",
        "delta_a1": r"$\\delta a_1$",
        "delta_a2": r"$\\delta a_2$",
        "theta_jn": r"$\\theta_{JN}$",
        "psi": r"$\\psi$",
        "chi_1": r"$\\chi_1$",
        "chi_2": r"$\\chi_2$",
        "zenith": r"$\\kappa$",
        "azimuth": r"$\\mathrm{azimuth}$",
        "ra": r"$\\alpha$",
        "dec": r"$\\delta$",
        "geocent_time": r"$t_c$",
        "lambda_tilde": r"$\\tilde{\\Lambda}$",
        "delta_lambda_tilde": r"$\\delta\\tilde{\\Lambda}$",
        "mass_1": r"$m_1$",
        "mass_2": r"$m_2$",
        "log_likelihood": r"$\\log \\mathcal{L}$",
    }

    plot_labels = [label_map[p] for p in plot_params]
    truths = [truth_aug.get(p, np.nan) for p in plot_params]
    plot_df = post_df[plot_params].replace([np.inf, -np.inf], np.nan).dropna().copy()

    plot_res = copy.copy(res)
    plot_res.posterior = plot_df
    fig = plot_res.plot_corner(
        parameters=plot_params,
        labels=plot_labels,
        truths=truths,
        save=False,
        truth_color="red",
        quantiles=[0.16, 0.84],
    )
    corner_path = os.path.join(outdir, f"{out_label}_all_params_corner.png")
    fig.savefig(corner_path, dpi=300, bbox_inches="tight")
    print(f"[*] Saved augmented posterior to: {post_csv}", flush=True)
    print(f"[*] Saved corner plot to: {corner_path}", flush=True)


save_corner_and_csv(result, label, include_redshift=True)


# =========================================================
# 8. Reweight: reuse existing RB setup, posterior samples only
# =========================================================
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

    ok = (
        finite_fraction == 1.0
        and ess_fraction > 1.0e-3
        and max_normalized_weight < 0.99
    )

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
    res.posterior = convert_eosfit_to_lal_bns(res.posterior)
    res.label = RW_LABEL
    if RW_USE_NESTED:
        res.log_evidence += float(logsumexp(np.asarray(lnw, dtype=float)))
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
    f"ZERO_NOISE={ZERO_NOISE}, WIDEN_MC={WIDEN_MC}, DELTA_SIGMA={DELTA_SIGMA}, "
    f"SKY_FRAME={SKY_FRAME}, RW_METHOD={RW_METHOD}, RW_USE_NESTED={rw_use_nested_effective}",
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

save_corner_and_csv(rw_result, RW_LABEL, include_redshift=True)
print("Done.", flush=True)
