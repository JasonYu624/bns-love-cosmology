#!/usr/bin/env python3
import json
import os

import bilby
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, Planck18
from bilby.core.utils.random import seed as bilby_seed
from bilby_cython.geometry import rotation_matrix_from_delta
from bilby_cython.time import greenwich_mean_sidereal_time
from scipy.special import logsumexp

import copy

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


# =========================================================
# 0. Inputs
# =========================================================
POP_OUTDIR = os.environ.get("POP_OUTDIR", "outdir_population_exactfd")
EVENT_INDEX = int(os.environ.get("EVENT_INDEX", "1"))
EVENT_NAME = f"event_{EVENT_INDEX:04d}"

OUTDIR = os.environ.get("OUTDIR", "outdir_population_run_test")
LABEL = os.environ.get("LABEL", f"bns_{EVENT_NAME}_UR_eosfit_exactsignal_Mc0.1")
RW_LABEL = os.environ.get("RW_LABEL", LABEL + "_reweighted_full")

META_PATH = os.path.join(POP_OUTDIR, EVENT_NAME, "meta.json")
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

inj = dict(meta["injection_parameters"])
seg = meta["data_segment"]
noise_seed = int(meta["noise_seed"])

exact_info = meta["exact_signal_file"]
SIGNAL_PATH = os.path.join(POP_OUTDIR, EVENT_NAME, exact_info["filename"])
RESULT_CANDIDATES = [
    os.path.join(OUTDIR, f"{LABEL}_result.hdf5"),
    os.path.join(OUTDIR, f"{LABEL}_result.h5"),
    os.path.join(OUTDIR, f"{LABEL}_result.json"),
    os.path.join(OUTDIR, f"{LABEL}_result.pkl"),
]


def pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


RESULT_FILE = pick_existing(RESULT_CANDIDATES)
if RESULT_FILE is None:
    raise FileNotFoundError(
        "Cannot find PE result file. Tried:\n  " + "\n  ".join(RESULT_CANDIDATES)
    )

bilby.core.utils.setup_logger(outdir=OUTDIR, label=RW_LABEL)

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

ZERO_NOISE = os.environ.get("ZERO_NOISE", "0") == "1"
WIDEN_MC = float(os.environ.get("WIDEN_MC", "0.1"))
UPDATE_FIDUCIAL = os.environ.get("UPDATE_FIDUCIAL", "0") == "1"
Q_PRIOR = os.environ.get("Q_PRIOR", "uniform_reflected").lower()

H0_MIN = float(os.environ.get("H0_MIN", "10.0"))
H0_MAX = float(os.environ.get("H0_MAX", "140.0"))
DELTA_SIGMA = float(os.environ.get("DELTA_SIGMA", "2.0"))
PRIOR_FILE = os.environ.get("PRIOR_FILE", "BNS_spin.prior")
NPOOL = int(os.environ.get("BILBY_RW_NPOOL", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
N_CHECKPOINT = int(os.environ.get("RW_N_CHECKPOINT", "200"))
RW_METHOD = os.environ.get("RW_METHOD", "weighted").strip().lower()
USE_NESTED_SAMPLES = os.environ.get("RW_USE_NESTED_SAMPLES", "1") == "1"
CLEAN_RESUME = os.environ.get("RW_CLEAN_RESUME", "0") == "1"
RESUME_FILE = os.environ.get("RW_RESUME_FILE", os.path.join(OUTDIR, f"{RW_LABEL}_weights_resume_test_Mc0.1.npz"))
H0_TRUE = float(os.environ.get("H0_TRUE", str(Planck18.H0.value)))
if RW_METHOD not in {"rejection", "weighted"}:
    raise ValueError(f"Unsupported RW_METHOD={RW_METHOD}. Choose from rejection, weighted.")


# =========================================================
# 1. EOS-fit UR helpers
# =========================================================
LAMBDA_FIT_NORM = 3500.0
A0_FIT = -0.51480556
A1_FIT = 0.79244108
A2_FIT = -0.30781804
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


def add_ur_derived_parameters(parameters):
    p = dict(parameters)

    dL = np.asarray(p["luminosity_distance"], dtype=float)
    H0 = np.asarray(p.get("H0_sample", H0_TRUE), dtype=float)
    delta_a0 = np.asarray(p.get("delta_a0", DELTA_TRUE), dtype=float)
    delta_a1 = np.asarray(p.get("delta_a1", DELTA_TRUE), dtype=float)
    delta_a2 = np.asarray(p.get("delta_a2", DELTA_TRUE), dtype=float)

    if "mass_1" not in p:
        q = np.asarray(p["mass_ratio"], dtype=float)
        Mc = np.asarray(p["chirp_mass"], dtype=float)
        p["mass_1"] = Mc * (1.0 + q) ** (1.0 / 5.0) * q ** (-3.0 / 5.0)
        p["mass_2"] = np.asarray(p["mass_1"], dtype=float) * q

    z = z_from_dL_H0_vec(dL, H0)
    p["redshift_sample"] = z

    m1_src = np.asarray(p["mass_1"], dtype=float) / (1.0 + z)
    m2_src = np.asarray(p["mass_2"], dtype=float) / (1.0 + z)
    p["mass_1_source"] = m1_src
    p["mass_2_source"] = m2_src

    p["lambda_1"] = lambda_of_mbar_vec(m1_src, delta_a0, delta_a1, delta_a2)
    p["lambda_2"] = lambda_of_mbar_vec(m2_src, delta_a0, delta_a1, delta_a2)

    p["lambda_tilde"] = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(
        p["lambda_1"], p["lambda_2"], p["mass_1"], p["mass_2"]
    )
    p["delta_lambda_tilde"] = bilby.gw.conversion.lambda_1_lambda_2_to_delta_lambda_tilde(
        p["lambda_1"], p["lambda_2"], p["mass_1"], p["mass_2"]
    )
    return p


def _preserve_original_keys_after_conversion(converted, original, augmented):
    converted = dict(converted)
    for src in (original, augmented):
        for key, value in src.items():
            converted.setdefault(key, value)
    converted.setdefault("phase", original.get("phase", 0.0))
    return converted


def convert_for_waveform(parameters, *args, **kwargs):
    original = dict(parameters)
    augmented = add_ur_derived_parameters(original)
    out = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters(augmented)
    if isinstance(out, tuple):
        converted, added_keys = out
    else:
        converted, added_keys = out, []
    converted = _preserve_original_keys_after_conversion(converted, original, augmented)
    return converted, added_keys


# =========================================================
# 2. Exact-signal data helpers
# =========================================================
def load_exact_detector_signals(signal_path, exact_info):
    with np.load(signal_path) as data:
        frequency_array = np.asarray(data[exact_info["frequency_array_key"]])
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
        if not np.array_equal(np.asarray(ifo.frequency_array), EXACT_FREQUENCY_ARRAY):
            raise RuntimeError(f"Frequency grid mismatch for {ifo.name}")
        ifo.strain_data.frequency_domain_strain += EXACT_DETECTOR_SIGNALS[ifo.name]
    return ifos


# =========================================================
# 3. Priors and fiducial
# =========================================================
def make_priors():
    p = bilby.gw.prior.BNSPriorDict(aligned_spin=True, filename=PRIOR_FILE)
    p["phase"] = bilby.core.prior.Uniform(
        minimum=0.0,
        maximum=2.0 * np.pi,
        boundary="periodic",
        name="phase",
    )
    p.pop("ra", None)
    p.pop("dec", None)
    p["zenith"] = bilby.core.prior.Sine(name="zenith", latex_label="$\\kappa$")
    p["azimuth"] = bilby.core.prior.Uniform(
        minimum=0.0,
        maximum=2.0 * np.pi,
        boundary="periodic",
        name="azimuth",
    )

    m1_det = float(meta["mass_1_detector"])
    m2_det = float(meta["mass_2_detector"])
    Mc_inj = (m1_det * m2_det) ** (3.0 / 5.0) / (m1_det + m2_det) ** (1.0 / 5.0)

    p["chirp_mass"] = bilby.core.prior.Uniform(
        minimum=Mc_inj - WIDEN_MC,
        maximum=Mc_inj + WIDEN_MC,
        name="chirp_mass",
        latex_label="$\\mathcal{M}$",
    )

    if Q_PRIOR == "uniform":
        p["mass_ratio"] = bilby.core.prior.Uniform(
            minimum=0.5,
            maximum=1.0,
            name="mass_ratio",
            latex_label="$q$",
        )
    elif Q_PRIOR == "uniform_reflected":
        p["mass_ratio"] = UniformReflected(
            minimum=0.5,
            maximum=1.0,
            name="mass_ratio",
            latex_label="$q$",
        )
    else:
        raise ValueError(f"Unknown Q_PRIOR={Q_PRIOR}")

    p["geocent_time"] = bilby.core.prior.Uniform(
        minimum=inj["geocent_time"] - 0.05,
        maximum=inj["geocent_time"] + 0.05,
        name="geocent_time",
    )
    p["H0_sample"] = bilby.core.prior.Uniform(
        minimum=H0_MIN,
        maximum=H0_MAX,
        name="H0_sample",
        latex_label="$H_0$",
    )
    p["delta_a0"] = bilby.core.prior.Gaussian(mu=0.0, sigma=DELTA_SIGMA, name="delta_a0", latex_label="$\\delta a_0$")
    p["delta_a1"] = bilby.core.prior.Gaussian(mu=0.0, sigma=DELTA_SIGMA, name="delta_a1", latex_label="$\\delta a_1$")
    p["delta_a2"] = bilby.core.prior.Gaussian(mu=0.0, sigma=DELTA_SIGMA, name="delta_a2", latex_label="$\\delta a_2$")

    for key in [
        "lambda_tilde",
        "delta_lambda_tilde",
        "lambda_00",
        "lambda_01",
        "lambda_02",
        "lambda_1",
        "lambda_2",
        "redshift_sample",
        "mass_1_source",
        "mass_2_source",
    ]:
        p.pop(key, None)

    p.conversion_function = add_ur_derived_parameters
    return p


def get_zenith_azimuth(theta, phi, ifos):
    rot = rotation_matrix_from_delta(ifos[0].vertex - ifos[1].vertex)
    rotated = rot.T @ np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    zen = np.arccos(rotated[2])
    az = np.arctan2(rotated[1], rotated[0])
    return zen, np.mod(az, 2.0 * np.pi)


m1_det = float(meta["mass_1_detector"])
m2_det = float(meta["mass_2_detector"])
Mc_inj = (m1_det * m2_det) ** (3.0 / 5.0) / (m1_det + m2_det) ** (1.0 / 5.0)
q_inj = min(m1_det, m2_det) / max(m1_det, m2_det)

ifos_for_fid = build_interferometers_with_exact_data(zero_noise=ZERO_NOISE)
gmst = greenwich_mean_sidereal_time(inj["geocent_time"])
theta, phi = bilby.core.utils.conversion.ra_dec_to_theta_phi(inj["ra"], inj["dec"], gmst)
zenith, azimuth = get_zenith_azimuth(theta, phi, ifos_for_fid)

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
        zenith=float(zenith),
        azimuth=float(azimuth),
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
    zenith=float(fiducial_aug["zenith"]),
    azimuth=float(fiducial_aug["azimuth"]),
)


# =========================================================
# 4. Build likelihoods and reweight
# =========================================================
print(f"Reloading PE result from: {RESULT_FILE}", flush=True)
result = bilby.core.result.read_in_result(filename=RESULT_FILE)
result.posterior = pd.DataFrame(result.posterior) if isinstance(result.posterior, dict) else result.posterior.copy()
if getattr(result, "nested_samples", None) is not None:
    result.nested_samples = pd.DataFrame(result.nested_samples) if isinstance(result.nested_samples, dict) else result.nested_samples.copy()

if USE_NESTED_SAMPLES:
    if getattr(result, "nested_samples", None) is None:
        raise ValueError("RW_USE_NESTED_SAMPLES=1 but result.nested_samples is None")
    if "weights" in result.nested_samples.columns:
        w = np.asarray(result.nested_samples["weights"], dtype=float)
        good = np.isfinite(w) & (w > 0.0)
        result.nested_samples = result.nested_samples.loc[good].reset_index(drop=True).copy()

ifos_rw = build_interferometers_with_exact_data(zero_noise=ZERO_NOISE)

wg_rb = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star_relative_binning,
    parameter_conversion=convert_for_waveform,
    waveform_arguments=waveform_arguments_rb,
)
wg_full = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=convert_for_waveform,
    waveform_arguments=waveform_arguments_full,
)

priors_rb = make_priors()
old_likelihood = bilby.gw.likelihood.relative.RelativeBinningGravitationalWaveTransient(
    interferometers=ifos_rw,
    waveform_generator=wg_rb,
    fiducial_parameters=fiducial_parameters,
    update_fiducial_parameters=UPDATE_FIDUCIAL,
    reference_frame=ifos_rw,
    time_reference="geocenter",
    distance_marginalization=False,
    phase_marginalization=True,
    time_marginalization=False,
    priors=priors_rb,
    chi=20.0,
    epsilon=0.01,
)

priors_full = make_priors()
new_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos_rw,
    waveform_generator=wg_full,
    reference_frame=ifos_rw,
    time_reference="geocenter",
    distance_marginalization=False,
    phase_marginalization=True,
    time_marginalization=False,
    priors=priors_full,
)

DIAG_RESUME_FILE = (
    RESUME_FILE.replace(".npz", "_diag.npz")
    if RESUME_FILE.endswith(".npz")
    else RESUME_FILE + "_diag"
)

def compute_reweight_diagnostics(result_obj, use_nested_samples):
    result_for_weights = copy.copy(result_obj)

    if use_nested_samples:
        if getattr(result_for_weights, "nested_samples", None) is None:
            raise ValueError("Requested nested-sample reweight, but nested_samples is missing")
        result_for_weights.posterior = result_for_weights.nested_samples.copy()
    else:
        result_for_weights.posterior = result_for_weights.posterior.copy()

    if CLEAN_RESUME and os.path.exists(DIAG_RESUME_FILE):
        os.remove(DIAG_RESUME_FILE)

    lnw, new_ll, new_lp, old_ll, old_lp = bilby.core.result.get_weights_for_reweighting(
        result=result_for_weights,
        new_likelihood=new_likelihood,
        old_likelihood=old_likelihood,
        old_prior=None,
        new_prior=None,
        resume_file=DIAG_RESUME_FILE,
        n_checkpoint=N_CHECKPOINT,
        npool=NPOOL,
    )

    if use_nested_samples:
        lnw = lnw + np.log(result_for_weights.posterior["weights"].to_numpy())

    finite = np.isfinite(lnw)
    finite_fraction = float(np.mean(finite))

    if finite.sum() == 0:
        return {
            "ok": False,
            "reason": "no finite log-weights",
            "use_nested_samples": use_nested_samples,
            "finite_fraction": finite_fraction,
            "ess_fraction": 0.0,
            "max_normalized_weight": np.inf,
        }

    lnw_f = lnw[finite]
    lnw_shift = lnw_f - np.max(lnw_f)

    # only for diagnostics; avoids exp underflow warnings
    w_rel = np.exp(np.clip(lnw_shift, -745.0, 0.0))
    sw = np.sum(w_rel)
    if sw == 0.0 or not np.isfinite(sw):
        return {
            "ok": False,
            "reason": "all relative weights underflowed or are non-finite",
            "use_nested_samples": use_nested_samples,
            "finite_fraction": finite_fraction,
            "ess_fraction": 0.0,
            "max_normalized_weight": np.inf,
        }

    w_rel /= sw
    ess = 1.0 / np.sum(w_rel ** 2)
    ess_fraction = float(ess / len(w_rel))
    max_normalized_weight = float(np.max(w_rel))

    # conservative thresholds
    ok = (
        finite_fraction == 1.0
        and ess_fraction > 1.0e-3
        and max_normalized_weight < 0.99
    )

    return {
        "ok": bool(ok),
        "reason": "ok" if ok else "pathological importance weights",
        "use_nested_samples": use_nested_samples,
        "finite_fraction": finite_fraction,
        "num_finite_weights": int(len(w_rel)),
        "ess": float(ess),
        "ess_fraction": ess_fraction,
        "max_normalized_weight": max_normalized_weight,
        "diag_resume_file": DIAG_RESUME_FILE,
    }


if CLEAN_RESUME and os.path.exists(RESUME_FILE):
    os.remove(RESUME_FILE)

print(f"=== Starting bilby reweight for {EVENT_NAME} ===", flush=True)
print(f"RW_LABEL={RW_LABEL}", flush=True)
print(
    f"ZERO_NOISE={ZERO_NOISE}, WIDEN_MC={WIDEN_MC}, Q_PRIOR={Q_PRIOR}, "
    f"UPDATE_FIDUCIAL={UPDATE_FIDUCIAL}, RW_METHOD={RW_METHOD}",
    flush=True,
)

diag = compute_reweight_diagnostics(result, USE_NESTED_SAMPLES)
print("Preflight diagnostics:", json.dumps(diag, indent=2), flush=True)

summary_before_path = os.path.join(OUTDIR, f"{RW_LABEL}_summary_before_reweight.json")
with open(summary_before_path, "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2)

# fallback: nested samples -> posterior samples
if (not diag["ok"]) and USE_NESTED_SAMPLES:
    print("Nested-sample reweight looks pathological; retrying with posterior samples.", flush=True)
    USE_NESTED_SAMPLES = False
    diag = compute_reweight_diagnostics(result, USE_NESTED_SAMPLES)
    print("Posterior-sample diagnostics:", json.dumps(diag, indent=2), flush=True)
    with open(summary_before_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

if not diag["ok"]:
    raise RuntimeError(
        "Reweight aborted before rejection sampling: "
        f"reason={diag['reason']}, "
        f"finite_fraction={diag['finite_fraction']}, "
        f"ess_fraction={diag['ess_fraction']}, "
        f"max_normalized_weight={diag['max_normalized_weight']}"
    )

if RW_METHOD == "rejection":
    rw_result = bilby.core.result.reweight(
        result=result,
        label=RW_LABEL,
        new_likelihood=new_likelihood,
        old_likelihood=old_likelihood,
        old_prior=None,
        new_prior=None,
        resume_file=RESUME_FILE,
        n_checkpoint=N_CHECKPOINT,
        npool=NPOOL,
        verbose_output=False,
        use_nested_samples=USE_NESTED_SAMPLES,
    )
else:
    result_for_weights = copy.copy(result)
    if USE_NESTED_SAMPLES:
        result_for_weights.posterior = result.nested_samples.copy()
    else:
        result_for_weights.posterior = result.posterior.copy()

    lnw, new_ll, new_lp, old_ll, old_lp = bilby.core.result.get_weights_for_reweighting(
        result=result_for_weights,
        new_likelihood=new_likelihood,
        old_likelihood=old_likelihood,
        old_prior=None,
        new_prior=None,
        resume_file=RESUME_FILE,
        n_checkpoint=N_CHECKPOINT,
        npool=NPOOL,
    )
    if USE_NESTED_SAMPLES:
        lnw = np.asarray(lnw, dtype=float) + np.log(np.asarray(result_for_weights.posterior["weights"], dtype=float))

    finite = np.isfinite(lnw)
    if not np.any(finite):
        raise RuntimeError("No finite log-weights for weighted resampling")
    lnw_f = np.asarray(lnw, dtype=float)[finite]
    w_rel = np.exp(np.clip(lnw_f - np.max(lnw_f), -745.0, 0.0))
    w_norm = w_rel / np.sum(w_rel)
    n_eff = int(max(1, round(1.0 / np.sum(w_norm ** 2))))

    posterior = result_for_weights.posterior.copy().iloc[np.where(finite)[0]].copy()
    posterior["log_likelihood"] = np.asarray(new_ll, dtype=float)[finite]
    posterior["log_prior"] = np.asarray(new_lp, dtype=float)[finite]

    rw_posterior = posterior.sample(
        n=n_eff,
        replace=True,
        weights=w_norm,
        random_state=12345 + EVENT_INDEX,
    ).reset_index(drop=True)

    rw_result = copy.copy(result)
    rw_result.label = RW_LABEL
    rw_result.posterior = rw_posterior
    if USE_NESTED_SAMPLES:
        rw_result.log_evidence += float(logsumexp(np.asarray(lnw, dtype=float)))
    else:
        rw_result.log_evidence += float(logsumexp(np.asarray(lnw, dtype=float)) - np.log(len(result_for_weights.posterior)))
    rw_result.meta_data["reweighted_using_importance_resampling"] = True

# =========================================================
# 5. Save outputs
# =========================================================
def ensure_dataframe(x):
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, dict):
        return pd.DataFrame(x)
    return pd.DataFrame(x)


def add_plot_columns(df):
    df = ensure_dataframe(df).copy()
    if len(df) == 0:
        return df
    return ensure_dataframe(add_ur_derived_parameters(df))


rw_result.posterior = ensure_dataframe(rw_result.posterior)
post_rw = add_plot_columns(rw_result.posterior)

csv_path = os.path.join(OUTDIR, f"{RW_LABEL}_posterior_augmented.csv")
post_rw.to_csv(csv_path, index=False)
print(f"Saved augmented reweighted posterior: {csv_path}", flush=True)

summary = {
    "result_file": RESULT_FILE,
    "rw_label": RW_LABEL,
    "rw_method": RW_METHOD,
    "use_nested_samples": USE_NESTED_SAMPLES,
    "n_rb_posterior": int(len(result.posterior)),
    "n_rw_posterior": int(len(post_rw)),
    "resume_file": RESUME_FILE,
}
summary_path = os.path.join(OUTDIR, f"{RW_LABEL}_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary: {summary_path}", flush=True)

import matplotlib.pyplot as plt

truth_aug = add_ur_derived_parameters(
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
        geocent_time=float(inj["geocent_time"]),
        zenith=float(zenith),
        azimuth=float(azimuth),
    )
)

plot_params = [
    "luminosity_distance",
    "H0_sample",
    "redshift_sample",
    "delta_a0",
    "delta_a1",
    "delta_a2",
    "lambda_tilde",
    "delta_lambda_tilde",
]
plot_params = [p for p in plot_params if p in post_rw.columns]
label_map = {
    "luminosity_distance": r"$d_L$",
    "H0_sample": r"$H_0$",
    "redshift_sample": r"$z$",
    "delta_a0": r"$\\delta a_0$",
    "delta_a1": r"$\\delta a_1$",
    "delta_a2": r"$\\delta a_2$",
    "lambda_tilde": r"$\\tilde{\\Lambda}$",
    "delta_lambda_tilde": r"$\\delta\\tilde{\\Lambda}$",
}
plot_df = post_rw[plot_params].replace([np.inf, -np.inf], np.nan).dropna().copy()
rw_result.posterior = plot_df
fig = rw_result.plot_corner(
    parameters=plot_params,
    labels=[label_map[p] for p in plot_params],
    truths=[truth_aug.get(p, np.nan) for p in plot_params],
    save=False,
    truth_color="red",
    quantiles=[0.16, 0.84],
)
corner_path = os.path.join(OUTDIR, f"{RW_LABEL}_h0_dl_z_corner.png")
fig.savefig(corner_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved corner plot: {corner_path}", flush=True)

print("Done.", flush=True)
