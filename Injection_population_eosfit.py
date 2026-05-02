#!/usr/bin/env python3
import os
import json
import math
import platform
import numpy as np
import bilby
from bilby.core.utils.random import seed as bilby_seed
from astropy.cosmology import Planck18

# tqdm progress (fallback if not installed)
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# =========================
# User knobs (match BNSv5 setup)
# =========================
OUTDIR = "outdir_population_exactfd"
LABEL = "bns_pop_inj_exactfd"

TARGET_N = 100
SNR_THRESH = 30.0

# population requirements (adjust to your figure)
Z_MAX = 0.3
GAMMA_MERGER_RATE = 0.0  # p(z) ∝ dVc/dz * (1+z)^{gamma-1}

M_MIN = 1.1
M_MAX = 2.25
M_MU = 1.33
M_SIGMA = 0.09

CHI1 = 0.0
CHI2 = 0.0

GEOCENT_TIME_FIXED = 1126259642.413

FMIN = 10.0
FREF = 10.0
BASE_DURATION = 1024
SAMPLING_FREQUENCY = 8192
IFO_FILES = ["CE_B20.interferometer", "CE_A40.interferometer"]

PROGRESS_EVERY = 50

# exact waveform-save settings
SAVE_EXACT_SIGNAL = True
SIGNAL_FILENAME = "bns_signal_exact_fd.npz"
SIGNAL_FLOAT_DTYPE = np.float64
SIGNAL_COMPLEX_DTYPE = np.complex128

# reproducibility:
# - controls population sampling RNG
# - noise_seed per trial = BASE_SEED + trial_index
BASE_SEED = int(os.environ.get("BILBY_SEED", "123"))
RNG_POP = np.random.default_rng(BASE_SEED)

bilby.core.utils.setup_logger(outdir=OUTDIR, label=LABEL)
os.makedirs(OUTDIR, exist_ok=True)

accepted_path = os.path.join(OUTDIR, "accepted.jsonl")
config_path = os.path.join(OUTDIR, "catalog_config.json")

# =========================
# EOS-motivated lambda(M) fit -> lambda_1, lambda_2
# Mbar = M / Msun, and masses here are already in solar-mass units.
# For injections we use the best-fit coefficients directly, with all deltas set to 0.
# =========================
LAMBDA_FIT_NORM = 3500.0
A0_FIT = -0.30781804
A1_FIT = 0.79244108
A2_FIT = -0.51480556

DELTA_A0 = 0.0
DELTA_A1 = 0.0
DELTA_A2 = 0.0


def Lambda_of_m_phys(
    m_phys: float,
    delta_a0: float = DELTA_A0,
    delta_a1: float = DELTA_A1,
    delta_a2: float = DELTA_A2,
) -> float:
    """
    EOS-motivated fit:
        lambda(M) = 3500 / Mbar^5 * [1
                                      + a0 (1 + delta_a0)
                                      + a1 (1 + delta_a1) Mbar
                                      + a2 (1 + delta_a2) Mbar^2]

    where Mbar = M / Msun. Since m_phys is already given in solar masses,
    Mbar = m_phys numerically.
    """
    mbar = float(m_phys)
    if not np.isfinite(mbar) or mbar <= 0.0:
        return 0.0

    poly = (
        1.0
        + A0_FIT * (1.0 + delta_a0)
        + A1_FIT * (1.0 + delta_a1) * mbar
        + A2_FIT * (1.0 + delta_a2) * mbar**2
    )
    Lam = LAMBDA_FIT_NORM * poly / mbar**5
    return float(max(Lam, 0.0))


# =========================
# Sampling helpers
# =========================
def sample_trunc_gaussian(mu: float, sigma: float, low: float, high: float, rng: np.random.Generator) -> float:
    while True:
        x = float(rng.normal(mu, sigma))
        if low <= x <= high:
            return x


def make_z_sampler(z_max: float, gamma: float, ngrid: int = 6000):
    z_grid = np.linspace(0.0, z_max, ngrid)
    dVc_dz = (4.0 * np.pi) * Planck18.differential_comoving_volume(z_grid).value  # Mpc^3
    pdf = dVc_dz * (1.0 + z_grid) ** (gamma - 1.0)

    dz = np.diff(z_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dz)])
    cdf /= cdf[-1]

    def _sample_one(rng: np.random.Generator) -> float:
        u = float(rng.random())
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return float(np.interp(u, cdf, z_grid))

    return _sample_one


sample_z = make_z_sampler(Z_MAX, GAMMA_MERGER_RATE)


def sample_isotropic_sky(rng: np.random.Generator):
    ra = float(rng.uniform(0.0, 2.0 * np.pi))
    u = float(rng.uniform(-1.0, 1.0))
    dec = float(np.arcsin(u))
    return ra, dec


def sample_isotropic_orientation(rng: np.random.Generator):
    cosi = float(rng.uniform(-1.0, 1.0))
    theta_jn = float(np.arccos(cosi))
    psi = float(rng.uniform(0.0, np.pi))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    return theta_jn, psi, phase


# =========================
# Duration rule (MATCH bilby.check_signal_duration)
# =========================
def next_pow2_geq(x: float) -> int:
    x = float(max(1.0, x))
    return int(2 ** math.ceil(math.log2(x)))


def duration_from_time_to_merger(m1_det: float, m2_det: float, flow: float) -> tuple[int, float]:
    """
    Bilby check_signal_duration does:
      deltaT = gwutils.calculate_time_to_merger(frequency=self.minimum_frequency, mass_1, mass_2)
      deltaT = np.round(deltaT, 1)
      if deltaT > self.duration: raise
    We replicate that and then round UP to nearest power of 2 for FFT friendliness.
    """
    deltaT = bilby.gw.utils.calculate_time_to_merger(
        frequency=float(flow),
        mass_1=float(m1_det),
        mass_2=float(m2_det),
    )
    deltaT = float(np.round(deltaT, 1))
    dur = max(BASE_DURATION, next_pow2_geq(deltaT))
    return int(dur), float(deltaT)


# =========================
# Waveform generator cache (per duration)
# =========================
waveform_arguments = dict(
    waveform_approximant="IMRPhenomXP_NRTidalv3",
    reference_frequency=FREF,
    minimum_frequency=FMIN,
)

_waveform_generator_cache = {}


def get_waveform_generator(duration_s: int):
    key = (int(duration_s), float(SAMPLING_FREQUENCY))
    if key in _waveform_generator_cache:
        return _waveform_generator_cache[key]
    wg = bilby.gw.WaveformGenerator(
        duration=int(duration_s),
        sampling_frequency=float(SAMPLING_FREQUENCY),
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments,
    )
    _waveform_generator_cache[key] = wg
    return wg


def build_detector_signals_fd(waveform_generator, injection_parameters, interferometers):
    pols = waveform_generator.frequency_domain_strain(parameters=injection_parameters)
    wg_freqs = np.asarray(waveform_generator.frequency_array)

    detector_signals = {}
    for ifo in interferometers:
        ifo_freqs = np.asarray(ifo.frequency_array)
        if wg_freqs.shape != ifo_freqs.shape or not np.allclose(wg_freqs, ifo_freqs, rtol=0.0, atol=0.0):
            raise RuntimeError(f"Frequency grid mismatch for {ifo.name}")
        detector_signals[ifo.name] = np.asarray(
            ifo.get_detector_response(pols, injection_parameters),
            dtype=SIGNAL_COMPLEX_DTYPE,
        )
    return wg_freqs.astype(SIGNAL_FLOAT_DTYPE), detector_signals


def inject_detector_signals_into_data(interferometers, detector_signals):
    for ifo in interferometers:
        ifo.strain_data.frequency_domain_strain += detector_signals[ifo.name]


def save_detector_signals_fd(evdir, frequency_array, detector_signals, duration_s, start_time_s):
    save_dict = {
        "frequency_array": np.asarray(frequency_array, dtype=SIGNAL_FLOAT_DTYPE),
        "duration": np.array([float(duration_s)], dtype=np.float64),
        "start_time": np.array([float(start_time_s)], dtype=np.float64),
    }
    for ifo_name, h_ifo in detector_signals.items():
        save_dict[f"h_{ifo_name}"] = np.asarray(h_ifo, dtype=SIGNAL_COMPLEX_DTYPE)
    np.savez_compressed(os.path.join(evdir, SIGNAL_FILENAME), **save_dict)


# =========================
# IFOs (load once)
# =========================
interferometers = bilby.gw.detector.InterferometerList(
    [bilby.gw.detector.load_interferometer(f) for f in IFO_FILES]
)
for ifo in interferometers:
    ifo.minimum_frequency = FMIN


# =========================
# Save global config
# =========================
catalog_config = dict(
    outdir=OUTDIR,
    label=LABEL,
    target_n=TARGET_N,
    snr_thresh=SNR_THRESH,
    base_seed=BASE_SEED,
    z_max=Z_MAX,
    gamma_merger_rate=GAMMA_MERGER_RATE,
    mass_prior=dict(mu=M_MU, sigma=M_SIGMA, mmin=M_MIN, mmax=M_MAX),
    chi_1=CHI1,
    chi_2=CHI2,
    geocent_time_fixed=GEOCENT_TIME_FIXED,
    base_duration=BASE_DURATION,
    sampling_frequency=SAMPLING_FREQUENCY,
    fmin=FMIN,
    fref=FREF,
    ifo_files=IFO_FILES,
    waveform_arguments=waveform_arguments,
    duration_rule="duration = max(base_duration, next_pow2(round(calculate_time_to_merger(flow, m1_det, m2_det), 0.1s)))",
    universal_relation=dict(
        parameterization="lambda(Mbar) = 3500 / Mbar^5 * [1 + a0(1+delta_a0) + a1(1+delta_a1) Mbar + a2(1+delta_a2) Mbar^2]",
        mass_variable="Mbar = M / Msun",
        norm=LAMBDA_FIT_NORM,
        coefficients=dict(a0=A0_FIT, a1=A1_FIT, a2=A2_FIT),
        deltas=dict(delta_a0=DELTA_A0, delta_a1=DELTA_A1, delta_a2=DELTA_A2),
    ),
    exact_signal_storage=dict(
        enabled=SAVE_EXACT_SIGNAL,
        filename=SIGNAL_FILENAME,
        domain="frequency",
        frame="detector_response",
        float_dtype=str(np.dtype(SIGNAL_FLOAT_DTYPE)),
        complex_dtype=str(np.dtype(SIGNAL_COMPLEX_DTYPE)),
        note="Saved arrays are the exact detector-frame frequency-domain signals injected into the strain data.",
    ),
    software=dict(
        python=platform.python_version(),
        bilby=getattr(bilby, "__version__", "unknown"),
        numpy=np.__version__,
    ),
)
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(catalog_config, f, indent=2, sort_keys=True)


# =========================
# Main loop (meta + exact detector-frame FD signal saving)
# =========================
n_trials = 0
n_accept = 0

pbar = None
if tqdm is not None:
    pbar = tqdm(
        total=TARGET_N,
        desc="Accepted injections (exact FD signal)",
        unit="ev",
        dynamic_ncols=True,
    )

with open(accepted_path, "w", encoding="utf-8") as fj:
    while n_accept < TARGET_N:
        n_trials += 1

        # ---- sample population parameters (source frame) ----
        m1s = sample_trunc_gaussian(M_MU, M_SIGMA, M_MIN, M_MAX, RNG_POP)
        m2s = sample_trunc_gaussian(M_MU, M_SIGMA, M_MIN, M_MAX, RNG_POP)
        if m2s > m1s:
            m1s, m2s = m2s, m1s

        z = sample_z(RNG_POP)
        dL_mpc = float(Planck18.luminosity_distance(z).value)

        # ---- detector-frame masses ----
        m1d = m1s * (1.0 + z)
        m2d = m2s * (1.0 + z)

        # ---- tidal lambdas from source masses ----
        lambda1 = Lambda_of_m_phys(m1s)
        lambda2 = Lambda_of_m_phys(m2s)

        # ---- angles ----
        ra, dec = sample_isotropic_sky(RNG_POP)
        theta_jn, psi, phase = sample_isotropic_orientation(RNG_POP)

        injection_parameters = dict(
            mass_1=m1d,
            mass_2=m2d,
            chi_1=CHI1,
            chi_2=CHI2,
            luminosity_distance=dL_mpc,
            theta_jn=theta_jn,
            psi=psi,
            phase=phase,
            geocent_time=GEOCENT_TIME_FIXED,
            ra=ra,
            dec=dec,
            lambda_1=lambda1,
            lambda_2=lambda2,
        )

        # ---- per-trial deterministic noise seed ----
        noise_seed = BASE_SEED + n_trials

        # ---- choose duration that bilby will accept (MATCH check_signal_duration) ----
        duration_this, deltaT_inband = duration_from_time_to_merger(m1d, m2d, FMIN)
        start_time_this = float(GEOCENT_TIME_FIXED + 2 - duration_this)

        waveform_generator = get_waveform_generator(duration_this)

        # ---- generate a new noise realization ----
        bilby_seed(noise_seed)
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=duration_this,
            start_time=start_time_this,
        )
        for ifo in interferometers:
            ifo.minimum_frequency = FMIN

        # ---- build exact detector-frame FD signal and inject that exact array into data ----
        frequency_array, detector_signals = build_detector_signals_fd(
            waveform_generator=waveform_generator,
            injection_parameters=injection_parameters,
            interferometers=interferometers,
        )
        inject_detector_signals_into_data(interferometers, detector_signals)

        # ---- matched-filter SNR using the exact injected detector-frame signal ----
        ifo_snrs = {}
        for ifo in interferometers:
            sig = detector_signals[ifo.name]
            snr_cplx = bilby.gw.utils.matched_filter_snr(
                signal=sig,
                frequency_domain_strain=ifo.strain_data.frequency_domain_strain,
                power_spectral_density=ifo.power_spectral_density_array,
                duration=duration_this,
            )
            ifo_snrs[ifo.name] = float(np.abs(snr_cplx))

        network_snr = float(np.sqrt(np.sum([s**2 for s in ifo_snrs.values()])))

        if (n_trials % PROGRESS_EVERY) == 0:
            acc_rate = n_accept / max(n_trials, 1)
            msg = (
                f"[PROG] trials={n_trials} accepted={n_accept} acc={acc_rate:.3f} "
                f"lastSNR={network_snr:.1f} dur={duration_this} deltaT={deltaT_inband:.1f}"
            )
            if tqdm is None:
                print(msg)
            else:
                pbar.set_postfix(trials=n_trials, acc=f"{acc_rate:.3f}", last=f"{network_snr:.1f}", dur=duration_this)

        if network_snr < SNR_THRESH:
            continue

        # ---- accepted: save meta + exact detector-frame FD signal ----
        n_accept += 1
        evname = f"event_{n_accept:04d}"
        evdir = os.path.join(OUTDIR, evname)
        os.makedirs(evdir, exist_ok=True)

        meta = dict(
            event=evname,
            accepted_index=n_accept,
            trial_index=n_trials,
            noise_seed=noise_seed,
            network_snr=network_snr,
            ifo_snrs=ifo_snrs,
            z=z,
            luminosity_distance_mpc=dL_mpc,
            mass_1_source=m1s,
            mass_2_source=m2s,
            mass_1_detector=m1d,
            mass_2_detector=m2d,
            lambda_1=lambda1,
            lambda_2=lambda2,
            injection_parameters=injection_parameters,
            duration_check=dict(
                method="bilby.gw.utils.calculate_time_to_merger + round(0.1s) + next_pow2",
                deltaT_inband_rounded_s=deltaT_inband,
                fmin_used_hz=FMIN,
            ),
            data_segment=dict(
                duration=duration_this,
                start_time=start_time_this,
                sampling_frequency=SAMPLING_FREQUENCY,
                fmin=FMIN,
                fref=FREF,
            ),
            exact_signal_file=dict(
                filename=SIGNAL_FILENAME,
                domain="frequency",
                frame="detector_response",
                frequency_array_key="frequency_array",
                detector_signal_keys={ifo.name: f"h_{ifo.name}" for ifo in interferometers},
                float_dtype=str(np.dtype(SIGNAL_FLOAT_DTYPE)),
                complex_dtype=str(np.dtype(SIGNAL_COMPLEX_DTYPE)),
            ),
            reproduction=dict(
                ifo_files=IFO_FILES,
                waveform_arguments=waveform_arguments,
                cosmology="Planck18",
                base_seed=BASE_SEED,
            ),
            software=dict(
                python=platform.python_version(),
                bilby=getattr(bilby, "__version__", "unknown"),
                numpy=np.__version__,
            ),
        )

        with open(os.path.join(evdir, "meta.json"), "w", encoding="utf-8") as fmeta:
            json.dump(meta, fmeta, indent=2, sort_keys=True)

        if SAVE_EXACT_SIGNAL:
            save_detector_signals_fd(
                evdir=evdir,
                frequency_array=frequency_array,
                detector_signals=detector_signals,
                duration_s=duration_this,
                start_time_s=start_time_this,
            )

        fj.write(json.dumps(meta) + "\n")
        fj.flush()

        if tqdm is None:
            print(
                f"[ACCEPT] {evname} netSNR={network_snr:.2f} trials={n_trials} "
                f"noise_seed={noise_seed} dur={duration_this} deltaT={deltaT_inband:.1f}"
            )
        else:
            pbar.update(1)
            acc_rate = n_accept / max(n_trials, 1)
            pbar.set_postfix(trials=n_trials, acc=f"{acc_rate:.3f}", last=f"{network_snr:.1f}", dur=duration_this)
            tqdm.write(
                f"[ACCEPT] {evname} netSNR={network_snr:.2f} trials={n_trials} "
                f"noise_seed={noise_seed} dur={duration_this} deltaT={deltaT_inband:.1f}"
            )

if pbar is not None:
    pbar.close()

print(f"[DONE] trials={n_trials}, accepted={n_accept}, outdir={OUTDIR}")
