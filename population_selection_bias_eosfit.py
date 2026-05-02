#!/usr/bin/env python3
import os
import json
import math
import signal
import sys
import time
import numpy as np
import bilby
from bilby.core.utils.random import seed as bilby_seed
from astropy.cosmology import Planck18

# -------------------------
# Config
# -------------------------
OUTDIR = os.environ.get("OUTDIR", "outdir_selection_calib_eosfit")
LABEL = os.environ.get("LABEL", "bns_sel_calib_eosfit")

SNR_THRESH = float(os.environ.get("SNR_THRESH", "30.0"))

# Selection-calibration mass proposal.
# Kept flat by default to preserve the original script's intent.
MASS_DIST = os.environ.get("MASS_DIST", "flat").lower()
M_MIN = float(os.environ.get("M_MIN", "1.0"))
M_MAX = float(os.environ.get("M_MAX", "1.7"))
M_MU = float(os.environ.get("M_MU", "1.33"))
M_SIGMA = float(os.environ.get("M_SIGMA", "0.09"))

# p(z) ∝ dVc/dz * (1+z)^{gamma-1}
Z_MAX = float(os.environ.get("Z_MAX", "0.3"))
GAMMA = float(os.environ.get("GAMMA", "0.0"))

CHI1 = float(os.environ.get("CHI1", "0.0"))
CHI2 = float(os.environ.get("CHI2", "0.0"))
GEOCENT_TIME_FIXED = float(os.environ.get("GEOCENT_TIME_FIXED", "1126259642.413"))

FMIN = float(os.environ.get("FMIN", "10.0"))
FREF = float(os.environ.get("FREF", "10.0"))
BASE_DURATION = int(os.environ.get("BASE_DURATION", "1024"))
FS = float(os.environ.get("FS", "8192"))
IFO_FILES = os.environ.get("IFO_FILES", "CE_B20.interferometer,CE_A40.interferometer").split(",")

# EOS-motivated lambda(M) fit, aligned with the current injection-population script.
LAMBDA_FIT_NORM = float(os.environ.get("LAMBDA_FIT_NORM", "3500.0"))
A0_FIT = float(os.environ.get("A0_FIT", "-0.30781804"))
A1_FIT = float(os.environ.get("A1_FIT", "0.79244108"))
A2_FIT = float(os.environ.get("A2_FIT", "-0.51480556"))
DELTA_A0 = float(os.environ.get("DELTA_A0", "0.0"))
DELTA_A1 = float(os.environ.get("DELTA_A1", "0.0"))
DELTA_A2 = float(os.environ.get("DELTA_A2", "0.0"))

# Detected events per file
CHUNK = int(os.environ.get("CHUNK", "128"))
PROGRESS_EVERY_DET = int(os.environ.get("PROGRESS_EVERY_DET", "128"))


# -------------------------
# Helpers
# -------------------------
def sample_trunc_gaussian(mu: float, sigma: float, low: float, high: float, rng: np.random.Generator) -> float:
    while True:
        x = float(rng.normal(mu, sigma))
        if low <= x <= high:
            return x


def sample_component_masses(rng: np.random.Generator):
    if MASS_DIST == "flat":
        m1 = float(rng.uniform(M_MIN, M_MAX))
        m2 = float(rng.uniform(M_MIN, M_MAX))
    elif MASS_DIST in {"gaussian", "trunc_gaussian", "truncated_gaussian"}:
        m1 = sample_trunc_gaussian(M_MU, M_SIGMA, M_MIN, M_MAX, rng)
        m2 = sample_trunc_gaussian(M_MU, M_SIGMA, M_MIN, M_MAX, rng)
    else:
        raise ValueError(f"Unknown MASS_DIST={MASS_DIST}")
    if m2 > m1:
        m1, m2 = m2, m1
    return m1, m2


def Lambda_of_m_phys(
    m_phys: float,
    delta_a0: float = DELTA_A0,
    delta_a1: float = DELTA_A1,
    delta_a2: float = DELTA_A2,
) -> float:
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


def make_z_sampler(z_max: float, gamma: float, ngrid: int = 6000):
    z_grid = np.linspace(0.0, z_max, ngrid)
    dVc_dz = (4.0 * np.pi) * Planck18.differential_comoving_volume(z_grid).value
    pdf = dVc_dz * (1.0 + z_grid) ** (gamma - 1.0)
    dz = np.diff(z_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dz)])
    cdf /= cdf[-1]

    def sample_one(rng: np.random.Generator) -> float:
        u = float(rng.random())
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return float(np.interp(u, cdf, z_grid))

    return sample_one


sample_z = make_z_sampler(Z_MAX, GAMMA)


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


def next_pow2_geq(x: float) -> int:
    x = float(max(1.0, x))
    return int(2 ** math.ceil(math.log2(x)))


def duration_from_time_to_merger(m1_det: float, m2_det: float, flow: float):
    deltaT = bilby.gw.utils.calculate_time_to_merger(
        frequency=float(flow), mass_1=float(m1_det), mass_2=float(m2_det)
    )
    deltaT = float(np.round(deltaT, 1))
    dur = max(BASE_DURATION, next_pow2_geq(deltaT))
    return int(dur), deltaT


def make_noise_seed(base_seed: int, worker_id: int, trial_index: int) -> int:
    mod = 2**32 - 5
    return int((base_seed + 1000003 * worker_id + 9176 * trial_index) % mod)


waveform_arguments = dict(
    waveform_approximant="IMRPhenomXP_NRTidalv3",
    reference_frequency=FREF,
    minimum_frequency=FMIN,
    fiducial=1,
)
_wg_cache = {}


def get_waveform_generator(duration_s: int):
    key = (int(duration_s), float(FS))
    if key in _wg_cache:
        return _wg_cache[key]
    wg = bilby.gw.WaveformGenerator(
        duration=int(duration_s),
        sampling_frequency=float(FS),
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments=waveform_arguments,
    )
    _wg_cache[key] = wg
    return wg


def build_detector_signals_fd(waveform_generator, injection_parameters, interferometers):
    pols = waveform_generator.frequency_domain_strain(parameters=injection_parameters)
    wg_freqs = np.asarray(waveform_generator.frequency_array)
    detector_signals = {}
    for ifo in interferometers:
        ifo_freqs = np.asarray(ifo.frequency_array)
        if wg_freqs.shape != ifo_freqs.shape or not np.allclose(wg_freqs, ifo_freqs, rtol=0.0, atol=0.0):
            raise RuntimeError(f"Frequency grid mismatch for {ifo.name}")
        detector_signals[ifo.name] = np.asarray(ifo.get_detector_response(pols, injection_parameters))
    return wg_freqs, detector_signals


def inject_detector_signals_into_data(interferometers, detector_signals):
    for ifo in interferometers:
        ifo.strain_data.frequency_domain_strain += detector_signals[ifo.name]


def infer_worker_info():
    wid = os.environ.get("WORKER_ID")
    if wid is None:
        wid = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_PROCID") or "0"

    nw = os.environ.get("N_WORKERS")
    if nw is None:
        nw = os.environ.get("SLURM_ARRAY_TASK_COUNT") or os.environ.get("SLURM_NTASKS") or "64"

    return int(wid), int(nw)


def main():
    target_success = int(os.environ.get("TARGET_SUCCESS", "1024"))
    base_seed = int(os.environ.get("BILBY_SEED", "123"))

    worker_id, n_workers = infer_worker_info()

    os.makedirs(OUTDIR, exist_ok=True)
    wdir = os.path.join(OUTDIR, f"worker_{worker_id:03d}")
    os.makedirs(wdir, exist_ok=True)

    print(
        f"[w{worker_id:03d}] PWD={os.getcwd()} OUTDIR={os.path.abspath(OUTDIR)} "
        f"TARGET_SUCCESS={target_success} CHUNK={CHUNK} PROGRESS_EVERY_DET={PROGRESS_EVERY_DET} "
        f"MASS_DIST={MASS_DIST}",
        flush=True,
    )

    from numpy.random import SeedSequence, default_rng
    ss = SeedSequence(base_seed)
    child = ss.spawn(n_workers)[worker_id]
    rng = default_rng(child)

    bilby.core.utils.setup_logger(outdir=OUTDIR, label=f"{LABEL}_w{worker_id:03d}")

    ifos = bilby.gw.detector.InterferometerList(
        [bilby.gw.detector.load_interferometer(f) for f in IFO_FILES]
    )
    for ifo in ifos:
        ifo.minimum_frequency = FMIN

    buf = {k: [] for k in [
        "worker_id", "trial_index", "noise_seed",
        "network_snr", "snr_ce_b20", "snr_ce_a40",
        "z", "dL_mpc",
        "m1_src", "m2_src", "m1_det", "m2_det",
        "lambda1", "lambda2",
        "delta_a0", "delta_a1", "delta_a2",
        "ra", "dec", "theta_jn", "psi", "phase",
        "duration", "start_time",
    ]}
    chunk_id = 0
    n_trials = 0
    n_det = 0
    t0 = time.time()

    def flush():
        nonlocal chunk_id
        if len(buf["trial_index"]) == 0:
            return
        path = os.path.join(wdir, f"detected_chunk{chunk_id:04d}.npz")
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in buf.items()})
        for k in buf:
            buf[k].clear()
        chunk_id += 1

    def write_partial_summary(tag: str):
        out = dict(
            worker_id=worker_id,
            n_workers=n_workers,
            n_trials_total=n_trials,
            n_detected=n_det,
            target_success=target_success,
            snr_thresh=SNR_THRESH,
            elapsed_s=float(time.time() - t0),
            tag=tag,
        )
        with open(os.path.join(wdir, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)

    def _handle_term(signum, frame):
        flush()
        write_partial_summary("partial")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    while n_det < target_success:
        n_trials += 1

        m1s, m2s = sample_component_masses(rng)
        z = float(sample_z(rng))
        dL = float(Planck18.luminosity_distance(z).value)

        m1d = m1s * (1.0 + z)
        m2d = m2s * (1.0 + z)

        lambda1 = Lambda_of_m_phys(m1s)
        lambda2 = Lambda_of_m_phys(m2s)

        ra, dec = sample_isotropic_sky(rng)
        theta_jn, psi, phase = sample_isotropic_orientation(rng)

        inj = dict(
            mass_1=m1d,
            mass_2=m2d,
            chi_1=CHI1,
            chi_2=CHI2,
            luminosity_distance=dL,
            theta_jn=theta_jn,
            psi=psi,
            phase=phase,
            geocent_time=GEOCENT_TIME_FIXED,
            ra=ra,
            dec=dec,
            lambda_1=lambda1,
            lambda_2=lambda2,
        )

        duration_this, _dt = duration_from_time_to_merger(m1d, m2d, FMIN)
        start_time_this = float(GEOCENT_TIME_FIXED + 2 - duration_this)
        wg = get_waveform_generator(duration_this)

        noise_seed = make_noise_seed(base_seed, worker_id, n_trials)

        bilby_seed(noise_seed)
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=FS,
            duration=duration_this,
            start_time=start_time_this,
        )
        for ifo in ifos:
            ifo.minimum_frequency = FMIN

        _, detector_signals = build_detector_signals_fd(
            waveform_generator=wg,
            injection_parameters=inj,
            interferometers=ifos,
        )
        inject_detector_signals_into_data(ifos, detector_signals)

        snrs = {}
        for ifo in ifos:
            sig = detector_signals[ifo.name]
            snr_c = bilby.gw.utils.matched_filter_snr(
                signal=sig,
                frequency_domain_strain=ifo.strain_data.frequency_domain_strain,
                power_spectral_density=ifo.power_spectral_density_array,
                duration=duration_this,
            )
            snrs[ifo.name] = float(np.abs(snr_c))

        net = float(np.sqrt(np.sum([v * v for v in snrs.values()])))
        if net < SNR_THRESH:
            continue

        n_det += 1
        buf["worker_id"].append(worker_id)
        buf["trial_index"].append(n_trials)
        buf["noise_seed"].append(noise_seed)
        buf["network_snr"].append(net)
        buf["snr_ce_b20"].append(snrs.get("CE-B20", np.nan))
        buf["snr_ce_a40"].append(snrs.get("CE-A40", np.nan))
        buf["z"].append(z)
        buf["dL_mpc"].append(dL)
        buf["m1_src"].append(m1s)
        buf["m2_src"].append(m2s)
        buf["m1_det"].append(m1d)
        buf["m2_det"].append(m2d)
        buf["lambda1"].append(lambda1)
        buf["lambda2"].append(lambda2)
        buf["delta_a0"].append(DELTA_A0)
        buf["delta_a1"].append(DELTA_A1)
        buf["delta_a2"].append(DELTA_A2)
        buf["ra"].append(ra)
        buf["dec"].append(dec)
        buf["theta_jn"].append(theta_jn)
        buf["psi"].append(psi)
        buf["phase"].append(phase)
        buf["duration"].append(duration_this)
        buf["start_time"].append(start_time_this)

        if (n_det % PROGRESS_EVERY_DET) == 0:
            elapsed = time.time() - t0
            acc = n_det / max(n_trials, 1)
            rate = n_det / max(elapsed, 1e-9)
            print(
                f"[w{worker_id:03d}] det={n_det}/{target_success} "
                f"trials={n_trials} acc={acc:.4f} "
                f"last_net={net:.1f} det_rate={rate:.3f}/s elapsed={elapsed/60:.1f}min",
                flush=True,
            )

        if (n_det % CHUNK) == 0:
            flush()

    flush()

    summary = dict(
        worker_id=worker_id,
        n_workers=n_workers,
        target_success=target_success,
        n_detected=n_det,
        n_trials_total=n_trials,
        snr_thresh=SNR_THRESH,
        base_seed=base_seed,
        ifo_files=IFO_FILES,
        waveform_arguments=waveform_arguments,
        z_max=Z_MAX,
        gamma=GAMMA,
        mass_dist=MASS_DIST,
        mmin=M_MIN,
        mmax=M_MAX,
        m_mu=M_MU,
        m_sigma=M_SIGMA,
        fs=FS,
        fmin=FMIN,
        chunk=CHUNK,
        progress_every_det=PROGRESS_EVERY_DET,
        exact_injection_method="build detector response in frequency domain and add directly to strain",
        universal_relation=dict(
            parameterization="lambda(Mbar) = 3500 / Mbar^5 * [1 + a0(1+delta_a0) + a1(1+delta_a1) Mbar + a2(1+delta_a2) Mbar^2]",
            mass_variable="Mbar = M / Msun",
            norm=LAMBDA_FIT_NORM,
            coefficients=dict(a0=A0_FIT, a1=A1_FIT, a2=A2_FIT),
            deltas=dict(delta_a0=DELTA_A0, delta_a1=DELTA_A1, delta_a2=DELTA_A2),
        ),
        elapsed_s=float(time.time() - t0),
    )
    with open(os.path.join(wdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[DONE worker {worker_id}] detected={n_det} trials={n_trials} out={wdir}", flush=True)


if __name__ == "__main__":
    main()
