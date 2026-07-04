#!/usr/bin/env python3
import argparse
import json
import math
import os
import signal
import sys
import time

LAL_DATA_PATH = "/scratch/gpfs/ANDREASB/lalsuite-extra/lalsuite-waveform-data/waveform_data"
os.environ["LAL_DATA_PATH"] = LAL_DATA_PATH
WAVEFORM_APPROXIMANT = "SEOBNRv5_ROM_NRTidalv3"

import bilby
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck18
from bilby.core.utils.random import seed as bilby_seed


def parse_args():
    p = argparse.ArgumentParser(description="Section 4.3 VT injections")
    p.add_argument("--outdir", type=str, default="outputs/outdir_selection_calib_SEOBNR")
    p.add_argument("--label", type=str, default="bns_sel_calib_SEOBNR")
    p.add_argument("--target-success", type=int, default=1024)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=64)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--progress-every-det", type=int, default=128)
    p.add_argument("--ifo-files", type=str, default="CE_B20.interferometer,CE_A40.interferometer")
    return p.parse_args()


def sample_component_masses(rng):
    m1 = float(rng.uniform(0.8, 1.8))
    m2 = float(rng.uniform(0.8, 1.8))
    if m2 > m1:
        m1, m2 = m2, m1
    return m1, m2


def make_z_sampler(z_max=0.3, gamma=0.0, ngrid=6000):
    z_grid = np.linspace(0.0, z_max, ngrid)
    dvc_dz = (4.0 * np.pi) * Planck18.differential_comoving_volume(z_grid).value
    pdf = dvc_dz * (1.0 + z_grid) ** (gamma - 1.0)
    dz = np.diff(z_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dz)])
    cdf /= cdf[-1]

    def sample_one(rng):
        u = float(rng.random())
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return float(np.interp(u, cdf, z_grid))

    return sample_one


def qz_density(z, z_max=0.3, gamma=0.0):
    z_grid = np.linspace(0.0, z_max, 10000)
    dvc_dz = (4.0 * np.pi) * Planck18.differential_comoving_volume(z_grid).value
    pdf = dvc_dz * (1.0 + z_grid) ** (gamma - 1.0)
    norm = float(np.trapz(pdf, z_grid))
    if z <= 0.0 or z >= z_max:
        return 0.0
    val = (4.0 * np.pi) * Planck18.differential_comoving_volume(z).value * (1.0 + z) ** (gamma - 1.0)
    return float(val / norm)


def sample_isotropic_sky(rng):
    ra = float(rng.uniform(0.0, 2.0 * np.pi))
    dec = float(np.arcsin(rng.uniform(-1.0, 1.0)))
    return ra, dec


def sample_isotropic_orientation(rng):
    theta_jn = float(np.arccos(rng.uniform(-1.0, 1.0)))
    psi = float(rng.uniform(0.0, np.pi))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    return theta_jn, psi, phase


def lambda_of_m_phys(m_phys):
    a0, a1, a2 = -0.30781804, 0.79244108, -0.51480556
    mbar = float(m_phys)
    poly = 1.0 + a0 + a1 * mbar + a2 * mbar**2
    lam = 3500.0 * poly / mbar**5
    return float(max(lam, 0.0))


def next_pow2_geq(x):
    return int(2 ** math.ceil(math.log2(max(1.0, float(x)))))


def duration_from_time_to_merger(m1_det, m2_det, flow=10.0, base_duration=1024):
    delta_t = bilby.gw.utils.calculate_time_to_merger(frequency=float(flow), mass_1=float(m1_det), mass_2=float(m2_det))
    delta_t = float(np.round(delta_t, 1))
    return max(base_duration, next_pow2_geq(delta_t)), delta_t


def make_noise_seed(base_seed, worker_id, trial_index):
    return int((base_seed + 1000003 * worker_id + 9176 * trial_index) % (2**32 - 5))


def ddL_dz_mpc(h0, z, z_max=0.3):
    dz = 1e-5
    z_lo = max(0.0, z - dz)
    z_hi = min(z_max, z + dz)
    if z_hi <= z_lo:
        return np.nan
    cosmo = FlatLambdaCDM(H0=h0, Om0=Planck18.Om0, Tcmb0=Planck18.Tcmb0)
    d_lo = float(cosmo.luminosity_distance(z_lo).value)
    d_hi = float(cosmo.luminosity_distance(z_hi).value)
    return (d_hi - d_lo) / (z_hi - z_lo)


def q_inj_detector_prior(m1s, m2s, z, h0):
    q_m = 2.0 / ((1.8 - 0.8) ** 2)
    q_z = qz_density(z, z_max=0.3, gamma=0.0)
    q_h = 1.0 / (150.0 - 10.0)
    ddz = ddL_dz_mpc(h0, z, z_max=0.3)
    if q_z <= 0.0 or not np.isfinite(ddz) or ddz <= 0.0:
        return 0.0
    jac = 1.0 / ((1.0 + z) ** 2 * ddz)
    return float(q_m * q_z * q_h * jac)


def main():
    args = parse_args()

    sample_z = make_z_sampler(0.3, 0.0)
    chi1_prior = bilby.gw.prior.AlignedSpin(name="chi_1", a_prior=bilby.core.prior.Uniform(minimum=0.0, maximum=0.05))
    chi2_prior = bilby.gw.prior.AlignedSpin(name="chi_2", a_prior=bilby.core.prior.Uniform(minimum=0.0, maximum=0.05))

    os.makedirs(args.outdir, exist_ok=True)
    wdir = os.path.join(args.outdir, f"worker_{args.worker_id:03d}")
    os.makedirs(wdir, exist_ok=True)

    bilby.core.utils.setup_logger(outdir=args.outdir, label=f"{args.label}_w{args.worker_id:03d}")

    from numpy.random import SeedSequence, default_rng

    ss = SeedSequence(args.base_seed)
    rng = default_rng(ss.spawn(args.n_workers)[args.worker_id])

    ifo_files = [x.strip() for x in args.ifo_files.split(",") if x.strip()]
    ifos = bilby.gw.detector.InterferometerList([bilby.gw.detector.load_interferometer(x) for x in ifo_files])
    for ifo in ifos:
        ifo.minimum_frequency = 10.0

    waveform_arguments = dict(
        waveform_approximant=WAVEFORM_APPROXIMANT,
        reference_frequency=10.0,
        minimum_frequency=10.0,
    )

    wg_cache = {}

    def get_wg(duration):
        key = (int(duration), 8192.0)
        if key not in wg_cache:
            wg_cache[key] = bilby.gw.WaveformGenerator(
                duration=int(duration),
                sampling_frequency=8192.0,
                frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
                waveform_arguments=waveform_arguments,
            )
        return wg_cache[key]

    keys = [
        "worker_id", "trial_index", "noise_seed", "network_snr", "snr_ce_b20", "snr_ce_a40",
        "network_snr_complex_real", "network_snr_complex_imag",
        "z", "dL_mpc", "H0", "prior", "m1_src", "m2_src", "m1_det", "m2_det",
        "lambda1", "lambda2", "chi1", "chi2", "ra", "dec", "theta_jn", "psi", "phase",
        "geocent_time", "duration", "start_time",
        "snr_ce_b20_real", "snr_ce_b20_imag", "snr_ce_a40_real", "snr_ce_a40_imag",
    ]
    buf = {k: [] for k in keys}

    chunk_id = 0
    n_trials = 0
    n_det = 0
    t0 = time.time()

    def flush():
        nonlocal chunk_id
        if len(buf["trial_index"]) == 0:
            return
        np.savez_compressed(os.path.join(wdir, f"detected_chunk{chunk_id:04d}.npz"), **{k: np.asarray(v) for k, v in buf.items()})
        for k in buf:
            buf[k].clear()
        chunk_id += 1

    def handle_term(signum, frame):
        flush()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    while n_det < args.target_success:
        n_trials += 1

        m1s, m2s = sample_component_masses(rng)
        z = sample_z(rng)
        h0 = float(rng.uniform(10.0, 150.0))

        cosmo = FlatLambdaCDM(H0=h0, Om0=Planck18.Om0, Tcmb0=Planck18.Tcmb0)
        d_l = float(cosmo.luminosity_distance(z).value)
        m1d, m2d = m1s * (1.0 + z), m2s * (1.0 + z)
        prior_det = q_inj_detector_prior(m1s, m2s, z, h0)
        if prior_det <= 0.0:
            continue

        lam1, lam2 = lambda_of_m_phys(m1s), lambda_of_m_phys(m2s)
        ra, dec = sample_isotropic_sky(rng)
        theta_jn, psi, phase = sample_isotropic_orientation(rng)
        chi1 = float(chi1_prior.rescale(float(rng.random())))
        chi2 = float(chi2_prior.rescale(float(rng.random())))

        tc = 1126259642.413 + n_trials * 4096.0
        duration, _ = duration_from_time_to_merger(m1d, m2d, flow=10.0, base_duration=1024)
        start_time = tc + 2.0 - duration

        inj = dict(
            mass_1=m1d, mass_2=m2d, chi_1=chi1, chi_2=chi2,
            luminosity_distance=d_l, theta_jn=theta_jn, psi=psi, phase=phase,
            geocent_time=tc, ra=ra, dec=dec, lambda_1=lam1, lambda_2=lam2,
        )

        noise_seed = make_noise_seed(args.base_seed, args.worker_id, n_trials)
        bilby_seed(noise_seed)
        ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=8192.0, duration=duration, start_time=start_time)
        for ifo in ifos:
            ifo.minimum_frequency = 10.0

        wg = get_wg(duration)
        pols = wg.frequency_domain_strain(parameters=inj)

        snr_complex = {}
        snr_diag = {}
        for ifo in ifos:
            sig = np.asarray(ifo.get_detector_response(pols, inj))
            ifo.strain_data.frequency_domain_strain += sig
            snr_c = bilby.gw.utils.matched_filter_snr(
                signal=sig,
                frequency_domain_strain=ifo.strain_data.frequency_domain_strain,
                power_spectral_density=ifo.power_spectral_density_array,
                duration=duration,
            )
            snr_complex[ifo.name] = complex(snr_c)
            snr_diag[ifo.name] = {
                "real": float(np.real(snr_c)),
                "imag": float(np.imag(snr_c)),
                "abs": float(np.abs(snr_c)),
            }

        network_snr_complex = np.sqrt(np.sum([rho * rho for rho in snr_complex.values()]))
        net = float(np.real(network_snr_complex))
        if net < 30.0:
            continue

        n_det += 1
        buf["worker_id"].append(args.worker_id)
        buf["trial_index"].append(n_trials)
        buf["noise_seed"].append(noise_seed)
        buf["network_snr"].append(net)
        buf["snr_ce_b20"].append(snr_diag.get("CE-B20", {}).get("abs", np.nan))
        buf["snr_ce_a40"].append(snr_diag.get("CE-A40", {}).get("abs", np.nan))
        buf["snr_ce_b20_real"].append(snr_diag.get("CE-B20", {}).get("real", np.nan))
        buf["snr_ce_b20_imag"].append(snr_diag.get("CE-B20", {}).get("imag", np.nan))
        buf["snr_ce_a40_real"].append(snr_diag.get("CE-A40", {}).get("real", np.nan))
        buf["snr_ce_a40_imag"].append(snr_diag.get("CE-A40", {}).get("imag", np.nan))
        buf["network_snr_complex_real"].append(float(np.real(network_snr_complex)))
        buf["network_snr_complex_imag"].append(float(np.imag(network_snr_complex)))
        buf["z"].append(z)
        buf["dL_mpc"].append(d_l)
        buf["H0"].append(h0)
        buf["prior"].append(prior_det)
        buf["m1_src"].append(m1s)
        buf["m2_src"].append(m2s)
        buf["m1_det"].append(m1d)
        buf["m2_det"].append(m2d)
        buf["lambda1"].append(lam1)
        buf["lambda2"].append(lam2)
        buf["chi1"].append(chi1)
        buf["chi2"].append(chi2)
        buf["ra"].append(ra)
        buf["dec"].append(dec)
        buf["theta_jn"].append(theta_jn)
        buf["psi"].append(psi)
        buf["phase"].append(phase)
        buf["geocent_time"].append(tc)
        buf["duration"].append(duration)
        buf["start_time"].append(start_time)

        if (n_det % args.progress_every_det) == 0:
            elapsed = time.time() - t0
            print(f"[w{args.worker_id:03d}] det={n_det}/{args.target_success} trials={n_trials} elapsed={elapsed/60:.1f}min", flush=True)

        if (n_det % args.chunk) == 0:
            flush()

    flush()
    with open(os.path.join(wdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            dict(
                worker_id=args.worker_id,
                n_workers=args.n_workers,
                n_trials_total=n_trials,
                n_detected=n_det,
                target_success=args.target_success,
                total_generated=n_trials,
                prior_definition="q_m*q_z*q_H*(1+z)^(-2)*(ddL/dz)^(-1)",
                spin_prior="AlignedSpin(a_prior=Uniform(0,0.05))",
                waveform_arguments=waveform_arguments,
                lal_data_path=LAL_DATA_PATH,
                elapsed_s=float(time.time() - t0),
            ),
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"[DONE worker {args.worker_id}] detected={n_det} trials={n_trials} out={wdir}", flush=True)


if __name__ == "__main__":
    main()
