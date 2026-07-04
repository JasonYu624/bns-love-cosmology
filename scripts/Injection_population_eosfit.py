#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform

import bilby
import numpy as np
from astropy.cosmology import Planck18
from bilby.core.utils.random import seed as bilby_seed


def parse_args():
    p = argparse.ArgumentParser(description="Section 2 mock detected catalog injections")
    p.add_argument("--outdir", type=str, default="outputs/outdir_population_exactfd_mband0")
    p.add_argument("--label", type=str, default="bns_pop_inj_exactfd_mband0")
    p.add_argument("--target-n", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--signal-filename", type=str, default="bns_signal_exact_fd.npz")
    return p.parse_args()


def sample_trunc_gaussian(mu, sigma, low, high, rng):
    while True:
        x = float(rng.normal(mu, sigma))
        if low <= x <= high:
            return x


def make_z_sampler(z_max, gamma, ngrid=6000):
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


def sample_isotropic_sky(rng):
    ra = float(rng.uniform(0.0, 2.0 * np.pi))
    dec = float(np.arcsin(rng.uniform(-1.0, 1.0)))
    return ra, dec


def sample_isotropic_orientation(rng):
    theta_jn = float(np.arccos(rng.uniform(-1.0, 1.0)))
    psi = float(rng.uniform(0.0, np.pi))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    return theta_jn, psi, phase


def next_pow2_geq(x):
    return int(2 ** math.ceil(math.log2(max(1.0, float(x)))))


def duration_from_time_to_merger(m1_det, m2_det, fmin, base_duration):
    delta_t = bilby.gw.utils.calculate_time_to_merger(frequency=float(fmin), mass_1=float(m1_det), mass_2=float(m2_det))
    delta_t = float(np.round(delta_t, 1))
    return max(base_duration, next_pow2_geq(delta_t)), delta_t


def lambda_of_m_phys(m_phys):
    a0, a1, a2 = -0.30781804, 0.79244108, -0.51480556
    mbar = float(m_phys)
    poly = 1.0 + a0 + a1 * mbar + a2 * mbar**2
    lam = 3500.0 * poly / mbar**5
    return float(max(lam, 0.0))


def save_detector_signals_fd(evdir, signal_filename, frequency_array, detector_signals, duration_s, start_time_s):
    out = {
        "frequency_array": np.asarray(frequency_array, dtype=np.float64),
        "duration": np.array([float(duration_s)], dtype=np.float64),
        "start_time": np.array([float(start_time_s)], dtype=np.float64),
    }
    for ifo_name, h_ifo in detector_signals.items():
        out[f"h_{ifo_name}"] = np.asarray(h_ifo, dtype=np.complex128)
    np.savez_compressed(os.path.join(evdir, signal_filename), **out)


def main():
    args = parse_args()

    z_max = 0.3
    gamma = 0.0
    m_min, m_max = 1.1, 2.25
    m_mu, m_sigma = 1.33, 0.09
    snr_thresh = 30.0
    fmin, fref = 10.0, 10.0
    fs = 8192.0
    base_duration = 1024
    geocent_time_0 = 1126259642.413
    trial_spacing = 4096.0
    ifo_files = ["CE_B20.interferometer", "CE_A40.interferometer"]

    chi1_prior = bilby.gw.prior.AlignedSpin(
        name="chi_1", a_prior=bilby.core.prior.Uniform(minimum=0.0, maximum=0.05)
    )
    chi2_prior = bilby.gw.prior.AlignedSpin(
        name="chi_2", a_prior=bilby.core.prior.Uniform(minimum=0.0, maximum=0.05)
    )

    rng = np.random.default_rng(args.base_seed)
    sample_z = make_z_sampler(z_max, gamma)

    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    os.makedirs(args.outdir, exist_ok=True)

    accepted_path = os.path.join(args.outdir, "accepted.jsonl")
    config_path = os.path.join(args.outdir, "catalog_config.json")

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXP_NRTidalv3",
        reference_frequency=fref,
        minimum_frequency=fmin,
        PhenomXHMThresholdMband=0,
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                outdir=args.outdir,
                label=args.label,
                target_n=args.target_n,
                base_seed=args.base_seed,
                snr_thresh=snr_thresh,
                z_max=z_max,
                gamma=gamma,
                mass_prior=dict(type="truncated_gaussian", mu=m_mu, sigma=m_sigma, mmin=m_min, mmax=m_max),
                spin_prior="AlignedSpin(a_prior=Uniform(0,0.05))",
                geocent_time_0=geocent_time_0,
                trial_spacing=trial_spacing,
                fmin=fmin,
                fref=fref,
                base_duration=base_duration,
                sampling_frequency=fs,
                ifo_files=ifo_files,
                waveform_arguments=waveform_arguments,
                software=dict(python=platform.python_version(), bilby=getattr(bilby, "__version__", "unknown"), numpy=np.__version__),
            ),
            f,
            indent=2,
            sort_keys=True,
        )

    ifos = bilby.gw.detector.InterferometerList([bilby.gw.detector.load_interferometer(x) for x in ifo_files])
    for ifo in ifos:
        ifo.minimum_frequency = fmin

    wg_cache = {}

    def get_wg(duration):
        key = (int(duration), float(fs))
        if key not in wg_cache:
            wg_cache[key] = bilby.gw.WaveformGenerator(
                duration=int(duration),
                sampling_frequency=float(fs),
                frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
                waveform_arguments=waveform_arguments,
            )
        return wg_cache[key]

    n_trials = 0
    n_accept = 0
    n_support_reject = 0

    with open(accepted_path, "w", encoding="utf-8") as fj:
        while n_accept < args.target_n:
            n_trials += 1

            m1s = sample_trunc_gaussian(m_mu, m_sigma, m_min, m_max, rng)
            m2s = sample_trunc_gaussian(m_mu, m_sigma, m_min, m_max, rng)
            if m2s > m1s:
                m1s, m2s = m2s, m1s

            z = sample_z(rng)
            d_l = float(Planck18.luminosity_distance(z).value)
            m1d, m2d = m1s * (1.0 + z), m2s * (1.0 + z)

            lam1, lam2 = lambda_of_m_phys(m1s), lambda_of_m_phys(m2s)
            pe_support_ok = (
                (0.8 <= m1s <= 1.8)
                and (0.8 <= m2s <= 1.8)
                and (10.0 <= lam1 <= 1.0e4)
                and (10.0 <= lam2 <= 1.0e4)
            )
            if not pe_support_ok:
                n_support_reject += 1
                if (n_support_reject % args.progress_every) == 0:
                    print(f"[REJECT support] count={n_support_reject} trial={n_trials}", flush=True)
                continue

            ra, dec = sample_isotropic_sky(rng)
            theta_jn, psi, phase = sample_isotropic_orientation(rng)
            chi1 = float(chi1_prior.rescale(float(rng.random())))
            chi2 = float(chi2_prior.rescale(float(rng.random())))

            tc = geocent_time_0 + n_trials * trial_spacing
            duration, delta_t = duration_from_time_to_merger(m1d, m2d, fmin, base_duration)
            start_time = tc + 2.0 - duration

            inj = dict(
                mass_1=m1d,
                mass_2=m2d,
                chi_1=chi1,
                chi_2=chi2,
                luminosity_distance=d_l,
                theta_jn=theta_jn,
                psi=psi,
                phase=phase,
                geocent_time=tc,
                ra=ra,
                dec=dec,
                lambda_1=lam1,
                lambda_2=lam2,
            )

            bilby_seed(args.base_seed + n_trials)
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=fs,
                duration=duration,
                start_time=start_time,
            )
            for ifo in ifos:
                ifo.minimum_frequency = fmin

            wg = get_wg(duration)
            pols = wg.frequency_domain_strain(parameters=inj)
            detector_signals = {}
            snr_complex = {}
            snr_diag = {}
            frequency_array = np.asarray(wg.frequency_array, dtype=np.float64)
            for ifo in ifos:
                sig = np.asarray(ifo.get_detector_response(pols, inj))
                detector_signals[ifo.name] = sig
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
            network_snr = float(np.real(network_snr_complex))
            if network_snr < snr_thresh:
                continue

            n_accept += 1
            evname = f"event_{n_accept:04d}"
            evdir = os.path.join(args.outdir, evname)
            os.makedirs(evdir, exist_ok=True)

            meta = dict(
                event=evname,
                accepted_index=n_accept,
                trial_index=n_trials,
                noise_seed=args.base_seed + n_trials,
                network_snr=network_snr,
                network_snr_complex_real=float(np.real(network_snr_complex)),
                network_snr_complex_imag=float(np.imag(network_snr_complex)),
                ifo_snrs=snr_diag,
                ifo_snrs_abs={k: v["abs"] for k, v in snr_diag.items()},
                z=z,
                luminosity_distance_mpc=d_l,
                mass_1_source=m1s,
                mass_2_source=m2s,
                mass_1_detector=m1d,
                mass_2_detector=m2d,
                lambda_1=lam1,
                lambda_2=lam2,
                injection_parameters=inj,
                duration_check=dict(deltaT_inband_rounded_s=delta_t, fmin_used_hz=fmin),
                data_segment=dict(duration=duration, start_time=start_time, geocent_time=tc, sampling_frequency=fs),
                exact_signal_file=dict(
                    filename=args.signal_filename,
                    frequency_array_key="frequency_array",
                    duration_key="duration",
                    start_time_key="start_time",
                    detector_signal_keys={ifo.name: f"h_{ifo.name}" for ifo in ifos},
                ),
                pe_support_check=dict(
                    mass_bounds=[0.8, 1.8],
                    lambda_bounds=[10.0, 1.0e4],
                    passed=True,
                ),
            )

            with open(os.path.join(evdir, "meta.json"), "w", encoding="utf-8") as fmeta:
                json.dump(meta, fmeta, indent=2, sort_keys=True)
            save_detector_signals_fd(
                evdir=evdir,
                signal_filename=args.signal_filename,
                frequency_array=frequency_array,
                detector_signals=detector_signals,
                duration_s=duration,
                start_time_s=start_time,
            )
            fj.write(json.dumps(meta) + "\n")
            fj.flush()

            if (n_trials % args.progress_every) == 0:
                print(
                    f"[PROG] trials={n_trials} accepted={n_accept} "
                    f"support_reject={n_support_reject} lastSNR={network_snr:.1f}",
                    flush=True,
                )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["pe_support_check"] = dict(
        mass_bounds=[0.8, 1.8],
        lambda_bounds=[10.0, 1.0e4],
        strict_rejection=True,
        n_support_reject=n_support_reject,
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(
        f"[DONE] trials={n_trials}, accepted={n_accept}, support_reject={n_support_reject}, outdir={args.outdir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
