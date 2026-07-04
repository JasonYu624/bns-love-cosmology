#!/usr/bin/env python
"""
Detected-population PP-style rank diagnostic for the BNS SEOBNR population PE runs.

This is NOT a formal Bayesian coverage PP plot.  The catalog is detection-selected
(network-SNR threshold + PE support cuts), so the per-event ranks u_i are not
guaranteed uniform even under a perfect pipeline.  Deviations from the diagonal can
reflect selection effects (distance/inclination), reweighting ESS, prior/population
mismatch, OR genuine PE issues.  Read it as a *diagnostic*, not a calibration proof.

For each event i and each parameter theta:

    u_i = posterior CDF evaluated at the injected truth
        = sum_j w_j * 1[ theta_j < theta_true ] / sum_j w_j

(strict "<", matching bilby's Result.get_injection_credible_level).  We then plot the
empirical CDF of {u_i} per parameter against the diagonal, with Bilby-style 1/2/3-sigma
binomial bands and a per-parameter KS p-value in the legend.

Run on the cluster inside the `GW` conda env:
    module load anaconda3/2025.6 && conda activate GW
    python pp_diagnostic_detected_population.py
"""

# =====================================================================
# 0. Configuration (edit these)
# =====================================================================
POP_OUTDIR = "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outputs/outdir_population_SEOBNR"
PE_OUTDIR = "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outputs/outdir_population_run_SEOBNR"

EVENT_INDICES = list(range(1, 101))  # event_0001 .. event_0100

# Posterior-source precedence (see README block below):
#   USE_CACHE_WEIGHTS=True  -> use the reweight cache npz: ORIGINAL posterior + ln_weights,
#                              weighted CDF.  This is the statistically cleanest path and
#                              OVERRIDES USE_REWEIGHTED to avoid double-weighting.
#   USE_CACHE_WEIGHTS=False, USE_REWEIGHTED=True  -> use the *_reweighted_*_augmented.csv
#                              (already equal-weight resampled to ESS), unweighted CDF.
#   USE_CACHE_WEIGHTS=False, USE_REWEIGHTED=False -> use the un-reweighted augmented CSV
#                              (relative-binning posterior, BEFORE full-likelihood reweight).
USE_REWEIGHTED = True
USE_CACHE_WEIGHTS = True

OUT_PREFIX = "detected_population_pp_diagnostic"

# Optional secondary sanity check using bilby.make_pp_plot on the raw, unreweighted
# Result hdf5 files (natively sampled params only).  Off by default; it is slower
# (loads 100 hdf5 files) and is secondary to the custom diagnostic above.
RUN_BILBY_SANITY = False

# Parameters to rank.  Periodic params (phase/psi/ra) are intentionally excluded:
# naive CDF ranks are not meaningful on a circle.
PARAMETERS = [
    "chirp_mass",
    "mass_ratio",
    "mass_1_detector",
    "mass_2_detector",
    "luminosity_distance",
    "redshift_sample",
    "H0_sample",
    "theta_jn",
    "lambda_1",
    "lambda_2",
    "lambda_tilde",
    "delta_a0",
    "delta_a1",
    "delta_a2",
]

# Pretty labels for the legend.
PARAM_LABELS = {
    "chirp_mass": r"$\mathcal{M}_c^{\rm det}$",
    "mass_ratio": r"$q$",
    "mass_1_detector": r"$m_1^{\rm det}$",
    "mass_2_detector": r"$m_2^{\rm det}$",
    "luminosity_distance": r"$d_L$",
    "redshift_sample": r"$z$",
    "H0_sample": r"$H_0$",
    "theta_jn": r"$\theta_{JN}$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "lambda_tilde": r"$\tilde{\Lambda}$",
    "delta_a0": r"$\delta a_0$",
    "delta_a1": r"$\delta a_1$",
    "delta_a2": r"$\delta a_2$",
}

# Some parameters live under a different column name depending on the source.
# In the reweight cache the bilby-native detector-frame masses are mass_1 / mass_2;
# the augmented CSVs additionally expose mass_1_detector / mass_2_detector.
COLUMN_ALIASES = {
    "mass_1_detector": ["mass_1_detector", "mass_1"],
    "mass_2_detector": ["mass_2_detector", "mass_2"],
}


# =====================================================================
# 1. Imports, cosmology, and truth helpers (mirrors PE_SEOBNR_reweight.py)
# =====================================================================
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from astropy.cosmology import FlatLambdaCDM, Planck18
import bilby

# True sample-level H0 used by the PE script: H0_TRUE = Planck18.H0.value
H0_TRUE = float(Planck18.H0.value)

# Reproduce the PE script's fiducial redshift interpolation EXACTLY so the truth
# for `redshift_sample` is in the same coordinate as the posterior column.
# (PE_SEOBNR_reweight.py: cosmo_fid = FlatLambdaCDM(H0=70, Om0=Planck18.Om0, Tcmb0=Planck18.Tcmb0))
_cosmo_fid = FlatLambdaCDM(H0=70.0, Om0=float(Planck18.Om0), Tcmb0=Planck18.Tcmb0)
_z_grid = np.linspace(0.0, 2.0, 20000)
_dL_grid = np.asarray(_cosmo_fid.luminosity_distance(_z_grid).value, dtype=float)


def z_from_dL_H0_vec(dL_mpc, H0):
    dL_scaled = np.asarray(dL_mpc, dtype=float) * (np.asarray(H0, dtype=float) / 70.0)
    return np.interp(np.clip(dL_scaled, _dL_grid[0], _dL_grid[-1]), _dL_grid, _z_grid)


def build_truths_from_meta(meta):
    """Construct injected-truth values keyed by the PE parameter names."""
    m1d = float(meta["mass_1_detector"])
    m2d = float(meta["mass_2_detector"])
    dL = float(meta["luminosity_distance_mpc"])
    lam1 = float(meta["lambda_1"])
    lam2 = float(meta["lambda_2"])
    theta_jn = float(meta["injection_parameters"]["theta_jn"])

    # Detector-frame chirp mass and (sub-unity) mass ratio from injected component masses.
    chirp_mass = (m1d * m2d) ** 0.6 / (m1d + m2d) ** 0.2
    mass_ratio = min(m1d, m2d) / max(m1d, m2d)

    # lambda_tilde truth uses DETECTOR-frame masses, matching PE_SEOBNR_reweight.py:263.
    lambda_tilde = float(
        bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(lam1, lam2, m1d, m2d)
    )

    return {
        "chirp_mass": chirp_mass,
        "mass_ratio": mass_ratio,
        "mass_1_detector": m1d,
        "mass_2_detector": m2d,
        "luminosity_distance": dL,
        "redshift_sample": float(z_from_dL_H0_vec(dL, H0_TRUE)),
        "H0_sample": H0_TRUE,
        "theta_jn": theta_jn,
        "lambda_1": lam1,
        "lambda_2": lam2,
        "lambda_tilde": lambda_tilde,
        "delta_a0": 0.0,
        "delta_a1": 0.0,
        "delta_a2": 0.0,
    }


# =====================================================================
# 2. Per-event posterior + weight loader (source precedence)
# =====================================================================
def _find_one(pattern, must_exclude=None):
    """Return the single path matching `pattern` (optionally excluding a substring)."""
    hits = sorted(glob.glob(pattern))
    if must_exclude is not None:
        hits = [h for h in hits if must_exclude not in os.path.basename(h)]
    return hits[0] if hits else None


def normalize_log_weights(ln_weights):
    """Stable exp-normalise; returns weights summing to 1, or None if degenerate."""
    lnw = np.asarray(ln_weights, dtype=float)
    good = np.isfinite(lnw)
    if good.sum() == 0:
        return None
    w = np.zeros_like(lnw)
    w[good] = np.exp(lnw[good] - np.max(lnw[good]))
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return w / s


def load_event_samples(idx):
    """
    Return (posterior_df, weights, source_tag) for event `idx`, following the
    USE_CACHE_WEIGHTS / USE_REWEIGHTED precedence.  weights is an array summing to 1.
    Returns (None, None, reason) if the event cannot be loaded.
    """
    tag = f"event_{idx:04d}"
    glob_base = os.path.join(PE_OUTDIR, f"*{tag}*")

    # --- Path A: reweight cache npz (original posterior + ln_weights, weighted) ---
    if USE_CACHE_WEIGHTS:
        cache = _find_one(glob_base + "reweight_arrays.npz")
        if cache is not None:
            data = np.load(cache, allow_pickle=True)
            if "posterior_dict" in data and "ln_weights" in data:
                post = pd.DataFrame(data["posterior_dict"][0])
                w = normalize_log_weights(data["ln_weights"])
                if w is not None and len(w) == len(post):
                    return post, w, "cache_weighted"
                return None, None, f"{tag}: cache weights degenerate/mismatched"
        # Fall through to CSV if cache missing (requirement 6).

    # --- Path B/C: augmented CSV (already equal-weight), unweighted ---
    if USE_REWEIGHTED:
        csv = _find_one(glob_base + "reweighted*posterior_augmented.csv")
        src = "reweighted_csv"
    else:
        csv = _find_one(glob_base + "posterior_augmented.csv", must_exclude="reweighted")
        src = "unreweighted_csv"

    if csv is None:
        return None, None, f"{tag}: no posterior file found"
    post = pd.read_csv(csv)
    w = np.full(len(post), 1.0 / len(post))
    return post, w, src


def resolve_column(post, param):
    """Map a logical parameter name to an actual column in `post`, honouring aliases."""
    for cand in COLUMN_ALIASES.get(param, [param]):
        if cand in post.columns:
            return cand
    return None


# =====================================================================
# 3. Main loop: build credible levels (events x parameters)
# =====================================================================
records = []  # one dict per successfully loaded event
source_counter = {}
n_skipped = 0

for idx in EVENT_INDICES:
    tag = f"event_{idx:04d}"
    meta_path = os.path.join(POP_OUTDIR, tag, "meta.json")
    if not os.path.exists(meta_path):
        print(f"[skip] {tag}: missing meta.json")
        n_skipped += 1
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    truths = build_truths_from_meta(meta)

    post, weights, src = load_event_samples(idx)
    if post is None:
        print(f"[skip] {src}")
        n_skipped += 1
        continue
    source_counter[src] = source_counter.get(src, 0) + 1

    row = {"event": tag}
    for param in PARAMETERS:
        col = resolve_column(post, param)
        truth = truths.get(param, np.nan)
        if col is None or not np.isfinite(truth):
            row[param] = np.nan
            continue
        samples = post[col].to_numpy()
        # weighted CDF at truth (strict "<", matching bilby convention)
        u = float(np.sum(weights * (samples < truth)))
        row[param] = u
    records.append(row)

if not records:
    raise SystemExit("No events were loaded; check POP_OUTDIR / PE_OUTDIR / EVENT_INDICES.")

credible_levels = pd.DataFrame(records).set_index("event")
N = len(credible_levels)

print("\n=== Source breakdown ===")
for k, v in sorted(source_counter.items()):
    print(f"  {k}: {v} events")
print(f"  skipped: {n_skipped}")
if USE_CACHE_WEIGHTS and "reweighted_csv" in source_counter:
    print("  note: some events fell back to reweighted CSV (cache missing).")
if USE_CACHE_WEIGHTS:
    print("  note: USE_CACHE_WEIGHTS=True overrides USE_REWEIGHTED (no double-weighting).")


# =====================================================================
# 4. Save the percentile table
# =====================================================================
csv_out = f"{OUT_PREFIX}_percentiles.csv"
credible_levels.to_csv(csv_out)
print(f"\nWrote percentile table: {csv_out}")


# =====================================================================
# 5. Per-parameter KS p-values and usage counts
# =====================================================================
print("\n=== Per-parameter summary (N used, KS p-value vs Uniform[0,1]) ===")
print(f"{'parameter':<20}{'N_used':>8}{'KS p-value':>14}")
ks_pvalues = {}
n_used = {}
for param in PARAMETERS:
    u = credible_levels[param].dropna().to_numpy()
    n_used[param] = len(u)
    if len(u) >= 2:
        p = float(scipy.stats.kstest(u, "uniform").pvalue)
    else:
        p = np.nan
    ks_pvalues[param] = p
    print(f"{param:<20}{len(u):>8}{p:>14.4f}")

# Combined p-value (Fisher), diagnostic only.
finite_p = [p for p in ks_pvalues.values() if np.isfinite(p)]
combined_p = (
    float(scipy.stats.combine_pvalues(finite_p)[1]) if len(finite_p) > 1 else np.nan
)
print(f"\nCombined (Fisher) p-value: {combined_p:.4f}  [diagnostic only]")


# =====================================================================
# 6. Bilby-style plot (diagonal + binomial bands + one CDF curve per param)
# =====================================================================
x_values = np.linspace(0, 1, 1001)
fig, ax = plt.subplots(figsize=(7.5, 7))

# 1/2/3-sigma binomial confidence bands (same construction as bilby.make_pp_plot).
for ci, alpha in zip([0.68, 0.95, 0.997], [0.1, 0.1, 0.1]):
    edge = (1.0 - ci) / 2.0
    lower = scipy.stats.binom.ppf(1 - edge, N, x_values) / N
    upper = scipy.stats.binom.ppf(edge, N, x_values) / N
    lower[0] = 0.0
    upper[0] = 0.0
    ax.fill_between(x_values, lower, upper, alpha=alpha, color="k", linewidth=0)

# Diagonal.
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, zorder=1)

# One empirical-CDF curve per parameter.
colors = [f"C{i}" for i in range(10)]
linestyles = ["-", "--", ":", "-."]
for ii, param in enumerate(PARAMETERS):
    u = credible_levels[param].dropna().to_numpy()
    if len(u) < 2:
        continue
    pp = np.array([np.mean(u < xx) for xx in x_values])
    label = f"{PARAM_LABELS.get(param, param)} ({ks_pvalues[param]:.3f})"
    ax.plot(
        x_values,
        pp,
        linestyle=linestyles[ii % len(linestyles)],
        color=colors[ii % len(colors)],
        linewidth=1.6,
        label=label,
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Credible interval  (posterior CDF at injected truth)")
ax.set_ylabel("Fraction of events with $u_i$ below")
ax.set_title(
    f"Detected-population PP-style rank diagnostic\n"
    f"N={N} events, combined KS p={combined_p:.3f}  (NOT a formal coverage PP plot)"
)
ax.legend(handlelength=2.2, labelspacing=0.25, fontsize="x-small", loc="lower right")
fig.tight_layout()

png_out, pdf_out = f"{OUT_PREFIX}.png", f"{OUT_PREFIX}.pdf"
fig.savefig(png_out, dpi=200)
fig.savefig(pdf_out)
plt.close(fig)
print(f"\nWrote plot: {png_out}")
print(f"Wrote plot: {pdf_out}")


# =====================================================================
# 7. Suggested caption
# =====================================================================
print(
    "\n=== Suggested caption ===\n"
    f"Detected-population PP-style rank diagnostic for {N} SNR-selected BNS events.\n"
    "For each event and parameter we plot the empirical CDF of the per-event rank\n"
    "u_i = posterior CDF at the injected truth (weighted by the full-likelihood\n"
    "importance weights when the reweight cache is used). Grey bands are the central\n"
    "68/95/99.7% binomial intervals expected for uniform ranks. This is a diagnostic,\n"
    "NOT a formal Bayesian coverage PP plot: because the catalog is detection-selected\n"
    "(network SNR > 30 plus PE support cuts), departures from the diagonal can arise\n"
    "from selection effects, distance-inclination selection, reweighting ESS, or\n"
    "prior/population mismatch, in addition to genuine PE biases. KS p-values are\n"
    "shown for orientation only."
)


# =====================================================================
# 8. (Optional, secondary) bilby.make_pp_plot sanity check on raw Result files
# =====================================================================
def bilby_sanity_pp_plot():
    """
    Secondary cross-check: run bilby.make_pp_plot on the raw, UN-reweighted Result
    hdf5 files (natively sampled params only). This does NOT apply full-likelihood
    reweighting or handle derived truths as carefully as the custom diagnostic above,
    so treat it as a sanity check, not the headline result.
    """
    native = [
        "chirp_mass", "mass_ratio", "luminosity_distance",
        "H0_sample", "theta_jn", "delta_a0", "delta_a1", "delta_a2",
    ]
    results = []
    for idx in EVENT_INDICES:
        tag = f"event_{idx:04d}"
        meta_path = os.path.join(POP_OUTDIR, tag, "meta.json")
        res_path = _find_one(os.path.join(PE_OUTDIR, f"*{tag}*result.hdf5"))
        if not (res_path and os.path.exists(meta_path)):
            continue
        try:
            res = bilby.result.read_in_result(res_path)
            with open(meta_path) as f:
                meta = json.load(f)
            res.injection_parameters = build_truths_from_meta(meta)
            results.append(res)
        except Exception as exc:  # noqa: BLE001 - sanity check, keep going
            print(f"[bilby-sanity skip] {tag}: {exc}")
    if not results:
        print("[bilby-sanity] no raw results loaded.")
        return
    keys = [k for k in native if k in results[0].posterior.columns]
    bilby.core.result.make_pp_plot(
        results, filename=f"{OUT_PREFIX}_bilby_sanity_raw.png", keys=keys, save=True
    )
    print(f"[bilby-sanity] wrote {OUT_PREFIX}_bilby_sanity_raw.png (raw, unreweighted)")


if RUN_BILBY_SANITY:
    bilby_sanity_pp_plot()
