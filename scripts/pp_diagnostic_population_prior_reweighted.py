#!/usr/bin/env python
"""
Population-prior-reweighted PP-style rank diagnostic (advisor's stricter version).

The companion script `pp_diagnostic_detected_population.py` ranks the injected truth
inside the full-likelihood posterior that still carries the PE *sampling* prior
(pi_PE).  Because the injected truths are drawn from the astrophysical population
(p_pop), not from pi_PE, the ranks are not expected uniform even for a perfect
pipeline -- the curves mix a genuine-PE component with a prior/population-mismatch
component.

This script removes the prior-mismatch component the way the advisor suggested:
reweight each posterior so its effective prior becomes p_pop (the distribution the
injections were drawn from).  Concretely, on top of the full-likelihood importance
weights stored in the reweight cache (w_fullL = L_full / L_relbin) we multiply by the
prior ratio

    r(theta) = p_pop(theta) / pi_PE(theta).

Key simplifications (all exact for this pipeline):
  * Spin, sky, orientation and time priors are IDENTICAL between the injection draw
    and the PE prior (same AlignedSpin(U[0,0.05]); isotropic sky/orientation), so they
    cancel in r and need not be evaluated.
  * H0_sample and delta_a0/1/2 have NO population to reweight to (the injection fixes
    H0 = H0_true and delta_a = 0), so we leave them at the PE prior.  Their ranks are
    therefore NOT a clean coverage test (same caveat as before) and are shown only for
    context.
  * Detection selection is on the *noisy* matched-filter network SNR (data = signal +
    noise, SNR > 30), i.e. a selection on the DATA.  Selection on data preserves
    per-event coverage, so reweighting pi_PE -> p_pop is the correct and (to good
    approximation) sufficient fix; we do NOT additionally fold in the selection
    function.

So r only involves the blocks where the two priors differ: the masses and the
distance.  Working in (m1_det, m2_det, dL):
  * pi_PE(m1d, m2d, dL) ∝ 1 (uniform in component detector masses) × dL^2 (PowerLaw,
    alpha=2).  -> only the dL^2 factor is theta-dependent.
  * p_pop is induced by drawing source masses ~ TruncGauss(mu, sigma, [m_lo, m_hi]) and
    z ~ dVc/dz (1+z)^(gamma-1) on [0, z_max] (Planck18), with m_det = m_src(1+z) and
    dL = D(z).  Changing variables (m1s, m2s, z) -> (m1d, m2d, dL):

      p_pop(m1d, m2d, dL) ∝ g(m1s) g(m2s) h(z) / [ (1+z)^2  D'(z) ],
      z = D^{-1}(dL),  m_src = m_det / (1+z).

Only event-independent constants are dropped (they cancel in the per-event weight
normalisation), so we keep, up to an additive constant,

    ln r = ln g(m1s) + ln g(m2s) + ln h(z) - 2 ln(1+z) - ln D'(z) - 2 ln dL,

with ln r = -inf outside the population support (m_src in [m_lo, m_hi], z <= z_max).

Run on the cluster inside the `GW` conda env:
    module load anaconda3/2025.6 && conda activate GW
    python pp_diagnostic_population_prior_reweighted.py
"""

# =====================================================================
# 0. Configuration
# =====================================================================
POP_OUTDIR = "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outputs/outdir_population_SEOBNR"
PE_OUTDIR = "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outputs/outdir_population_run_SEOBNR"

EVENT_INDICES = list(range(1, 101))  # event_0001 .. event_0100

OUT_PREFIX = "population_prior_reweighted_pp_diagnostic"

# Existing (pi_PE) percentile table, used only to print a before/after comparison.
BEFORE_PERCENTILES = (
    "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/detected_population_pp_diagnostic_percentiles.csv"
)

# --- Injection population constants (must match Injection_population_SEOBNR.py) ---
Z_MAX = 0.3
GAMMA = 0.0
M_MU, M_SIGMA = 1.33, 0.09
M_LO, M_HI = 1.1, 2.25  # TruncGaussian source-mass bounds

# Parameters to rank.  "clean" = prior now matches the injection draw (mass/dL) or was
# already shared (theta_jn); "entangled" = involves the H0 nuisance or has no
# population, so its rank is NOT a clean coverage test under this reweighting.
CLEAN_PARAMS = [
    "chirp_mass",
    "mass_ratio",
    "mass_1_detector",
    "mass_2_detector",
    "luminosity_distance",
    "theta_jn",
]
ENTANGLED_PARAMS = [
    "redshift_sample",
    "H0_sample",
    "lambda_1",
    "lambda_2",
    "lambda_tilde",
    "delta_a0",
    "delta_a1",
    "delta_a2",
]
PARAMETERS = CLEAN_PARAMS + ENTANGLED_PARAMS

PARAM_LABELS = {
    "chirp_mass": r"$\mathcal{M}_c^{\rm det}$",
    "mass_ratio": r"$q$",
    "mass_1_detector": r"$m_1^{\rm det}$",
    "mass_2_detector": r"$m_2^{\rm det}$",
    "luminosity_distance": r"$d_L$",
    "theta_jn": r"$\theta_{JN}$",
    "redshift_sample": r"$z$",
    "H0_sample": r"$H_0$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "lambda_tilde": r"$\tilde{\Lambda}$",
    "delta_a0": r"$\delta a_0$",
    "delta_a1": r"$\delta a_1$",
    "delta_a2": r"$\delta a_2$",
}

# Column aliases: the reweight cache uses bilby-native mass_1/mass_2 (detector frame);
# the augmented CSV fallback uses mass_1_detector/mass_2_detector.
COLUMN_ALIASES = {
    "mass_1_detector": ["mass_1_detector", "mass_1"],
    "mass_2_detector": ["mass_2_detector", "mass_2"],
}


# =====================================================================
# 1. Imports, cosmology, truth helpers
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

H0_TRUE = float(Planck18.H0.value)

# --- PE fiducial redshift map (H0=70), only for the redshift_sample truth ---
_cosmo_fid = FlatLambdaCDM(H0=70.0, Om0=float(Planck18.Om0), Tcmb0=Planck18.Tcmb0)
_z_grid_fid = np.linspace(0.0, 2.0, 20000)
_dL_grid_fid = np.asarray(_cosmo_fid.luminosity_distance(_z_grid_fid).value, dtype=float)


def z_from_dL_H0_vec(dL_mpc, H0):
    dL_scaled = np.asarray(dL_mpc, dtype=float) * (np.asarray(H0, dtype=float) / 70.0)
    return np.interp(np.clip(dL_scaled, _dL_grid_fid[0], _dL_grid_fid[-1]), _dL_grid_fid, _z_grid_fid)


# --- TRUE cosmology (Planck18) grids for the p_pop density: dL <-> z and D'(z) ---
_zg = np.linspace(0.0, 0.45, 45001)
_Dg = np.asarray(Planck18.luminosity_distance(_zg).value, dtype=float)
_dDdz_g = np.gradient(_Dg, _zg)
_dVcdz_g = 4.0 * np.pi * np.asarray(Planck18.differential_comoving_volume(_zg).value, dtype=float)
_D_at_zmax = float(np.interp(Z_MAX, _zg, _Dg))

# ln h(z) = ln(dVc/dz) + (gamma - 1) ln(1+z);  ln D'(z).  (z=0 gives -inf in ln h but
# no samples live there.)
with np.errstate(divide="ignore"):
    _lnh_g = np.log(_dVcdz_g) + (GAMMA - 1.0) * np.log(1.0 + _zg)
    _lnDprime_g = np.log(_dDdz_g)


def log_prior_ratio(m1d, m2d, dL):
    """Unnormalised ln r = ln p_pop - ln pi_PE on (m1_det, m2_det, dL).  -inf outside
    the population support.  Constants that do not depend on (m1d, m2d, dL) are dropped
    (they cancel in the per-event weight normalisation)."""
    m1d = np.asarray(m1d, dtype=float)
    m2d = np.asarray(m2d, dtype=float)
    dL = np.asarray(dL, dtype=float)

    z = np.interp(dL, _Dg, _zg)
    one_pz = 1.0 + z
    m1s = m1d / one_pz
    m2s = m2d / one_pz

    lng1 = -((m1s - M_MU) ** 2) / (2.0 * M_SIGMA ** 2)
    lng2 = -((m2s - M_MU) ** 2) / (2.0 * M_SIGMA ** 2)
    lnh = np.interp(z, _zg, _lnh_g)
    lnDprime = np.interp(z, _zg, _lnDprime_g)

    lnr = lng1 + lng2 + lnh - 2.0 * np.log(one_pz) - lnDprime - 2.0 * np.log(dL)

    support = (
        (dL <= _D_at_zmax)
        & (m1s >= M_LO) & (m1s <= M_HI)
        & (m2s >= M_LO) & (m2s <= M_HI)
    )
    lnr = np.where(support, lnr, -np.inf)
    return lnr


def build_truths_from_meta(meta):
    m1d = float(meta["mass_1_detector"])
    m2d = float(meta["mass_2_detector"])
    dL = float(meta["luminosity_distance_mpc"])
    lam1 = float(meta["lambda_1"])
    lam2 = float(meta["lambda_2"])
    theta_jn = float(meta["injection_parameters"]["theta_jn"])

    chirp_mass = (m1d * m2d) ** 0.6 / (m1d + m2d) ** 0.2
    mass_ratio = min(m1d, m2d) / max(m1d, m2d)
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
# 2. Per-event loader: ORIGINAL posterior + base log-weights
#    (cache: full-likelihood ln_weights;  CSV fallback: equal weight)
# =====================================================================
def _find_one(pattern, must_exclude=None):
    hits = sorted(glob.glob(pattern))
    if must_exclude is not None:
        hits = [h for h in hits if must_exclude not in os.path.basename(h)]
    return hits[0] if hits else None


def load_event(idx):
    """Return (posterior_df, base_log_weights, source_tag) or (None, None, reason)."""
    tag = f"event_{idx:04d}"
    glob_base = os.path.join(PE_OUTDIR, f"*{tag}*")

    cache = _find_one(glob_base + "reweight_arrays.npz")
    if cache is not None:
        data = np.load(cache, allow_pickle=True)
        if "posterior_dict" in data and "ln_weights" in data:
            post = pd.DataFrame(data["posterior_dict"][0])
            base_lnw = np.asarray(data["ln_weights"], dtype=float)
            if len(base_lnw) == len(post):
                return post, base_lnw, "cache_fullL"
            return None, None, f"{tag}: cache weights length mismatch"

    # Fallback: equal-weight reweighted CSV (already full-L resampled under pi_PE).
    csv = _find_one(glob_base + "reweighted*posterior_augmented.csv")
    if csv is None:
        return None, None, f"{tag}: no cache and no reweighted CSV"
    post = pd.read_csv(csv)
    base_lnw = np.zeros(len(post), dtype=float)
    return post, base_lnw, "reweighted_csv"


def resolve_column(post, param):
    for cand in COLUMN_ALIASES.get(param, [param]):
        if cand in post.columns:
            return cand
    return None


def softmax_weights(log_w):
    lw = np.asarray(log_w, dtype=float)
    good = np.isfinite(lw)
    if good.sum() == 0:
        return None
    w = np.zeros_like(lw)
    w[good] = np.exp(lw[good] - np.max(lw[good]))
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return w / s


# =====================================================================
# 3. Main loop: prior-reweighted credible levels + ESS bookkeeping
# =====================================================================
records = []
ess_rows = []
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

    post, base_lnw, src = load_event(idx)
    if post is None:
        print(f"[skip] {src}")
        n_skipped += 1
        continue

    col_m1 = resolve_column(post, "mass_1_detector")
    col_m2 = resolve_column(post, "mass_2_detector")
    if col_m1 is None or col_m2 is None or "luminosity_distance" not in post.columns:
        print(f"[skip] {tag}: missing mass/dL columns")
        n_skipped += 1
        continue

    lnr = log_prior_ratio(post[col_m1].to_numpy(), post[col_m2].to_numpy(),
                          post["luminosity_distance"].to_numpy())

    # Diagnostic: ESS of the base (full-L, pi_PE) weights vs the prior-reweighted ones.
    w_base = softmax_weights(base_lnw)
    w_new = softmax_weights(base_lnw + lnr)
    if w_new is None:
        print(f"[skip] {tag}: all prior-reweighted weights underflowed")
        n_skipped += 1
        continue
    ess_base = (1.0 / np.sum(w_base ** 2)) if w_base is not None else float("nan")
    ess_new = 1.0 / np.sum(w_new ** 2)
    ess_rows.append({
        "event": tag,
        "n_samples": len(post),
        "frac_in_support": float(np.mean(np.isfinite(lnr))),
        "ess_base_fullL": float(ess_base),
        "ess_new_reweighted": float(ess_new),
        "ess_new_frac": float(ess_new / len(post)),
    })
    source_counter[src] = source_counter.get(src, 0) + 1

    row = {"event": tag}
    for param in PARAMETERS:
        col = resolve_column(post, param)
        truth = truths.get(param, np.nan)
        if col is None or not np.isfinite(truth):
            row[param] = np.nan
            continue
        samples = post[col].to_numpy()
        row[param] = float(np.sum(w_new * (samples < truth)))
    records.append(row)

if not records:
    raise SystemExit("No events loaded; check POP_OUTDIR / PE_OUTDIR / EVENT_INDICES.")

credible_levels = pd.DataFrame(records).set_index("event")
ess_df = pd.DataFrame(ess_rows).set_index("event")
N = len(credible_levels)

print("\n=== Source breakdown ===")
for k, v in sorted(source_counter.items()):
    print(f"  {k}: {v} events")
print(f"  skipped: {n_skipped}")

print("\n=== Effective sample size after prior reweighting ===")
print(f"  median ESS (full-L, pi_PE)      : {ess_df['ess_base_fullL'].median():.0f}")
print(f"  median ESS (population-prior)    : {ess_df['ess_new_reweighted'].median():.0f}")
print(f"  median ESS fraction (pop-prior)  : {ess_df['ess_new_frac'].median():.3f}")
print(f"  min ESS (population-prior)       : {ess_df['ess_new_reweighted'].min():.0f}")
print(f"  events with ESS < 50             : {(ess_df['ess_new_reweighted'] < 50).sum()}")


# =====================================================================
# 4. Save tables
# =====================================================================
credible_levels.to_csv(f"{OUT_PREFIX}_percentiles.csv")
ess_df.to_csv(f"{OUT_PREFIX}_ess.csv")
print(f"\nWrote percentile table: {OUT_PREFIX}_percentiles.csv")
print(f"Wrote ESS table:        {OUT_PREFIX}_ess.csv")


# =====================================================================
# 5. Per-parameter KS + before/after comparison
# =====================================================================
before = None
if os.path.exists(BEFORE_PERCENTILES):
    before = pd.read_csv(BEFORE_PERCENTILES).set_index("event")

ks_pvalues, n_used = {}, {}
print("\n=== Per-parameter rank summary (after = population-prior-reweighted) ===")
print(f"{'parameter':<20}{'N':>5}{'mean_u_before':>15}{'mean_u_after':>14}{'KS_before':>12}{'KS_after':>11}")
for param in PARAMETERS:
    u = credible_levels[param].dropna().to_numpy()
    n_used[param] = len(u)
    ks_after = float(scipy.stats.kstest(u, "uniform").pvalue) if len(u) >= 2 else np.nan
    ks_pvalues[param] = ks_after
    mean_after = float(np.mean(u)) if len(u) else np.nan
    if before is not None and param in before.columns:
        ub = before[param].dropna().to_numpy()
        mean_before = float(np.mean(ub)) if len(ub) else np.nan
        ks_before = float(scipy.stats.kstest(ub, "uniform").pvalue) if len(ub) >= 2 else np.nan
    else:
        mean_before, ks_before = np.nan, np.nan
    flag = "" if param in CLEAN_PARAMS else "  (entangled)"
    print(f"{param:<20}{len(u):>5}{mean_before:>15.3f}{mean_after:>14.3f}"
          f"{ks_before:>12.3f}{ks_after:>11.3f}{flag}")


# =====================================================================
# 6. Bilby-style PP plot (clean params solid, entangled params thin/dashed)
# =====================================================================
x_values = np.linspace(0, 1, 1001)
fig, ax = plt.subplots(figsize=(7.5, 7))

for ci in [0.68, 0.95, 0.997]:
    edge = (1.0 - ci) / 2.0
    lower = scipy.stats.binom.ppf(1 - edge, N, x_values) / N
    upper = scipy.stats.binom.ppf(edge, N, x_values) / N
    lower[0] = upper[0] = 0.0
    ax.fill_between(x_values, lower, upper, alpha=0.1, color="k", linewidth=0)

ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, zorder=1)

colors = [f"C{i}" for i in range(10)]
for ii, param in enumerate(PARAMETERS):
    u = credible_levels[param].dropna().to_numpy()
    if len(u) < 2:
        continue
    pp = np.array([np.mean(u < xx) for xx in x_values])
    clean = param in CLEAN_PARAMS
    label = f"{PARAM_LABELS.get(param, param)} ({ks_pvalues[param]:.3f})"
    if not clean:
        label += "*"
    # q and m2_det ranks coincide (Mc is tightly measured -> q is a function of m2_det
    # along the fixed-chirp-mass track).  Draw m2_det as a wide underlay and q as a thin
    # line on top so BOTH are visible (orange core with a red halo).
    if param == "mass_ratio":
        lw, zo, ls, al = 1.8, 6, "-", 1.0
    elif param == "mass_2_detector":
        lw, zo, ls, al = 3.6, 4, "-", 0.85
        label += "  (= q)"
    elif clean:
        lw, zo, ls, al = 2.0, 5, "-", 1.0
    else:
        lw, zo, ls, al = 1.0, 3, ":", 0.6
    ax.plot(
        x_values, pp,
        linestyle=ls,
        color=colors[ii % len(colors)],
        linewidth=lw,
        alpha=al,
        zorder=zo,
        label=label,
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Credible interval  (posterior CDF at injected truth)")
ax.set_ylabel("Fraction of events with $u_i$ below")
ax.set_title(
    f"Population-prior-reweighted PP-style rank diagnostic\n"
    f"N={N} events  (solid = prior now matches injection draw; "
    f"* = H0-entangled / no population)"
)
ax.legend(handlelength=2.2, labelspacing=0.25, fontsize="x-small", loc="lower right", ncol=2)
fig.tight_layout()

fig.savefig(f"{OUT_PREFIX}.png", dpi=200)
fig.savefig(f"{OUT_PREFIX}.pdf")
plt.close(fig)
print(f"\nWrote plot: {OUT_PREFIX}.png")
print(f"Wrote plot: {OUT_PREFIX}.pdf")


# =====================================================================
# 7. Focused before/after overlay for q and dL (the two flagged problems)
# =====================================================================
if before is not None:
    fig2, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax2, param in zip(axes, ["mass_ratio", "luminosity_distance"]):
        for ci in [0.68, 0.95, 0.997]:
            edge = (1.0 - ci) / 2.0
            lo = scipy.stats.binom.ppf(1 - edge, N, x_values) / N
            hi = scipy.stats.binom.ppf(edge, N, x_values) / N
            lo[0] = hi[0] = 0.0
            ax2.fill_between(x_values, lo, hi, alpha=0.1, color="k", linewidth=0)
        ax2.plot([0, 1], [0, 1], "k--", linewidth=1.0)

        ub = before[param].dropna().to_numpy()
        ua = credible_levels[param].dropna().to_numpy()
        ppb = np.array([np.mean(ub < xx) for xx in x_values])
        ppa = np.array([np.mean(ua < xx) for xx in x_values])
        ax2.plot(x_values, ppb, color="C3", lw=2.0,
                 label=f"before (PE prior)  mean_u={ub.mean():.3f}")
        ax2.plot(x_values, ppa, color="C0", lw=2.0,
                 label=f"after (pop. prior)  mean_u={ua.mean():.3f}")
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.set_xlabel("posterior CDF at injected truth")
        ax2.set_ylabel("fraction of events below")
        ax2.set_title(f"{PARAM_LABELS.get(param, param)}")
        ax2.legend(loc="lower right", fontsize="small")
    fig2.suptitle("Prior-mismatch removal: PE prior vs injection (population) prior")
    fig2.tight_layout()
    fig2.savefig(f"{OUT_PREFIX}_q_dL_before_after.png", dpi=200)
    plt.close(fig2)
    print(f"Wrote plot: {OUT_PREFIX}_q_dL_before_after.png")
