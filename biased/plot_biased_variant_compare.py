#!/usr/bin/env python
"""Biased sub-population: compare orig-prior baseline vs the 3 experimental
variants (v1 aligned / v2 fix-spin / v3 flat-spin) on events 14 & 54.

Outputs a diagnostic table (stdout + CSV) and TWO controlled-comparison corner
overlays per event, each isolating a single variable:
  A: orig vs v1   -> same spin (AlignedSpin), only the mass/q PRIOR differs
  B: v1/v2/v3     -> same new q-prior, only the SPIN treatment differs

Corner style follows notebooks/PE_SEOBNR_plot.ipynb: bilby.plot_multiple with
bins=30, plot_datapoints=False, smooth=0.9, quantiles=[0.16,0.84], and red truth
lines drawn by add_truth_lines(). All posteriors are the FULL-likelihood
reweighted, equal-weight resamples.

Run inside the `GW` conda env:
    python biased/plot_biased_variant_compare.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bilby
np.random.seed(0)

ROOT = "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
POP = os.path.join(ROOT, "outputs/outdir_population_SEOBNR")
FIGDIR = os.path.join(ROOT, "figures")

EVENTS = [14, 54]
# label -> (posterior_csv_template, summary_json_template, colour)
SRC = {
    "orig prior": (os.path.join(ROOT, "outputs/outdir_population_run_SEOBNR/bns_event_{ev:04d}_seobnr_reweighted_posterior_augmented.csv"),
                   os.path.join(ROOT, "outputs/outdir_population_run_SEOBNR/bns_event_{ev:04d}_seobnr_reweighted_summary.json"),
                   "0.35"),
    "v1 aligned": (os.path.join(ROOT, "biased/outputs/outdir_biased_v1_qprior/bns_event_{ev:04d}_seobnr_reweighted_posterior_augmented.csv"),
                   os.path.join(ROOT, "biased/outputs/outdir_biased_v1_qprior/bns_event_{ev:04d}_seobnr_reweighted_summary.json"),
                   "#1f77b4"),
    "v2 fix":     (os.path.join(ROOT, "biased/outputs/outdir_biased_v2_fixspin/bns_event_{ev:04d}_seobnr_reweighted_posterior_augmented.csv"),
                   os.path.join(ROOT, "biased/outputs/outdir_biased_v2_fixspin/bns_event_{ev:04d}_seobnr_reweighted_summary.json"),
                   "#d62728"),
    "v3 flat":    (os.path.join(ROOT, "biased/outputs/outdir_biased_v3_flatspin/bns_event_{ev:04d}_seobnr_reweighted_posterior_augmented.csv"),
                   os.path.join(ROOT, "biased/outputs/outdir_biased_v3_flatspin/bns_event_{ev:04d}_seobnr_reweighted_summary.json"),
                   "#2ca02c"),
}
ORDER = ["orig prior", "v1 aligned", "v2 fix", "v3 flat"]

PARAMS = ["mass_ratio", "chirp_mass", "luminosity_distance", "chi_1", "chi_2", "theta_jn"]
LABELS = {"mass_ratio": r"$q$", "chirp_mass": r"$\mathcal{M}_c$",
          "luminosity_distance": r"$d_L$", "chi_1": r"$\chi_{1z}$",
          "chi_2": r"$\chi_{2z}$", "theta_jn": r"$\theta_{JN}$"}


def load_truth(ev):
    inj = json.load(open(os.path.join(POP, f"event_{ev:04d}", "meta.json")))["injection_parameters"]
    m1, m2 = float(inj["mass_1"]), float(inj["mass_2"])
    inj["mass_ratio"] = m2 / m1
    inj["chirp_mass"] = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return inj


def q68(x):
    lo, med, hi = np.percentile(x, [16, 50, 84])
    return med, lo, hi


def ess_of(sumpath):
    try:
        s = json.load(open(sumpath))
        return s.get("ess", np.nan), s.get("ess_fraction", np.nan), s.get("n_rw_posterior", np.nan)
    except Exception:
        return np.nan, np.nan, np.nan


rows = []
posteriors = {}   # (ev,label) -> df
for ev in EVENTS:
    truth = load_truth(ev)
    q_t = float(truth["mass_ratio"]); dL_t = float(truth["luminosity_distance"])
    chi1_t = float(truth.get("chi_1", 0.0)); chi2_t = float(truth.get("chi_2", 0.0))
    for lab in ORDER:
        csv_t, sum_t, _ = SRC[lab]
        df = pd.read_csv(csv_t.format(ev=ev))
        posteriors[(ev, lab)] = df
        qm, qlo, qhi = q68(df["mass_ratio"])
        dm, dlo, dhi = q68(df["luminosity_distance"])
        c1m = np.median(df["chi_1"]); c2m = np.median(df["chi_2"])
        ess, essf, npost = ess_of(sum_t.format(ev=ev))
        rows.append(dict(
            event=ev, variant=lab,
            q_true=round(q_t, 4), q_med=round(qm, 4), q_bias=round(qm - q_t, 4),
            q16=round(qlo, 4), q84=round(qhi, 4),
            dL_true=round(dL_t, 1), dL_med=round(dm, 1), dL_ratio=round(dm / dL_t, 3),
            chi1_true=round(chi1_t, 4), chi1_med=round(c1m, 4),
            chi2_med=round(c2m, 4),
            ess=round(ess, 1) if np.isfinite(ess) else np.nan,
            ess_frac=round(essf, 3) if np.isfinite(essf) else np.nan,
        ))

tab = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 40)
print("\n================  BIASED VARIANT COMPARISON  (reweighted, full-likelihood)  ================\n")
for ev in EVENTS:
    print(f"----- event {ev:04d}  (q_true={tab[tab.event==ev].q_true.iloc[0]}, "
          f"dL_true={tab[tab.event==ev].dL_true.iloc[0]} Mpc) -----")
    sub = tab[tab.event == ev][["variant", "q_med", "q_bias", "q16", "q84",
                                "dL_med", "dL_ratio", "chi1_med", "chi2_med",
                                "ess", "ess_frac"]]
    print(sub.to_string(index=False))
    print()

out_csv = os.path.join(ROOT, "biased/biased_variant_comparison_summary.csv")
tab.to_csv(out_csv, index=False)
print(f"[*] wrote diagnostic table -> {out_csv}")


# ---------------- overlay corner per event (PE_SEOBNR_plot.ipynb style) ----------------
def make_result(df, params, label):
    priors = {}
    for p in params:
        x = df[p].dropna().to_numpy(dtype=float)
        lo, hi = (np.percentile(x, [0.1, 99.9]) if len(x) else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1e-6
        priors[p] = bilby.core.prior.Uniform(lo, hi, name=p)
    labels = [LABELS.get(p, p) for p in params]
    return bilby.core.result.Result(
        label=label, outdir=".", sampler="dynesty",
        search_parameter_keys=params, priors=priors,
        posterior=df[params].copy(),
        parameter_labels=labels, parameter_labels_with_unit=labels,
    )


def add_truth_lines(fig, params, truths):
    ndim = len(params)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i, yp in enumerate(params):
        yt = truths.get(yp)
        if yt is not None and np.isfinite(yt):
            axes[i, i].axvline(yt, color="red", lw=1.5)
        for j in range(i):
            xt = truths.get(params[j])
            if xt is not None and np.isfinite(xt):
                axes[i, j].axvline(xt, color="red", lw=1.0)
            if yt is not None and np.isfinite(yt):
                axes[i, j].axhline(yt, color="red", lw=1.0)
    return fig


def jitter_constant(df, params):
    # Only v2 fix-spin holds chi at a delta function; corner cannot form a 2D contour on
    # a zero-width column. A small jitter renders that fixed spin as a tight spike at its
    # value (clearly narrower than the v1/v3 spin posteriors, and centred on the truth
    # line), while v1 and v3 keep their real chi distributions. The <1e-10 threshold
    # catches float-noise "zeros" (~1e-19) that an ==0 test misses.
    df = df.copy()
    for p in params:
        x = df[p].to_numpy(dtype=float)
        if np.nanstd(x) < 1e-10:
            df[p] = x + np.random.normal(0.0, 2e-3, size=len(x))
    return df


# Two controlled-comparison groupings (each isolates ONE variable).
GROUPS = {
    "A_prioreffect_orig_vs_v1": (["orig prior", "v1 aligned"], ["C0", "C3"]),
    "B_spineffect_v1_v2_v3": (["v1 aligned", "v2 fix", "v3 flat"], ["C0", "C1", "C2"]),
}

for ev in EVENTS:
    truth = load_truth(ev)
    tdict = {p: float(truth.get(p, np.nan)) for p in PARAMS}
    for gkey, (labs, cols) in GROUPS.items():
        results = [make_result(jitter_constant(posteriors[(ev, lab)], PARAMS), PARAMS, lab)
                   for lab in labs]
        fig = bilby.core.result.plot_multiple(
            results, labels=labs, parameters=PARAMS, save=False, bins=30,
            plot_datapoints=False, smooth=0.9, quantiles=[0.16, 0.84], colors=cols,
        )
        add_truth_lines(fig, PARAMS, tdict)
        outpng = os.path.join(FIGDIR, f"fig_corner_biased_event_{ev:04d}_{gkey}.png")
        fig.savefig(outpng, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[*] saved {outpng}")

print("\nDONE.")
