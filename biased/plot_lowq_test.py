#!/usr/bin/env python
"""Low-q over-correction test: on two genuinely low-q events (0047, 0050),
overlay the ORIGINAL flat-q prior baseline against the v1 new q-prior
(reflected half-Gaussian, sigma_q=0.0957, peaked at q=1).

Plotting is IDENTICAL to notebooks/PE_SEOBNR_plot.ipynb: same make_result,
same add_truth_lines, same bilby.plot_multiple call (bins=30,
plot_datapoints=False, smooth=0.9, quantiles=[0.16,0.84], colors=['C0','C3']).
Nothing else is passed to plot_multiple.

Run inside the `GW` conda env:
    python biased/plot_lowq_test.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bilby

RUNDIR = "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
os.chdir(RUNDIR)
POP_OUTDIR = "outputs/outdir_population_SEOBNR"
ORIG_OUTDIR = "outputs/outdir_population_run_SEOBNR"
NEW_OUTDIR = "biased/outputs/outdir_lowq_test"
FIGDIR = "figures"

EVENTS = [47, 50]

# --- label map (notebook LABEL_MAP + the spin/inclination labels this figure needs) ---
LABEL_MAP = {
    "mass_ratio": r"$q$", "chirp_mass": r"$\mathcal{M}_c$",
    "luminosity_distance": r"$d_L$", "H0_sample": r"$H_0$",
    "lambda_tilde": r"$\tilde{\Lambda}$", "delta_lambda_tilde": r"$\delta\tilde{\Lambda}$",
    "delta_a0": r"$\delta a_0$", "delta_a1": r"$\delta a_1$", "delta_a2": r"$\delta a_2$",
    "chi_1": r"$\chi_{1z}$", "chi_2": r"$\chi_{2z}$", "theta_jn": r"$\theta_{JN}$",
}

CORNER_PARAMS = ["mass_ratio", "chirp_mass", "luminosity_distance",
                 "chi_1", "chi_2", "theta_jn"]


# ---- verbatim from notebooks/PE_SEOBNR_plot.ipynb ----
def make_result(df, params, label):
    priors = {}
    for p in params:
        x = df[p].dropna().to_numpy(dtype=float)
        lo, hi = np.percentile(x, [0.1, 99.9]) if len(x) > 0 else (0, 1)
        priors[p] = bilby.core.prior.Uniform(lo, hi, name=p)
    labels = [LABEL_MAP.get(p, p) for p in params]
    return bilby.core.result.Result(
        label=label, outdir='.', sampler='dynesty',
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
            axes[i, i].axvline(yt, color='red', lw=1.5)
        for j in range(i):
            xp = params[j]; xt = truths.get(xp)
            if xt is not None and np.isfinite(xt):
                axes[i, j].axvline(xt, color='red', lw=1.0)
            if yt is not None and np.isfinite(yt):
                axes[i, j].axhline(yt, color='red', lw=1.0)
    return fig
# ------------------------------------------------------


def load_truth(ev):
    inj = json.load(open(os.path.join(POP_OUTDIR, f"event_{ev:04d}", "meta.json")))["injection_parameters"]
    m1, m2 = float(inj["mass_1"]), float(inj["mass_2"])
    return {
        "mass_ratio": min(m1, m2) / max(m1, m2),
        "chirp_mass": (m1 * m2) ** (3. / 5.) / (m1 + m2) ** (1. / 5.),
        "luminosity_distance": float(inj["luminosity_distance"]),
        "chi_1": float(inj.get("chi_1", 0.0)),
        "chi_2": float(inj.get("chi_2", 0.0)),
        "theta_jn": float(inj.get("theta_jn", np.nan)),
    }


for ev in EVENTS:
    tag = f"bns_event_{ev:04d}_seobnr"
    df_orig = pd.read_csv(f"{ORIG_OUTDIR}/{tag}_reweighted_posterior_augmented.csv")
    df_new = pd.read_csv(f"{NEW_OUTDIR}/{tag}_reweighted_posterior_augmented.csv")
    truths = load_truth(ev)
    params = [p for p in CORNER_PARAMS if p in df_orig.columns and p in df_new.columns]

    r_orig = make_result(df_orig, params, 'orig prior')
    r_new = make_result(df_new, params, 'v1 new-prior')

    fig = bilby.core.result.plot_multiple(
        [r_orig, r_new],
        labels=['orig prior', 'v1 new-prior'],
        parameters=params, save=False, bins=30,
        plot_datapoints=False, smooth=0.9, quantiles=[0.16, 0.84],
        colors=['C0', 'C3'],
    )
    add_truth_lines(fig, params, truths)

    save_path = f"{FIGDIR}/fig_corner_lowq_event_{ev:04d}_orig_vs_v1.png"
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {save_path}  (true q={truths['mass_ratio']:.3f})")

print("DONE.")
