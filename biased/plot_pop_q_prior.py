#!/usr/bin/env python
"""Presentation figure: the true mass-ratio (q) distribution of the injected
population (histogram) with the analytical single-event q-prior overlaid.

The prior is the exact one used by PE_biased.py in the deployment (v1) config:
a half-Gaussian peaked at q=1, sigma_q=0.0957, truncated to [q_min, 1] with
q_min=0.5 -- i.e. bilby TruncatedGaussian(mu=1, sigma=0.0957, min=0.5, max=1).
Instantiating the actual bilby prior guarantees the curve's normalization matches
what the sampler saw.

Run inside the `GW` conda env:
    python biased/plot_pop_q_prior.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bilby

RUNDIR = "/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
os.chdir(RUNDIR)
POP_JSONL = "outputs/outdir_population_SEOBNR/accepted.jsonl"
FIGDIR = "figures"

# Population NS-mass Gaussian (scripts/Injection_population_eosfit.py:96).
M_MU, M_SIGMA = 1.33, 0.09
# sigma_q is ANALYTIC, not fitted: error-propagate m ~ N(mu_m, sigma_m) through
# q = m2/m1 near q=1  ->  Var(q) ~ 2 sigma_m^2 / mu_m^2  ->  sigma_q = sqrt(2) sigma_m / mu_m.
SIGMA_Q = np.sqrt(2.0) * M_SIGMA / M_MU   # = 0.0957
Q_MIN = 0.5

# --- true q of every event in the population (q = m2/m1, frame-independent) ---
q_true = []
for line in open(POP_JSONL):
    d = json.loads(line)
    m1, m2 = float(d["mass_1_detector"]), float(d["mass_2_detector"])
    q_true.append(min(m1, m2) / max(m1, m2))
q_true = np.array(q_true)

# --- analytical prior (exact bilby object used in PE) ---
q_prior = bilby.core.prior.TruncatedGaussian(
    mu=1.0, sigma=SIGMA_Q, minimum=Q_MIN, maximum=1.0, name="mass_ratio")
qq = np.linspace(Q_MIN, 1.0, 2000)
pdf = q_prior.prob(qq)

# --- figure ---
fig, ax = plt.subplots(figsize=(6.0, 5.0))
bins = np.linspace(0.75, 1.0, 26)
ax.hist(q_true, bins=bins, density=True, color="C0", alpha=0.55,
        edgecolor="white", linewidth=0.6,
        label=f"population true $q$  ($N={len(q_true)}$)")
ax.plot(qq, pdf, color="k", lw=2.6,
        label=r"prior: $\sigma_q=\sqrt{2}\,\sigma_m/\mu_m=0.0957$")

ax.set_xlim(0.72, 1.005)
ax.set_xlabel(r"$q = m_2/m_1$", fontsize=16)
ax.set_ylabel("density", fontsize=16)
ax.tick_params(labelsize=13)
ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.set_title("Population mass-ratio distribution vs. single-event prior", fontsize=15)
fig.tight_layout()

out = os.path.join(FIGDIR, "fig_pop_q_true_vs_prior.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[*] saved {out}")
print(f"    true q: min={q_true.min():.3f} median={np.median(q_true):.3f} "
      f"max={q_true.max():.3f}; implied sigma={np.sqrt(np.mean((1-q_true)**2)):.4f}")
