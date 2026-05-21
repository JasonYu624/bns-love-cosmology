# gwtc3_cosmo.py
import os
import dill
import numpy as np
import pandas as pd
import bilby as bb
import gwpopulation as gwpop
import jax

from bilby.core.prior import PriorDict, Uniform
from gwpopulation.experimental.cosmo_models import CosmoModel
from gwpopulation.experimental.jax import JittedLikelihood
from wcosmo.astropy import Planck15
from wcosmo.utils import disable_units

disable_units()
gwpop.set_backend("jax")

def main():
    print("JAX devices:", jax.devices())

    # -------------------------
    # Load posteriors (GWTC-3 samples)
    # -------------------------
    posteriors = pd.read_pickle("gwtc-3-samples.pkl")
    del posteriors[15]
    del posteriors[38]

    for post in posteriors:
        zs = post.pop("redshift").values
        post["mass_1_detector"] = post.pop("mass_1") * (1 + zs)
        post["luminosity_distance"] = np.asarray(Planck15.luminosity_distance(zs))
        post["prior"] /= np.asarray(Planck15.dDLdz(zs) * (1 + zs))

    # -------------------------
    # Load injections (sensitivity)
    # -------------------------
    with open("gwtc-3-injections.pkl", "rb") as ff:
        injections = dill.load(ff)

    zs = np.asarray(injections.pop("redshift"))
    injections["mass_1_detector"] = injections.pop("mass_1") * (1 + zs)
    injections["luminosity_distance"] = np.asarray(Planck15.luminosity_distance(zs))
    injections["prior"] /= np.asarray(Planck15.dDLdz(zs) * (1 + zs))

    # -------------------------
    # Model + likelihood
    # -------------------------
    model = CosmoModel(
        model_functions=[
            gwpop.models.mass.two_component_primary_mass_ratio,
            gwpop.models.spin.iid_spin,
            gwpop.models.redshift.PowerLawRedshift(cosmo_model="FlatwCDM"),
        ],
        cosmo_model="FlatwCDM",
    )

    vt = gwpop.vt.ResamplingVT(model=model, data=injections, n_events=len(posteriors))
    likelihood = gwpop.hyperpe.HyperparameterLikelihood(
        posteriors=posteriors,
        hyper_prior=model,
        selection_function=vt,
    )

    # -------------------------
    # Priors
    # -------------------------
    priors = PriorDict()
    priors["alpha"] = Uniform(minimum=-2, maximum=4, latex_label=r"$\alpha$")
    priors["beta"] = Uniform(minimum=-4, maximum=12, latex_label=r"$\beta$")
    priors["mmin"] = Uniform(minimum=2, maximum=2.5, latex_label=r"$m_{\min}$")
    priors["mmax"] = Uniform(minimum=80, maximum=100, latex_label=r"$m_{\max}$")
    priors["lam"] = Uniform(minimum=0, maximum=1, latex_label=r"$\lambda_{m}$")
    priors["mpp"] = Uniform(minimum=10, maximum=50, latex_label=r"$\mu_{m}$")
    priors["sigpp"] = Uniform(minimum=1, maximum=10, latex_label=r"$\sigma_{m}$")
    priors["gaussian_mass_maximum"] = 100

    priors["amax"] = 1
    priors["alpha_chi"] = Uniform(minimum=1, maximum=6, latex_label=r"$\alpha_{\chi}$")
    priors["beta_chi"] = Uniform(minimum=1, maximum=6, latex_label=r"$\beta_{\chi}$")
    priors["xi_spin"] = Uniform(minimum=0, maximum=1, latex_label=r"$\xi$")
    priors["sigma_spin"] = Uniform(minimum=0.3, maximum=4, latex_label=r"$\sigma$")

    priors["H0"] = Uniform(minimum=20, maximum=200, latex_label=r"$H_0$")
    priors["Om0"] = Uniform(minimum=0, maximum=1, latex_label=r"$\Omega_{m,0}$")
    priors["w0"] = Uniform(minimum=-1.5, maximum=-0.5, latex_label=r"$w_0$")
    priors["lamb"] = Uniform(minimum=-1, maximum=10, latex_label=r"$\lambda_{z}$")

    # -------------------------
    # JIT compile
    # -------------------------
    parameters = priors.sample()
    _ = likelihood.log_likelihood_ratio(parameters)  # warmup
    jit_likelihood = JittedLikelihood(likelihood)
    _ = jit_likelihood.log_likelihood_ratio(parameters)  # compile

    # -------------------------
    # Run dynesty via bilby
    # -------------------------
    outdir = os.path.abspath("outdir")
    os.makedirs(outdir, exist_ok=True)

    result = bb.run_sampler(
        likelihood=jit_likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=100,
        label="cosmo",
        sample="acceptance-walk",
        naccept=5,
        save="hdf5",
        resume=False,
        outdir=outdir,
    )

    # -------------------------
    # Plot corner (save to file)
    # -------------------------
    fig = result.plot_corner(
        save=False,
        parameters=["H0", "Om0", "w0", "mpp", "lamb"],
        truths=[67.74, 0.3075, -1, np.nan, np.nan],
    )
    fig.savefig(os.path.join(outdir, "corner_cosmo.png"), dpi=200, bbox_inches="tight")

    # -------------------------
    # Post-processing variance checks
    # -------------------------
    func = jax.jit(likelihood.generate_extra_statistics)
    full_posterior = pd.DataFrame(
        [func(p) for p in result.posterior.to_dict(orient="records")]
    ).astype(float)

    full_posterior.to_csv(os.path.join(outdir, "posterior_with_variance.csv"), index=False)
    print("Wrote:", os.path.join(outdir, "corner_cosmo.png"))
    print("Wrote:", os.path.join(outdir, "posterior_with_variance.csv"))

if __name__ == "__main__":
    main()
