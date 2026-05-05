# Working Note: From Single-Event PE to Hierarchical Inference in the EOS-fit / Love-Siren Pipeline

## 1. Purpose of this note

This note summarizes the current understanding of the two-step pipeline we are using for binary neutron-star (BNS) standard-siren inference with an EOS-/UR-informed tidal model.

The goals are:

1. to state clearly what the **single-event PE** is doing,
2. to state clearly what the **hierarchical inference** is doing,
3. to explain why we chose this architecture,
4. to distinguish the **physically exact model** from the **recycling-compatible numerical surrogate** we actually implement,
5. to document the assumptions, approximations, Jacobians, and selection treatment, and
6. to map the mathematics onto the current scripts.

The intended audience is ourselves while developing and debugging the pipeline.

---

## 2. Big picture

Our current pipeline is:

1. **Single-event PE**
   - For each event, we sample detector-frame CBC parameters **and also**
     \(H_0\), \(\delta a_0\), \(\delta a_1\), \(\delta a_2\).
   - For every sample, we use \((d_L, H_0)\) to infer \(z\), then use \(m^{\rm det}/(1+z)\) to infer source-frame masses, then use the current EOS-fit / UR parameterization to compute \(\lambda_1, \lambda_2\), and therefore the tidal phase parameters.

2. **Hierarchical inference**
   - We recycle the posterior samples from step 1.
   - We fit a source-frame mass population model and a redshift population model.
   - Because we are using **posterior-sample recycling**, not a continuous single-event likelihood representation (such as a GMM), we do **not** directly fit strict shared constants \(H_0\) and \((\delta a_0,\delta a_1,\delta a_2)\).
   - Instead, in the current recycling-compatible implementation, we model these event-level sampled quantities with narrow or broader **hyper-distributions**, e.g. \((\mu_H,\sigma_H)\), \((\mu_{a_0},\sigma_{a_0})\), etc.

So the current architecture is a **two-step PE + hierarchical recycling** pipeline, with the hierarchical stage implemented as a **continuous surrogate** to the physically exact shared-parameter model.

---

## 3. Scientific motivation

### 3.1 Why Love sirens work

For BNS signals:

- the point-particle inspiral phase measures **detector-frame** masses,
- the tidal sector depends on **source-frame** masses through the tidal deformability,
- therefore the tidal sector can help break the mass-redshift degeneracy.

Schematically,

\[
m_A^{\rm det} = (1+z)\, m_A^{\rm src},
\qquad A\in\{1,2\}.
\]

If the waveform amplitude gives information about \(d_L\), and the tidal sector constrains source-frame masses through a low-dimensional EOS-/UR-informed model, then the event contains information about the distance-redshift relation and therefore about \(H_0\).

This is the physical idea behind the Messenger--Read / Binary-Love / Cosmology-with-Love line of work.

### 3.2 Why we sample \(H_0\) already at the PE stage

We deliberately include \(H_0\) and the three EOS-fit / UR parameters in the **single-event** PE because this preserves, inside each event posterior, the joint correlations among

\[
(H_0,\delta a_0,\delta a_1,\delta a_2,d_L,z,m_1^{\rm src},m_2^{\rm src},\lambda_1,\lambda_2,\tilde\Lambda,\delta\tilde\Lambda).
\]

This is important.

If we did **not** sample \(H_0\) at the event level, then at the hierarchical stage we would have to reconstruct that coupling indirectly from posterior samples in other coordinates. Sampling \(H_0\) already in PE makes the event-level posterior geometry much closer to the actual physical structure we want to stack later.

### 3.3 Why we still need a cosmology / Jacobian treatment in the hierarchical stage

Even though \(H_0\) is already sampled at the event level, the hierarchical model is written in terms of **source-frame** population variables such as

\[
(m_1^{\rm src}, m_2^{\rm src}, z).
\]

The posterior samples are naturally stored in PE coordinates such as

\[
(\mathcal{M}_c^{\rm det}, q, d_L, H_0, \delta a_0, \delta a_1, \delta a_2, \ldots).
\]

Therefore the hierarchical stage still needs the transformation between these coordinate systems, including the Jacobian factor involving

\[
\frac{\partial z}{\partial d_L}
\quad\text{or equivalently}\quad
\left(\frac{\partial d_L}{\partial z}\right)^{-1}.
\]

So the hierarchical stage is **not** "purely statistical" in the sense of merely fitting sample means and variances. It still contains cosmology-dependent frame conversion and Jacobian factors.

---

## 4. The current EOS-fit / UR parameterization

Our current tidal parameterization is

\[
\lambda(\bar M;\delta a_0,\delta a_1,\delta a_2)
=
\frac{3500}{\bar M^5}
\left[
1
+
a_0(1+\delta a_0)
+
a_1(1+\delta a_1)\bar M
+
a_2(1+\delta a_2)\bar M^2
\right],
\]

where

\[
\bar M \equiv \frac{m^{\rm src}}{M_\odot},
\]

and the baseline coefficients \((a_0,a_1,a_2)\) are fixed by the adopted fit.

For each component,

\[
\lambda_A = \lambda(\bar M_A;\delta a_0,\delta a_1,\delta a_2),
\qquad A\in\{1,2\}.
\]

Then

\[
\tilde\Lambda = \tilde\Lambda(\lambda_1,\lambda_2,m_1^{\rm det},m_2^{\rm det}),
\]

\[
\delta\tilde\Lambda = \delta\tilde\Lambda(\lambda_1,\lambda_2,m_1^{\rm det},m_2^{\rm det}).
\]

This is **not** the same algebraic parameterization as the older Taylor-expansion-based Binary-Love notes using \(\bar\lambda_0^{(0)}\) and \(\bar\lambda_0^{(k)}\), but it serves the same overall purpose: compress the tidal sector into a small number of parameters tied to source-frame masses.

---

## 5. Single-event PE: exact forward model

### 5.1 Sampled PE variables

For event \(i\), the sampled PE coordinates are

\[
x_i = \bigl(\mathcal{M}_{c,i}^{\rm det}, q_i, d_{L,i}, H_{0,i}, \delta a_{0,i}, \delta a_{1,i}, \delta a_{2,i}, \xi_i\bigr),
\]

where \(\xi_i\) collects all other nuisance variables.

### 5.2 Detector-frame masses

From detector-frame chirp mass and mass ratio,

\[
m_{1,i}^{\rm det} = \mathcal{M}_{c,i}^{\rm det}(1+q_i)^{1/5} q_i^{-3/5},
\]

\[
m_{2,i}^{\rm det} = q_i \, m_{1,i}^{\rm det}.
\]

### 5.3 Redshift from \((d_L,H_0)\)

We adopt a cosmological model with fixed \(\Omega_m\) and \(w_0\), and allow \(H_0\) to vary. Then for each posterior sample,

\[
z_i = \hat z(d_{L,i};H_{0,i}).
\]

In the PE script, this is implemented with a fiducial cosmology grid in \(z\) and \(d_L\), using the fact that for fixed \(\Omega_m\) and \(w_0\), changing \(H_0\) is effectively an overall scaling of the distance-redshift relation.

### 5.4 Source-frame masses

\[
m_{A,i}^{\rm src} = \frac{m_{A,i}^{\rm det}}{1+z_i},
\qquad A\in\{1,2\}.
\]

### 5.5 Tidal parameters from source-frame masses

Using the EOS-fit / UR model,

\[
\lambda_{A,i} = \lambda(m_{A,i}^{\rm src};\delta a_{0,i},\delta a_{1,i},\delta a_{2,i}).
\]

Then

\[
\tilde\Lambda_i = \tilde\Lambda(\lambda_{1,i},\lambda_{2,i},m_{1,i}^{\rm det},m_{2,i}^{\rm det}),
\]

\[
\delta\tilde\Lambda_i = \delta\tilde\Lambda(\lambda_{1,i},\lambda_{2,i},m_{1,i}^{\rm det},m_{2,i}^{\rm det}).
\]

### 5.6 Event-level likelihood and posterior

The event-level likelihood is

\[
\mathcal{L}_i(d_i\mid x_i)=p(d_i\mid x_i).
\]

The single-event PE posterior is

\[
p_{\rm PE}(x_i\mid d_i)
=
\frac{\mathcal{L}_i(d_i\mid x_i)\,\pi_{\rm PE}(x_i)}{\mathcal{Z}^{\rm PE}_i}.
\]

In practice, we save posterior samples and also save or reconstruct the event-level prior density sample by sample. In the current pipeline this is done through the saved `log_prior` column in the augmented posterior CSVs.

### 5.7 PE implementation details in our script

The current PE script:

- injects exact detector-frame frequency-domain signals,
- uses a relative-binning likelihood for fast PE,
- samples \(H_0\) and \(\delta a_k\) directly,
- augments the posterior with `redshift_sample`, `mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2`, etc.,
- then reweights from relative-binning to full likelihood using posterior-sample reweighting.

The current PE output used by the hierarchical stage is the reweighted augmented posterior CSV.

---

## 6. The physically exact hierarchical target model

The physically exact model we ultimately care about is one in which all events share the same universal cosmological and EOS-/UR parameters:

\[
H_{0,1}=H_{0,2}=\cdots=H_0,
\]

\[
(\delta a_{0,1},\delta a_{1,1},\delta a_{2,1})
=
(\delta a_{0,2},\delta a_{1,2},\delta a_{2,2})
=
\cdots
=
(\delta a_0,\delta a_1,\delta a_2).
\]

In that exact model, the catalog posterior would be

\[
p(\Lambda\mid \{d_i\},{\rm det})
\propto
p(\Lambda)\,\beta(\Lambda)^{-N}
\prod_{i=1}^N
\int dx_i\,\mathcal{L}_i(d_i\mid x_i)\,\Pi_i(x_i\mid \Lambda),
\]

where \(\Lambda\) includes the shared cosmological parameters, shared EOS-/UR parameters, and population hyperparameters, and \(\Pi_i\) contains **Dirac delta constraints** enforcing event-to-event equality of the shared quantities.

Schematically,

\[
\Pi_i \propto
\delta(H_{0,i}-H_0)
\prod_{k=0}^2 \delta(\delta a_{k,i}-\delta a_k)
\times \pi_{\rm pop}(m_{1,i}^{\rm src},m_{2,i}^{\rm src},z_i,\xi_i\mid \lambda_{\rm pop}).
\]

This is the physically clean model.

---

## 7. Why we do **not** implement the exact shared-parameter model directly

Because we are using **posterior-sample recycling**, not a continuous representation of the single-event likelihood.

The gwpopulation-style Monte Carlo recycling identity is

\[
\widehat{\mathcal{L}}_i(d_i\mid \Lambda)
\approx
\frac{1}{K_i}
\sum_{j=1}^{K_i}
\frac{\pi(x_{ij}\mid \Lambda)}{\pi_{\rm PE}(x_{ij})}.
\]

This requires the hierarchical model \(\pi(x\mid \Lambda)\) to define a **continuous, evaluable density** at each posterior sample.

If we use strict shared constants, then \(\pi(x\mid\Lambda)\) contains Dirac delta functions in \(H_0\) and \(\delta a_k\). For finite posterior samples, this is not numerically usable in ordinary sample averaging.

Therefore, with recycling alone, one cannot directly implement the strict shared-constant hierarchy unless one first replaces each single-event likelihood with a continuous approximation, such as:

- a Gaussian mixture model (GMM),
- a KDE,
- a flow,
- or some other explicit continuous density model.

This is exactly why our current hierarchical implementation uses a **continuous surrogate**.

---

## 8. The recycling-compatible hierarchical model we actually use

### 8.1 Hyper-distribution surrogate

Instead of strict shared constants, we model event-level sampled values of \(H_0\) and \(\delta a_k\) with hyper-distributions:

\[
H_{0,i} \sim \pi_H(H_{0,i}\mid \mu_H,\sigma_H),
\]

\[
\delta a_{k,i} \sim \pi_{a_k}(\delta a_{k,i}\mid \mu_{a_k},\sigma_{a_k}),
\qquad k=0,1,2.
\]

The source-frame mass population is modeled by

\[
(m_{1,i}^{\rm src},m_{2,i}^{\rm src}) \sim \pi_m(\cdot\mid \Theta_m),
\]

and the redshift population by

\[
z_i \sim \pi_z(z_i\mid \Theta_z).
\]

So the hierarchical hyperparameters are

\[
\Lambda
=
(\Theta_m,\Theta_z,
\mu_H,\sigma_H,
\mu_{a_0},\sigma_{a_0},
\mu_{a_1},\sigma_{a_1},
\mu_{a_2},\sigma_{a_2}).
\]

### 8.2 Practical interpretation

This is **not** the exact physical statement that different events truly have different values of \(H_0\) or \(\delta a_k\).

Instead, this is a **recycling-compatible continuous surrogate** to the exact shared-parameter model.

- In the ideal shared-parameter picture, \(\sigma_H\to 0\) and \(\sigma_{a_k}\to 0\).
- In the implemented recycling picture, finite nonzero widths make the model numerically evaluable on discrete posterior samples.

So the current hierarchical model should be interpreted as:

> a continuous approximation to the exact universal-parameter hierarchy, chosen because we are using posterior-sample recycling rather than a continuous single-event likelihood representation.

### 8.3 Event model density

For each sample, the hierarchical event density is

\[
\pi_{\rm astro}(x_i\mid \Lambda)
=
\pi_m(m_{1,i}^{\rm src},m_{2,i}^{\rm src}\mid \Theta_m)
\,\pi_z(z_i\mid \Theta_z)
\,\pi_H(H_{0,i}\mid \mu_H,\sigma_H)
\prod_{k=0}^2 \pi_{a_k}(\delta a_{k,i}\mid \mu_{a_k},\sigma_{a_k})
\,\pi_\xi(\xi_i)
\,J_i.
\]

Here \(J_i\) is the Jacobian taking us from the PE coordinates to the source-frame population coordinates.

### 8.4 How the two \(H_0\)-sensitive pieces combine

There are two distinct sources of information about \(H_0\).

First, the single-event PE likelihood already constrains \(H_0\) through the Love-siren forward model:

\[
(d_L,H_0)\to z\to (m_1^{\rm src},m_2^{\rm src})\to
(\lambda_1,\lambda_2)\to h(f).
\]

Second, the hierarchical recycling stage reweights the same correlated posterior samples by the population factors

\[
\pi_H(H_{0,i})\,\pi_z(z_i)\,\pi_m(m_{1,i}^{\rm src},m_{2,i}^{\rm src})\,J_i.
\]

Thus the mass and redshift population model also constrains \(H_0\), but it does so through the already-sampled correlations among
\((H_0,d_L,z,m^{\rm src},\lambda)\). No additional bilby-style conversion function is needed at the hierarchical stage as long as the PE output CSV already stores the derived quantities and the hierarchical density includes the correct coordinate Jacobian.

---

## 9. The detector-frame to source-frame Jacobian

The PE samples are naturally stored in detector-frame-inspired variables such as

\[
(\mathcal{M}_c^{\rm det},q,d_L,H_0,\delta a_0,\delta a_1,\delta a_2,\xi),
\]

while the source population model is written in terms of

\[
(m_1^{\rm src},m_2^{\rm src},z,H_0,\delta a_0,\delta a_1,\delta a_2,\xi).
\]

Therefore,

\[
J
=
\left|
\frac{\partial(m_1^{\rm src},m_2^{\rm src},z,H_0,\delta a_0,\delta a_1,\delta a_2,\xi)}
{\partial(\mathcal{M}_c^{\rm det},q,d_L,H_0,\delta a_0,\delta a_1,\delta a_2,\xi)}
\right|.
\]

Because \(H_0\), \(\delta a_k\), and \(\xi\) are unchanged in this transformation, the nontrivial part is

\[
J
=
\left|
\frac{\partial(m_1^{\rm src},m_2^{\rm src},z)}
{\partial(\mathcal{M}_c^{\rm det},q,d_L)}
\right|_{H_0}.
\]

In the implementation, this appears as

\[
J =
\left|\frac{\partial(m_1^{\rm det},m_2^{\rm det})}
{\partial(\mathcal{M}_c^{\rm det},q)}\right|
\times
\frac{dz}{dd_L}\times \frac{1}{(1+z)^2},
\]

up to the chosen storage convention for mass variables.

For our current sampled mass coordinates,

\[
\left|\frac{\partial(m_1^{\rm det},m_2^{\rm det})}
{\partial(\mathcal{M}_c^{\rm det},q)}\right|
=
\mathcal{M}_c^{\rm det}(1+q)^{2/5}q^{-6/5}.
\]

In our current hierarchical script, this is represented explicitly with a fiducial distance-redshift grid. For fixed \(\Omega_m\) and \(w_0\),

\[
d_L(z;H_0)
=
\frac{H_{0,\rm fid}}{H_0}\,
d_L(z;H_{0,\rm fid}),
\]

and

\[
\frac{dz}{dd_L}(z;H_0)
=
\frac{H_0}{H_{0,\rm fid}}
\left[
\frac{d d_L(z;H_{0,\rm fid})}{dz}
\right]^{-1}.
\]

The derivative \(d d_L(z;H_{0,\rm fid})/dz\) is precomputed on a grid and evaluated by interpolation. This avoids relying on a cosmology object that may not accept a vector of event-level \(H_0\) samples inside the JAX/gwpopulation recycling path.

The mass-frame part is

\[
\frac{\partial(m_1^{\rm src},m_2^{\rm src})}{\partial(m_1^{\rm det},m_2^{\rm det})} = \frac{1}{(1+z)^2}.
\]

This is why the hierarchical code still needs a cosmology/Jacobian treatment even though \(H_0\) is already sampled at the PE stage.

---

## 10. Posterior-sample recycling identity

For event \(i\),

\[
p_{\rm PE}(x_i\mid d_i)
=
\frac{\mathcal{L}_i(d_i\mid x_i)\,\pi_{\rm PE}(x_i)}{\mathcal{Z}_i^{\rm PE}}.
\]

Therefore,

\[
\mathcal{L}_i(d_i\mid x_i)
=
\frac{\mathcal{Z}_i^{\rm PE}\,p_{\rm PE}(x_i\mid d_i)}{\pi_{\rm PE}(x_i)}.
\]

Insert this into the hierarchical event integral:

\[
\mathcal{L}_i^{\rm hier}(\Lambda)
=
\int dx_i\,\mathcal{L}_i(d_i\mid x_i)\,\pi_{\rm astro}(x_i\mid \Lambda).
\]

Then

\[
\mathcal{L}_i^{\rm hier}(\Lambda)
=
\mathcal{Z}_i^{\rm PE}
\,\mathbb{E}_{p_{\rm PE}(x_i\mid d_i)}
\left[
\frac{\pi_{\rm astro}(x_i\mid \Lambda)}{\pi_{\rm PE}(x_i)}
\right].
\]

Dropping the evidence term, which is constant with respect to \(\Lambda\),

\[
\mathcal{L}_i^{\rm hier}(\Lambda)
\propto
\mathbb{E}_{p_{\rm PE}(x_i\mid d_i)}
\left[
\frac{\pi_{\rm astro}(x_i\mid \Lambda)}{\pi_{\rm PE}(x_i)}
\right].
\]

With posterior samples \(\{x_{ij}\}_{j=1}^{K_i}\), this becomes

\[
\widehat{\mathcal{L}}_i^{\rm hier}(\Lambda)
=
\frac{1}{K_i}
\sum_{j=1}^{K_i}
\frac{\pi_{\rm astro}(x_{ij}\mid \Lambda)}{\pi_{\rm PE}(x_{ij})}.
\]

This is the basic recycling formula used by `gwpopulation.hyperpe.HyperparameterLikelihood`.

### 10.1 Why storing `log_prior` is so useful

In our current pipeline, the denominator \(\pi_{\rm PE}(x_{ij})\) is read directly from the saved single-event PE output through the `log_prior` column.

This is much safer than trying to analytically reconstruct the full PE prior at the hierarchical stage, because the actual PE prior includes all the details of the event-level setup.

---

## 11. Population model used in the current hierarchical script

### 11.1 Mass model

At the moment the hierarchical source-frame mass spectrum is modeled as an ordered Gaussian evaluated inside the source-frame analysis support,

\[
\pi_m(m_1^{\rm src},m_2^{\rm src}\mid \mu_m,\sigma_m)
=
2\,p(m_1^{\rm src}\mid \mu_m,\sigma_m)\,p(m_2^{\rm src}\mid \mu_m,\sigma_m)
\,\Theta(m_1^{\rm src}-m_2^{\rm src}),
\]

with support restricted to the PE source-frame mass constraint interval used for recycling.

### 11.2 Redshift model

The redshift model is

\[
\pi_z(z\mid \gamma)
\propto
\frac{dV_c}{dz}(z;H_0,\Omega_m,w_0)
(1+z)^{\gamma-1},
\qquad 0\le z\le z_{\max}.
\]

In the current script, because \(\Omega_m\) and \(w_0\) are fixed and the normalized redshift distribution is precomputed, the effective redshift PDF used in the hierarchical stage is independent of the event-level \(H_0\) samples after normalization. This is a practical simplification consistent with the current implementation.

### 11.3 Hyper-distributions for \(H_0\) and \(\delta a_k\)

The current `hier_eosfit_hyper.py` models

- \(H_{0,i}\) with a Gaussian hyper-distribution,
- each \(\delta a_{k,i}\) with a Gaussian hyper-distribution.

This is the direct implementation of the recycling-compatible surrogate described above.

---

## 12. Selection effects

### 12.1 Exact selection factor

The exact selection factor is

\[
\beta(\Lambda)
=
\int d\psi\,P_{\rm det}(\psi)\,\pi_{\rm astro}(\psi\mid \Lambda),
\]

where \(\psi\) denotes the full latent source parameter set.

### 12.2 Monte Carlo estimator

If injections are drawn from a proposal distribution \(\pi_{\rm inj}(\psi)\), then

\[
\widehat{\beta}(\Lambda)
=
\frac{1}{N_{\rm inj}}
\sum_{r\in\mathrm{found}}
\frac{\pi_{\rm astro}(\psi_r\mid \Lambda)}{\pi_{\rm inj}(\psi_r)}.
\]

### 12.3 What we currently do

The current selection-generation scripts generate injections in source-frame masses and redshift and then keep detected injections in a merged file. In the current hierarchical implementation:

- the VT term is evaluated in **source-frame** variables,
- it depends on the source-frame mass population and redshift population,
- it does **not** currently depend on the hyper-distributions of \(H_0\) and \(\delta a_k\).

This is an approximation.

It should be interpreted as:

> detectability is modeled as being driven by the source-frame mass-redshift population, while the \(H_0\) and \(\delta a_k\) hyper-distributions are treated as recycling-level surrogate structure rather than explicit drivers of detectability.

This is consistent with the current scripts and with the fact that the selection injections were generated at fixed fiducial cosmology and fixed tidal-fit coefficients.

---

## 13. Final hierarchical posterior used in practice

Putting everything together, the current working hierarchical posterior is

\[
p(\Lambda\mid \{d_i\},{\rm det})
\propto
p(\Lambda)
\,\widehat{\beta}(\Lambda)^{-N}
\prod_{i=1}^N
\left[
\frac{1}{K_i}
\sum_{j=1}^{K_i}
\frac{\pi_{\rm astro}(x_{ij}\mid \Lambda)}{\pi_{\rm PE}(x_{ij})}
\right].
\]

Here:

- \(\pi_{\rm astro}(x_{ij}\mid\Lambda)\) contains
  - the source-frame mass model,
  - the redshift model,
  - the hyper-distributions for \(H_0\) and \(\delta a_k\),
  - the Jacobian factor,
- \(\pi_{\rm PE}(x_{ij})\) is taken from the single-event `log_prior`,
- \(\widehat{\beta}(\Lambda)\) is the VT correction from the injection campaign.

---

## 14. What is mathematically rigorous here, and what is approximate

### 14.1 Rigorous parts

The following statements are rigorous:

1. The event-level forward model:
   \((d_L,H_0)\to z\to m^{\rm src}\to \lambda_A\to (\tilde\Lambda,\delta\tilde\Lambda)\).
2. The use of posterior-sample recycling:
   \[
   \widehat{\mathcal{L}}_i \sim \frac{1}{K_i}\sum_j \frac{\pi_{\rm astro}}{\pi_{\rm PE}}.
   \]
3. The need for Jacobian factors when the population model is defined in source-frame variables but the PE samples live in different coordinates.
4. The need for a selection correction at the catalog level.

### 14.2 Controlled approximations

The following are approximations or modeling choices:

1. **Continuous surrogate for exact shared constants**
   - Physically, \(H_0\) and \(\delta a_k\) should be shared parameters.
   - Numerically, because we use posterior-sample recycling, we replace strict Dirac constraints by continuous hyper-distributions.

2. **VT independence from \((\mu_H,\sigma_H,\mu_{a_k},\sigma_{a_k})\)**
   - This is a practical approximation tied to the way the selection injections were generated.

3. **Fixed cosmological background apart from \(H_0\)**
   - We keep \(\Omega_m\) and \(w_0\) fixed in the current implementation.

4. **Current population model choice**
   - We currently use an ordered Gaussian in source-frame masses, evaluated inside the PE source-frame mass support.
   - This is a modeling choice, not a theorem.

---

## 15. Relationship to earlier internal notes

Earlier notes often used the Taylor-expansion / Binary-Love parameterization

\[
\Lambda(m)=\sum_k \frac{\bar\lambda_0^{(k)}}{k!}\left(1-\frac{m}{m_0}\right)^k,
\]

with quasi-universal relations among the coefficients. That is a valid and historically standard formulation.

Our current pipeline instead uses the low-dimensional EOS-fit / UR parameterization in \((\delta a_0,\delta a_1,\delta a_2)\). The physical role is the same:

- tidal information depends on source-frame mass,
- this couples the tidal sector to cosmology through \(m^{\rm det}=(1+z)m^{\rm src}\),
- and therefore Love-siren inference becomes possible.

The main implementation difference from many earlier notes is that we now sample \(H_0\) already at the event level and then perform hierarchical recycling on those posteriors.

---

## 16. Relationship to the current scripts

### 16.1 Single-event PE script

Current script family: `PE_eosfit_reweight.py`

This script:

- reads injected exact detector-frame FD signals,
- constructs a relative-binning likelihood,
- samples `H0_sample`, `delta_a0`, `delta_a1`, `delta_a2`,
- derives source-frame and tidal quantities per sample,
- saves augmented posterior CSVs,
- reweights from RB likelihood to full likelihood.

Current implementation note:

- The current PE script uses bilby’s standard relative-binning likelihood class directly (no instance monkey-patch and no custom safe wrapper).

### 16.2 Hierarchical script

Current preferred script: `hier_eosfit_hyper.py`

This script:

- loads the reweighted augmented posterior CSVs,
- reads the event-level prior from `log_prior`,
- evaluates a source-frame population model,
- evaluates hyper-distributions on `H0_sample` and `delta_a*`,
- includes `dz/ddL` and `1/(1+z)^2`,
- uses `gwpopulation.hyperpe.HyperparameterLikelihood`,
- uses JAX backend and `JittedLikelihood`,
- uses source-frame selection injections via `ResamplingVT`.

This is the script that is most consistent with our current conceptual understanding.

---

## 17. What would change if we wanted the exact shared-constant model

If we wanted to implement the exact physical model with a single shared

\[
H_0,\quad \delta a_0,\quad \delta a_1,\quad \delta a_2,
\]

then we would need a continuous representation of each single-event likelihood, for example via:

- GMM,
- KDE,
- normalizing flow,
- or another explicit density estimator.

Then the hierarchical stage could use exact shared-parameter constraints rather than hyper-distribution surrogates.

This is conceptually cleaner, but requires more custom machinery.

---

## 18. Why the current approach is reasonable

The current approach is reasonable because:

1. it keeps the correct event-level physics in the PE stage,
2. it preserves the important \((H_0,d_L,z,m^{\rm src},\lambda)\) correlations,
3. it is compatible with standard posterior-sample recycling machinery,
4. it keeps the hierarchical stage GPU/JAX-friendly,
5. it avoids the immediate need for a custom continuous single-event likelihood model,
6. it is straightforward to debug because the event-level and catalog-level pieces are separated cleanly.

The main price we pay is that the hierarchical model is currently a **surrogate** to the exact shared-universal-parameter model, rather than that exact model itself.

---

## 19. Recommended language for ourselves and for future writeups

A precise short description of the current pipeline is:

> We perform single-event PE in an enlarged parameter space that includes \(H_0\) and the three EOS-fit / UR nuisance parameters. We then perform hierarchical inference by posterior-sample recycling in source-frame variables, fitting a mass population model and redshift population model together with continuous hyper-distributions for the event-level sampled values of \(H_0\) and the EOS-fit parameters. This hierarchical model is a recycling-compatible continuous surrogate to the exact shared-parameter Love-siren model.

This wording avoids two common mistakes:

1. saying that the hierarchical stage is "just fitting means and variances" and therefore no longer involves cosmology/Jacobians,
2. saying that the current hierarchical script already implements the exact shared-constant model.

---

## 20. Immediate practical checklist

When running the current pipeline, verify all of the following.

### Single-event PE

- The PE script outputs `*_reweighted_posterior_augmented.csv`.
- These files contain at least:
  - `luminosity_distance`
  - `H0_sample`
  - `delta_a0`, `delta_a1`, `delta_a2`
  - `redshift_sample`
  - `mass_1_source`, `mass_2_source`
  - `log_prior`
- Reweighting from relative-binning to full likelihood succeeded and produced healthy effective sample sizes.

### Hierarchical stage

- `PE_POST_GLOB` matches the reweighted augmented CSVs.
- `SEL_MERGED_NPZ` and `SEL_MERGED_SUMMARY` point to the correct selection campaign.
- The selection mass proposal settings used in the hierarchical script match those used to generate the injections.
- The chosen mass-population model support matches the intended astrophysical support.
- `MAX_SAMPLES` is high enough that the recycled likelihood is stable.
- The `log_prior` denominator is finite and positive for all retained samples.

### Interpretation

- Remember that `hier_eosfit_hyper.py` is the preferred current script.
- Remember that it implements the recycling-compatible hyper-distribution surrogate, not the exact shared-constant model.

---

## 21. References to keep in mind

Key conceptual references:

- Messenger & Read: tidal information as a route to redshift measurement from BNS signals.
- Yagi & Yunes: Binary Love relations and universal relations.
- Chatterjee et al.: Love-siren \(H_0\) inference with universal relations.
- Ghosh et al.: source-frame hierarchical inference with Jacobians, PE-prior division, and selection correction.
- `gwpopulation`: posterior-sample recycling likelihood.
- `wcosmo`: detector/source-frame conversion and \(d d_L/dz\).
- `bilby`: single-event CBC PE and relative-binning likelihood.

---

## 22. Bottom line

The current pipeline is best thought of as:

1. **physically faithful single-event PE**, followed by
2. **recycling-based hierarchical stacking in source-frame variables**, implemented through
3. a **continuous hyper-distribution surrogate** for the exact shared-universal-parameter Love-siren model.

This is the correct mental model to use when interpreting both the scripts and the mathematics.
