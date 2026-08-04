---
title: "A Working Tutorial on Diffusion Models"
excerpt: "Diffusion from the ground up: VAE/ELBO, DDPM, score matching, DSM/Tweedie, SDEs, Fokker–Planck, and Anderson's reverse-time equation."
date: 2026-08-03
last_modified_at: 2026-08-04
---

These are my notes on diffusion models. I loosely mirror the canonical organization of this reference ([arXiv:2510.21890](https://arxiv.org/abs/2510.21890)).
This post introduces discrete-time Markov chains in DDPMs
as a complement to the encoders in (H)VAEs, whose ELBO decompositions are tractable under some
forward posterior. Score-matching (and contrastive divergence) avoids the partition constant in EBMs 
but requires computing the Jacobian gradient, which is especially intensive in high-dimensional spaces.
DSM avoids this and provides the motivation behind the perturbation kernel,
NCSN, and the Bayes-optimal quantities (noise, target, score) via Tweedie's.
Continuous-time formulations then configure the closed-form Gaussians for
VP, VE, and sub-VP SDEs for more stable generation. We provide the motivation behind 
the Fokker-Planck and continuity PDEs under Anderson's time-reversal, 
and why quadratic variation induces second-order terms via Itô's lemma,
which standard calculus is not equipped to consider.

---

# Latent-variable models

## VAE ELBO

Let $$x$$ be an observation and $$z$$ a latent variable with a joint model
$$p_\phi(x, z)$$ and an approximate posterior $$q_\theta(z \mid x)$$. The identity
that everything rests on is a single application of Bayes' rule inside an
expectation:

$$
\begin{aligned}
\mathbb{E}_{z \sim q_\theta(\cdot \mid x)}\left[ \log \frac{p_\phi(x,z)}{q_\theta(z \mid x)} \right]
&= \mathbb{E}_{z \sim q_\theta(\cdot \mid x)}\left[ \log \frac{p_\phi(z \mid x)\, p(x)}{q_\theta(z \mid x)} \right] \\
&= -\,\mathrm{KL}\left(q_\theta(z\mid x)\,\|\,p_\phi(z\mid x)\right) + \log p(x) \\
&=: \mathcal{L}_{\mathrm{ELBO}}(x).
\end{aligned}
$$

Because the KL term is nonnegative, $$\mathcal{L}_{\mathrm{ELBO}}(x) \le \log p(x)$$
— hence *evidence lower bound*. Splitting the joint the other way, using
$$p_\phi(x,z) = p_\phi(x \mid z)\,p(z)$$, gives the form we actually optimize:

$$
\mathcal{L}_{\mathrm{ELBO}}(x)
= \underbrace{\mathbb{E}_{z \sim q_\theta(\cdot \mid x)}\left[\log p_\phi(x \mid z)\right]}_{\text{reconstruction}}
\;-\;
\underbrace{\mathrm{KL}\left(q_\theta(z \mid x)\,\|\,p(z)\right)}_{\text{prior matching}}.
$$

Taking the data expectation,

$$
\mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\mathcal{L}_{\mathrm{ELBO}}\right]
=
\mathbb{E}_{x \sim p_{\mathrm{data}},\, z \sim q_\theta(z \mid x)}\left[\log p_\phi (x \mid z)\right]
- \mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\mathrm{KL}\left(q_\theta(z \mid x)\,\|\,p(z)\right)\right].
$$

## From a hierarchical VAE to DDPM

A diffusion model is a hierarchical VAE with a *fixed* Gaussian forward chain
$$x_0 \to x_1 \to \cdots \to x_L$$ and no learned encoder. Write the variational
bound on the negative log-likelihood, factor the reverse model as
$$p_\phi(x_0, x_{1:L}) = p_{\mathrm{prior}}(x_L)\prod_{i=1}^{L} p_\phi(x_{i-1}\mid x_i)$$,
and expand the forward posterior with Bayes' rule conditioned on $$x_0$$. The full
chain of steps:

$$
\begin{aligned}
-\log p_\phi(x_0) &=
-\log \int p_{\phi}(x_0, x_{1:L})
\frac{q(x_{1:L} \mid x_0)}{q(x_{1:L} \mid x_0)} \; dx_{1:L}\\
&=
-\log \mathbb{E}_{q(x_{1:L} \mid x_0)} \left[ \frac{p_{\phi}(x_0, x_{1:L})}{q(x_{1:L} \mid x_0)} \right]\\
&\leq
-\mathbb{E}_{q(x_{1:L} \mid x_0)} \left[ \log p_{\phi}(x_0, x_{1:L}) - \log q(x_{1:L} \mid x_0) \right]\\
&=
-\mathbb{E}_{q(x_{1:L} \mid x_0)} \left[
\log p_{\mathrm{prior}}(x_{L}) + \log \frac{p_{\phi}(x_{0:{L-1}} \mid x_L)}{q(x_{1:L} \mid x_0)} \right]\\
&=
-\mathbb{E}_{q(x_{1:L} \mid x_0)} \left[
\log \frac{p_{\mathrm{prior}}(x_{L})}{q(x_L \mid x_0)} +
\log p_\phi(x_0 \mid x_1) - \cancel{\log q(x_0 \mid x_1, x_0)} +
\log \prod_{i=2}^{L} \frac{p_\phi(x_{i-1} \mid x_{i})}{ q(x_{i-1} \mid x_{i}, x_0)}\right]\\
&= \mathbb{E}_{q(x_{L} \mid x_0)} \left[ \log \frac{q(x_L \mid x_0)}{p_{\mathrm{prior}}(x_L)} \right]
+ \mathbb{E}_{q(x_1 \mid x_0)} \left[ -\log p_\phi(x_0 \mid x_1) \right]
+ \mathbb{E}_{q(x_{1:L} \mid x_0)} \left[ \sum_{i=2}^{L} \log \frac{q(x_{i-1} \mid x_{i}, x_0)}{p_\phi(x_{i-1} \mid x_{i})} \right]\\
&=
 D_{\mathrm{KL}}\left(q(x_L \mid x_0) \,\|\, p_{\mathrm{prior}}(x_L)\right)
+ \mathbb{E}_{q(x_1 \mid x_0)} \left[-\log p_\phi(x_0 \mid x_1) \right]
+ \sum_{i=2}^{L} \mathbb{E}_{q(x_{i} \mid x_{0})}\left[ D_{\mathrm{KL}}\left(q(x_{i-1} \mid x_i, x_0) \,\|\, p_\phi(x_{i-1} \mid x_i)\right) \right]\\
 &=: \mathcal{L}_{\mathrm{prior}}(x_0) + \mathcal{L}_{\mathrm{recon}}(x_0; \phi) + \mathcal{L}_{\mathrm{diff}}(x_0; \phi).
\end{aligned}
$$

The telescoping in the fourth line uses
$$q(x_i \mid x_{i-1}) = q(x_i \mid x_{i-1}, x_0) = \dfrac{q(x_{i-1}\mid x_i, x_0)\,q(x_i \mid x_0)}{q(x_{i-1}\mid x_0)}$$;
the crossed-out term is the self-consistent $$q(x_0 \mid x_1, x_0)$$, which is a
point mass and contributes nothing. The intermediate $$q(x_i\mid x_0)/q(x_{i-1}\mid x_0)$$
factors telescope across the sum, leaving only the boundary term
$$q(x_L\mid x_0)$$ (folded into $$\mathcal{L}_{\mathrm{prior}}$$) and the per-step
posteriors.

The three pieces have clean interpretations: $$\mathcal{L}_{\mathrm{prior}}$$ asks
that the fully-noised data match the prior (no learnable parameters — it is
essentially zero for a well-chosen schedule); $$\mathcal{L}_{\mathrm{recon}}$$ is
the final denoising step back to data; and $$\mathcal{L}_{\mathrm{diff}}$$ is a sum
of *denoising matching* terms, one per timestep, each asking the learned reverse
kernel $$p_\phi(x_{i-1}\mid x_i)$$ to match the true forward posterior
$$q(x_{i-1}\mid x_i, x_0)$$.

## The Gaussian transition kernels

For DDPM the forward chain is Gaussian with a variance schedule
$$\{\beta_i\}$$; write $$\alpha_i = 1 - \beta_i$$ and $$\bar\alpha_i = \prod_{j=1}^{i}\alpha_j$$.
The single-step forward kernel, the closed-form marginal (from composing Gaussians),
and the $$x_0$$-conditioned reverse posterior (by conjugacy / completing the square)
are

$$
\begin{aligned}
q(x_i \mid x_{i-1}) &= \mathcal{N}\!\left(x_i;\; \sqrt{1-\beta_i}\,x_{i-1},\; \beta_i I\right), \\
q(x_i \mid x_0) &= \mathcal{N}\!\left(x_i;\; \sqrt{\bar\alpha_i}\,x_0,\; (1-\bar\alpha_i) I\right), \\
q(x_{i-1} \mid x_i, x_0) &= \mathcal{N}\!\left(x_{i-1};\; \tilde\mu_i(x_i, x_0),\; \tilde\beta_i I\right),
\end{aligned}
$$

with posterior mean and variance

$$
\tilde\mu_i(x_i, x_0) = \frac{\sqrt{\bar\alpha_{i-1}}\,\beta_i}{1-\bar\alpha_i}\,x_0 + \frac{\sqrt{\alpha_i}\,(1-\bar\alpha_{i-1})}{1-\bar\alpha_i}\,x_i,
\qquad
\tilde\beta_i = \frac{1-\bar\alpha_{i-1}}{1-\bar\alpha_i}\,\beta_i.
$$

Every KL in $$\mathcal{L}_{\mathrm{diff}}$$ is therefore between two Gaussians and
has a closed form, which is what makes the objective trainable.

{% comment %}
## Marginal consistency

The per-step objective conditions on $$x_0$$, but the *true* reverse kernel we
ultimately care about is the marginal $$p(x_{i-1}\mid x_i)$$, with no $$x_0$$. The
two are linked by a marginalization identity — the discrete analogue of
Chapman–Kolmogorov — which I'll call **marginal consistency**:

$$
p(x_i)\,p(x_{i-1}\mid x_i)
= \int p_{\mathrm{data}}(x_0)\, p(x_i \mid x_0)\, p(x_{i-1}\mid x_i, x_0)\; dx_0
= p(x_{i-1}, x_i).
$$

The middle equality is just the forward-process factorization
$$p(x_0, x_{i-1}, x_i) = p_{\mathrm{data}}(x_0)\,p(x_{i-1}\mid x_0)\,p(x_i\mid x_{i-1})$$
with the Markov property $$p(x_i\mid x_{i-1}, x_0) = p(x_i\mid x_{i-1})$$, integrated
over $$x_0$$. Equivalently, the marginal reverse kernel is the posterior average of
the $$x_0$$-conditioned one, $$p(x_{i-1}\mid x_i) = \mathbb{E}_{p(x_0\mid x_i)}[\,p(x_{i-1}\mid x_i, x_0)\,]$$.

I want to show this is exactly why training against the tractable
$$x_0$$-conditioned posterior is equivalent — up to a $$\phi$$-independent constant —
to training against the intractable marginal reverse. Start from the marginal
objective and split the log into a $$p_\phi$$-dependent part and a
$$p_\phi$$-independent part:

$$
\begin{aligned}
\mathbb{E}_{p(x_i)}\!\left[D_{\mathrm{KL}}\!\left(p(x_{i-1}\mid x_i)\,\|\,p_\phi(x_{i-1}\mid x_i)\right)\right]
&= \int p(x_i)\,p(x_{i-1}\mid x_i)\,\log\frac{p(x_{i-1}\mid x_i)}{p_\phi(x_{i-1}\mid x_i)}\; dx_{i-1}\,dx_i \\
&= \underbrace{-\int p(x_i)\,p(x_{i-1}\mid x_i)\,\log p_\phi(x_{i-1}\mid x_i)\; dx_{i-1}\,dx_i}_{=:\; A(\phi)\;\text{(cross-entropy — the only }\phi\text{-dependent part)}} \\
&\qquad + \underbrace{\int p(x_i)\,p(x_{i-1}\mid x_i)\,\log p(x_{i-1}\mid x_i)\; dx_{i-1}\,dx_i}_{=:\;-H\;\text{(}\phi\text{-independent)}}.
\end{aligned}
$$

Now attack the cross term. Substitute marginal consistency
$$p(x_i)\,p(x_{i-1}\mid x_i) = \int p_{\mathrm{data}}(x_0)\,p(x_i\mid x_0)\,p(x_{i-1}\mid x_i, x_0)\,dx_0$$
directly into $$A(\phi)$$:

$$
\begin{aligned}
A(\phi)
&= -\int \left[\int p_{\mathrm{data}}(x_0)\,p(x_i \mid x_0)\,p(x_{i-1}\mid x_i, x_0)\; dx_0\right] \log p_\phi(x_{i-1}\mid x_i)\; dx_{i-1}\,dx_i \\
&= -\int p_{\mathrm{data}}(x_0)\, p(x_i \mid x_0)\, p(x_{i-1}\mid x_i, x_0)\, \log p_\phi(x_{i-1}\mid x_i)\; dx_0\,dx_{i-1}\,dx_i \\
&= \mathbb{E}_{p_{\mathrm{data}}(x_0)}\,\mathbb{E}_{p(x_i\mid x_0)}\!\left[\,-\int p(x_{i-1}\mid x_i, x_0)\,\log p_\phi(x_{i-1}\mid x_i)\;dx_{i-1}\right].
\end{aligned}
$$

The bracket is the cross-entropy between $$p(x_{i-1}\mid x_i, x_0)$$ and $$p_\phi$$.
Adding and subtracting its own entropy turns it into a KL plus a
$$\phi$$-independent term:

$$
-\int p(x_{i-1}\mid x_i, x_0)\,\log p_\phi(x_{i-1}\mid x_i)\;dx_{i-1}
= D_{\mathrm{KL}}\!\left(p(x_{i-1}\mid x_i, x_0)\,\|\,p_\phi(x_{i-1}\mid x_i)\right) + H\!\left(p(\cdot\mid x_i, x_0)\right).
$$

Plugging this back into $$A(\phi)$$ and then back into the original equation, all
the entropy terms are independent of $$\phi$$ and collapse into a single constant
$$C$$:

$$
\boxed{\;
\mathbb{E}_{p(x_i)}\!\left[D_{\mathrm{KL}}\!\left(p(x_{i-1}\mid x_i)\,\|\,p_\phi(x_{i-1}\mid x_i)\right)\right]
=
\mathbb{E}_{p_{\mathrm{data}}(x_0)}\,\mathbb{E}_{p(x_i\mid x_0)}\!\left[D_{\mathrm{KL}}\!\left(p(x_{i-1}\mid x_i, x_0)\,\|\,p_\phi(x_{i-1}\mid x_i)\right)\right] + C
\;}
$$

So minimizing the tractable $$x_0$$-conditioned objective (the right-hand side,
which is exactly $$\mathcal{L}_{\mathrm{diff}}$$) minimizes the intractable marginal
objective (the left) — they share the same minimizer in $$\phi$$. The constant
$$C = \mathbb{E}[H(p(\cdot\mid x_i, x_0))] - H$$ is the gap between conditioning on
$$x_0$$ and marginalizing it out; it never touches $$p_\phi$$, so it drops out of the
gradient.

> **Where my earlier draft went wrong.** I had tried to push the marginalization
> *through* the log ratio, carrying a factor $$p(x_{i-1})/p(x_{i-1}\mid x_0)$$ and
> cancelling $$p(x_i)$$ term-by-term. That mixes the $$\phi$$-dependent and
> $$\phi$$-independent parts and doesn't cleanly close. The fix is to split off the
> cross-entropy $$A(\phi)$$ *first*, apply marginal consistency only inside $$A(\phi)$$,
> and dump every entropy into $$C$$.
{% endcomment %}

---

# Score-based Modeling

## Motivation behind the score

An energy-based model writes $$p_\phi(x) = \frac{e^{-E_\phi(x)}}{Z_\phi}$$ with
partition function $$Z_\phi = \int e^{-E_\phi(x)}\, dx$$, which is intractable in
high dimensions. The score sidesteps $$Z_\phi$$ entirely since it's constant in $$x$$. 

$$
\nabla_x \log p_\phi(x) = \nabla_x\!\left(-E_\phi(x) - \log Z_\phi\right) = -\nabla_x E_\phi(x).
$$

## Score matching

Instead of a likelihood, fit the *score* $$s(x) = \nabla_x \log p_{\mathrm{data}}(x)$$
directly with a model $$s_\phi$$. The explicit objective is the Fisher divergence

$$
\mathcal{L}_{\mathrm{SM}}(\phi)
= \tfrac12\,\mathbb{E}_{x \sim p_{\mathrm{data}}}\left[\left\| s_\phi(x) - \nabla_x \log p_{\mathrm{data}}(x)\right\|_2^2\right].
$$

This is not directly usable because the target $$\nabla_x \log p_{\mathrm{data}}$$ is
unknown. Expand the square into three terms:

$$
\mathcal{L}_{\mathrm{SM}}(\phi)
= \tfrac12\,\mathbb{E}_{p_{\mathrm{data}}}\!\left[\|s_\phi(x)\|_2^2\right]
\;-\; \mathbb{E}_{p_{\mathrm{data}}}\!\left[\langle s_\phi(x),\, \nabla_x \log p_{\mathrm{data}}(x)\rangle\right]
\;+\; \tfrac12\,\mathbb{E}_{p_{\mathrm{data}}}\!\left[\|\nabla_x \log p_{\mathrm{data}}(x)\|_2^2\right]
$$

The last term is irrelevant to optimization.
We handle the cross-term by applying IbP:

$$
\begin{aligned}
\mathbb{E}_{p_{\mathrm{data}}}\left[\langle s_\phi(x),\, \nabla_x \log p_{\mathrm{data}}(x)\rangle\right] &= \int s_\phi(x)^\top \nabla_x p_{\mathrm{data}}(x)\, dx \\
&= -\int p_{\mathrm{data}}(x)\, \bigl(\nabla_x \cdot s_\phi(x)\bigr)\, dx
+ \underbrace{\oint_{\partial} p_{\mathrm{data}}\,(s_\phi\cdot \nu)\, dS}_{\to\, 0} \\
&= -\,\mathbb{E}_{p_{\mathrm{data}}}\!\left[\operatorname{Tr}\!\left(\nabla_x s_\phi(x)\right)\right].
\end{aligned}
$$

The boundary term vanishes because $$p_{\mathrm{data}}(x)\to 0$$ at infinity faster
than $$s_\phi$$ grows. In the weak formulation of a PDE, the derivative moves off the unknown density and onto
the model, which we can differentiate. Our expanded objective becomes

$$
\tilde{\mathcal{L}}_{\mathrm{SM}}(\phi)
= \tfrac12\,\mathbb{E}_{p_{\mathrm{data}}}\!\left[\|s_\phi(x)\|_2^2\right]
-\,\mathbb{E}_{p_{\mathrm{data}}}\!\left[\operatorname{Tr}\!\left(\nabla_x s_\phi(x)\right)\right]
= \mathbb{E}_{x \sim p_{\mathrm{data}}}\!\left[\operatorname{Tr}\!\left(\nabla_x s_\phi(x)\right) + \tfrac12\left\|s_\phi(x)\right\|_2^2\right] + C.
$$

The two terms have geometric interpretations: the magnitude term pushes the score toward
zero in high-density regions, and the trace term contracts those stationary points *inward*
for the induced vector field.

## Denoising score matching and Tweedie's formula

The trace term is expensive, and the clean score is ill-defined when data lies on a
low-dimensional manifold. Both problems dissolve if we add noise. Let

$$
\tilde{x} = \alpha x + \sigma \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I), \qquad x \sim p_{\mathrm{data}},
$$

so the conditional corruption density is
$$p_\sigma(\tilde{x}\mid x) = \mathcal{N}(\tilde{x};\, \alpha x,\, \sigma^2 I)$$ with
*closed-form* conditional score

$$
\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}\mid x) = \frac{\alpha x - \tilde{x}}{\sigma^2}.
$$

The object we actually want is the score of the noisy *marginal*
$$p_\sigma(\tilde{x}) = \int p_\sigma(\tilde{x}\mid x)\, p_{\mathrm{data}}(x)\, dx$$.
Differentiating under the integral,

$$
\nabla_{\tilde{x}}\log p_\sigma(\tilde{x})
= \frac{\int \nabla_{\tilde{x}} p_\sigma(\tilde{x}\mid x)\,p_{\mathrm{data}}(x)\,dx}{p_\sigma(\tilde{x})}
= \mathbb{E}_{x \mid \tilde{x}}\left[\nabla_{\tilde{x}}\log p_\sigma(\tilde{x}\mid x)\right].
$$

So the marginal score is the Rao–Blackwellized posterior expectation of the
conditional score. Substituting the Gaussian conditional score and rearranging
yields Tweedie's formula:

$$
\nabla_{\tilde{x}}\log p_\sigma(\tilde{x})
= \frac{\alpha\,\mathbb{E}[x \mid \tilde{x}] - \tilde{x}}{\sigma^2}
\quad\Longleftrightarrow\quad
\alpha\,\mathbb{E}[x \mid \tilde{x}] = \tilde{x} + \sigma^2\,\nabla_{\tilde{x}}\log p_\sigma(\tilde{x}).
$$

Because a conditional expectation is the minimizer of the squared error, the
optimal network trained on the *conditional* target equals the *marginal* score:

$$
s^\star(\tilde{x}) = \mathbb{E}\left[\nabla_{\tilde{x}}\log p_\sigma(\tilde{x}\mid x)\;\middle|\;\tilde{x}\right] = \nabla_{\tilde{x}}\log p_\sigma(\tilde{x}).
$$

This is the denoising-score-matching (DSM) equivalence: for any fixed noise level,
the tractable conditional objective and the intractable marginal objective share a
minimizer.

## One object, three parameterizations

Tweedie's formula shows the score predictor, the noise predictor, and the
$$x_0$$-predictor (denoiser) are algebraically interchangeable. Writing the DDPM
optimum for noise prediction $$\epsilon^\star$$ against the score optimum
$$s^\star$$,

$$
s^\star(x_\sigma, \sigma) = -\frac{1}{\sigma}\,\epsilon^\star(x_\sigma, \sigma),
\qquad
D_\phi(\tilde{x}) = \tilde{x} + \sigma^2 s_\phi(\tilde{x}),
\qquad
s_\phi(\tilde{x}) = \frac{D_\phi(\tilde{x}) - \tilde{x}}{\sigma^2}.
$$

---

# Continuous time formulation

## From discrete updates to SDEs

Both the variance-exploding (NCSN) and variance-preserving (DDPM) chains are
Euler–Maruyama discretizations of an SDE. Take a general update

$$
x_{t+\Delta t} = x_t + a(x_t, t)\,\Delta t + b(x_t, t)\sqrt{\Delta t}\,\epsilon_t,
\qquad \epsilon_t \sim \mathcal{N}(0, I).
$$

The conditional mean and covariance to first order in $$\Delta t$$ are
$$\mathbb{E}[\Delta x \mid x_t] = a\,\Delta t$$ and
$$\operatorname{Cov}(\Delta x \mid x_t) = bb^\top \Delta t$$, which identify the
limiting SDE

$$
dx_t = a(x_t, t)\, dt + b(x_t, t)\, dw_t
$$

where $$dw_t \sim \mathcal{N}(0, dt \; I)$$.

For DDPM, expand the drift coefficient with the scalar Taylor series
$$\sqrt{1 - z} = 1 - \tfrac{z}{2} + O(z^2)$$:

$$
x_{t+\Delta t} = \sqrt{1 - \beta(t)\Delta t}\; x_t + \sqrt{\beta(t)\Delta t}\;\epsilon_t
\;\approx\; x_t - \tfrac12\beta(t)\, x_t\,\Delta t + \sqrt{\beta(t)}\sqrt{\Delta t}\;\epsilon_t,
$$

so the drift and diffusion are $$f(x,t) = -\tfrac12\beta(t)x$$ and
$$g(t) = \sqrt{\beta(t)}$$; typically, the drift is non-positive in $x_t$ to corrupt the signal
to the zero-centered mean.
This is a (time-inhomogeneous) Ornstein–Uhlenbeck
(OU) process. NCSN is non-affine in the drift, and only the variance tends to 1,
$$\sqrt{\sigma^2_{t+\Delta t} - \sigma^2_t} \approx \sqrt{\tfrac{d\sigma_t^2}{dt}}\sqrt{\Delta t}$$,
giving zero drift.

Ordinary calculus is not enough, because Brownian increments
scale as $$\Delta w = O(\sqrt{\Delta t})$$, so the $$(\Delta w)^2 = O(\Delta t)$$ term remains
at the same order as the drift. Concretely, in the second-order Taylor expansion of
$$h(x_t, t)$$ the products of differentials obey the Itô multiplication rules

$$
dt\cdot dt = 0, \qquad dt\cdot dw_t = 0, \qquad dw_t\cdot dw_t = dt,
$$

which hold in mean-square: $$dt\cdot dt$$ and $$dt\cdot dw_t$$ are $$O(\Delta t^{3/2})$$
or smaller and vanish, while $$\mathbb{E}[(dw_t)^2] = dt$$ with vanishing variance
survives. Keeping only the surviving second-order term gives the Itô correction:

$$
dh(x_t, t) = \left(\partial_t h + f\,\partial_x h + \tfrac12 g^2\,\partial_{xx} h\right) dt + g\,\partial_x h\, dw_t.
$$

The two canonical forward SDEs:

$$
\begin{aligned}
\text{VP (DDPM):}\quad dx_t &= -\tfrac12\beta(t)\,x_t\,dt + \sqrt{\beta(t)}\,dw_t,\\
\text{VE (NCSN):}\quad dx_t &= \sqrt{\tfrac{d\sigma^2(t)}{dt}}\,dw_t.
\end{aligned}
$$

The VP process has closed-form marginals
$$x_t \mid x_0 \sim \mathcal{N}(\alpha(t)x_0,\,(1-\alpha(t)^2)I)$$ with
$$\alpha(t) = \exp\left(-\tfrac12\int_0^t \beta(s)\,ds\right)$$.
This mean factor comes from the integrating factor of the (deterministic part of the)
linear SDE: for $$\tfrac{d}{dt}\mathbb{E}[x_t] = -\tfrac12\beta(t)\,\mathbb{E}[x_t]$$,
multiplying by $$\exp\!\left(\tfrac12\int_0^t\beta\right)$$ makes the left side an exact
derivative, giving $$\mathbb{E}[x_t] = \alpha(t)x_0$$. The variance then follows from
the Itô isometry, and the exponentially-decaying mean with variance approaching $$I$$
is preserves the variance, i.e. $$\alpha(t)^2 + (1-\alpha(t)^2) = 1$$
at all times when $$x_0$$ is standardized.

A third common choice interpolates between the two. The **sub-VP SDE** shares the VP
drift but scales the diffusion so the marginal variance stays *below* the VP variance
throughout:

$$
dx_t = -\tfrac12\beta(t)\,x_t\,dt + \sqrt{\beta(t)\left(1 - e^{-2\int_0^t \beta(s)\,ds}\right)}\; dw_t.
$$

Its variance is bounded by the VP variance at every $$t$$, which empirically yields
smoother likelihoods and lower sampling error while retaining the same limiting
prior as VP.

## Fokker–Planck and the probability current

The marginal density $$p_t$$ of the forward SDE
$$dX_t = f\,dt + G\,dW_t$$ (with diffusion matrix $$a = GG^\top$$) evolves by the
Fokker–Planck equation, which we write as a continuity equation:

$$
\partial_t p_t(x) = -\nabla\cdot\bigl(f(x,t)p_t(x)\bigr) + \tfrac12\sum_{i,j}\partial_{x_i}\partial_{x_j}\bigl(a_{ij}(x,t)p_t(x)\bigr)
= -\nabla\cdot J_t(x),
$$

with the **probability current**

$$
J_t(x) = f(x,t)p_t(x) - \tfrac12\,\nabla\cdot\bigl(a(x,t)p_t(x)\bigr),
\qquad
\bigl[\nabla\cdot(ap)\bigr]_i = \sum_j \partial_{x_j}(a_{ij}p).
$$

Using the product rule, $$\dfrac{1}{p_t}\nabla\cdot(a p_t) = \nabla\cdot a + a\,\nabla\log p_t$$, so

$$
\frac{J_t}{p_t} = f - \tfrac12\left(\nabla\cdot a + a\,\nabla\log p_t\right).
$$

## Anderson's reverse-time SDE

Run the clock backward with $$s = T - t$$, $$Y_s := X_{T-s}$$, so $$Y_s$$ has density
$$q_s(x) = p_{T-s}(x)$$. Since $$\partial_s q_s = \nabla\cdot J_{T-s}$$ while the
reverse process must satisfy its own continuity equation
$$\partial_s q_s = -\nabla\cdot J_s^{\mathrm{R}}$$, **reversing time reverses the
current**: $$J_s^{\mathrm{R}} = -J_{T-s}$$. Positing a reverse SDE with drift
$$b^{\mathrm{R}}$$ and the same local diffusion $$a$$, matching currents gives

$$
b^{\mathrm{R}}(x, s) = -f(x, T-s) + \nabla\cdot a(x, T-s) + a(x, T-s)\,\nabla_x \log p_{T-s}(x).
$$

In the diffusion-model case $$G(x,t) = g(t)I$$, we have $$a = g^2 I$$ and
$$\nabla\cdot a = 0$$, so — written in the original time label integrated backward
from $$T$$ to $$0$$ — this is the familiar reverse-time SDE:

$$
\boxed{\,d\bar X_t = \left[\,f(\bar X_t, t) - g^2(t)\,\nabla\log p_t(\bar X_t)\right] dt + g(t)\, d\bar W_t\,}
$$

For DDPM ($$f = -\tfrac12\beta x$$, $$g^2 = \beta$$), the reverse dynamics are

$$
d\bar X_t = \left[-\tfrac12\beta(t)\bar X_t - \beta(t)\,\nabla\log p_t(\bar X_t)\right] dt + \sqrt{\beta(t)}\, d\bar W_t.
$$

The only unknown is the time-dependent score $$\nabla\log p_t$$ — which is exactly
what DSM trains the network to approximate.

> **A note on time labels.** Writing the coefficients as $$f(\cdot, T-s)$$ under a
> forward-running reverse clock $$s$$ is not treating $$T-s$$ as increasing time; it
> is the *forward-time label of the slice currently being reversed*. At $$s=0$$ the
> reverse process starts from the noisy terminal $$p_T$$, so it must use the
> schedule value $$\beta(T)$$, i.e. $$\beta_{\mathrm{rev}}(s) = \beta(T-s)$$. Using
> $$\beta(s)$$ would incorrectly start from the data-end schedule value.

## The probability-flow ODE and marginal consistency

The current can also be reproduced by a *deterministic* flow. Since
$$-\tfrac12 g^2 \nabla p_t = -\tfrac12 g^2 p_t \nabla\log p_t$$, the Fokker–Planck
equation is equivalently a continuity equation for the velocity field
$$v(x,t) = f(x,t) - \tfrac12 g^2(t)\nabla\log p_t(x)$$:

$$
\partial_t p_t = -\nabla\cdot(v\,p_t),
\qquad
\frac{dx_t}{dt} = f(x_t, t) - \tfrac12 g^2(t)\,\nabla\log p_t(x_t).
$$

This **probability-flow ODE** has the *same marginals* $$p_t$$ as the SDE. It is the basis
for deterministic samplers and exact likelihood computation.

{% comment %}
## Girsanov's theorem and the continuous ELBO
COMMENTED OUT — to be revisited after studying Girsanov. Draft preserved below.

Girsanov's theorem is what lets us bound a likelihood in path space. Consider two
diffusions on [0,T] with the same diffusion coefficient but different drifts,
dX_t = f dt + g dW_t and dX_t = h dt + g dW_t, inducing path measures P and Q.
With u = g^{-1}(f - h), the Radon–Nikodym derivative is
dP/dQ = exp( ∫ u dW - 1/2 ∫ ||u||^2 dt ), and E_P[log(·)] gives the path-space KL
KL(P||Q) = 1/2 E_P[ ∫ ||f-h||^2 / g^2 dt ]. Taking P = true reverse process
(exact score) and Q = learned process (s_phi), the drifts differ by
f - h = g^2 (s_phi - ∇log p_t), so KL(P||Q) = 1/2 E[ ∫ g^2 ||s_phi - ∇log p_t||^2 dt ],
a time-weighted DSM objective bounding KL(p_data || p_phi).
{% endcomment %}

---

# Conceptual Summary

A diffusion model is a HVAE whose encoder is a fixed Gaussian noising
process. Its ELBO decomposes into per-step denoising-matching KLs; each is
minimized by learning the score of the noisy marginal, which DSM makes tractable
via Tweedie's formula. In continuous time the forward chain is an SDE whose
marginals are governed by Fokker–Planck (the $$\mathcal{L}^2$$ adjoint of the infintesimal generator);
Anderson's theorem reverses it using the score, and
the PF-ODE reproduces the same marginals and admits time-reversable integration.
Score matching, DDPMs, discrete diffusion, and SDEs are all views of the same underlying object:
conditioning on initial states and corrupting the data yields tractable objectives.
Studying the underlying density evolution allows us to formalize the theory without
singularities; e.g., computing deterministic sampling trajectories.

*These notes are my own derivations, written for my reference and anyone learning
the material; corrections are welcome.*
