---
title: "Laplace's Equation and the Fundamental Solution of the Poisson Equation"
excerpt: "The radial harmonic solution, the fundamental solution of Laplace's equation, and a proof that $\\Phi * f$ solves the Poisson equation."
date: 2026-05-01
last_modified_at: 2026-05-01
---

These notes work through the fundamental solution of Laplace's equation and prove
that convolution with it solves the Poisson equation, following the standard Evans reference.
This was my contribution to the final project for the second undergraduate course in honors real analysis.

## Laplace's Equation

Let $\Omega \subset \R^n$ be an open set. Let $u: \Omega \to \R$, which is
*harmonic* if $u \in C^2(\Omega)$ and satisfies, for all $x \in \Omega$,

$$
\laplace u = \sum_{i=1}^{n} u_{x_i x_i} = 0,
\tag{Laplace}
$$

commonly known as Laplace's equation.

### Radial Solution

Since Laplace's equation is invariant under rotations, we consider a radial
solution to determine its fundamental solution. Let $u(x) = v(r)$ with
$r = \norm{x}_2$. For each $i = 1, \ldots, n$,

$$
\frac{\partial r}{\partial x_i} = x_i \left(\sum_{i=1}^{n} x_i^2 \right)^{-1/2} = \frac{x_i}{r}.
$$

Differentiating,

$$
u_{x_i} = v'(r) \cdot \frac{x_i}{r},
$$

and

$$
\begin{aligned}
u_{x_i x_i} &= v''(r) \cdot \pr{\frac{x_i}{r}}^2 + v'(r) \cdot \frac{r - x_i \cdot \frac{x_i}{r}}{r^2}\\
&= v''(r) \cdot \pr{\frac{x_i}{r}}^2 + v'(r) \cdot \pr{\frac{1}{r} - \frac{x_i^2}{r^3}}.
\end{aligned}
$$

Hence,

$$
\laplace u = v''(r) + \frac{n-1}{r}v'(r) = 0.
$$

Rearranging, observe that

$$
\bigl(\log v'\bigr)' = \frac{v''}{v'} = \frac{1-n}{r}.
$$

Integrating,

$$
\begin{aligned}
\int \frac{v''}{v'}\, \dd r &= \int \frac{1-n}{r}\, \dd r\\
\Rightarrow \log(v') &= (1-n) \log r + C\\
\Rightarrow v' &= r^{1-n} \cdot e^{C} = \frac{b}{r^{n-1}}
\end{aligned}
$$

with $b, C \in \R$. Integrating once more,

$$
v(r) = \int \frac{b}{r^{n-1}}\, \dd r.
$$

With $c \in \R$,

$$
v(r) =
\begin{cases}
b \log r + c, & n = 2\\
\dfrac{b}{r^{n-2}} + c, & n \geq 3.
\end{cases}
$$

### Fundamental Solution

Set the constants $b, c$ such that:

> **Definition (fundamental solution).** For $x \in \R^n$ and $x \neq 0$,
>
> $$
> \Phi(x) :=
> \begin{cases}
> -\dfrac{1}{2\pi} \log \norm{x}_2, & n = 2\\[2mm]
> \dfrac{1}{n(n-2)\alpha(n)} \cdot \dfrac{1}{\norm{x}_2^{n-2}}, & n \geq 3
> \end{cases}
> $$
>
> is the fundamental solution to Laplace's equation. Note that $\Phi$ is singular
> (and undefined) at $x = 0$. Here $\alpha(n)$ is the volume of the unit ball in
> $\R^n$.

## The Poisson Equation

In this section, we focus on the Dirichlet boundary value problem

$$
\begin{cases}
-\laplace u = f & \text{on } \Omega \\
\phantom{-\laplace} u = 0 & \text{on } \partial \Omega
\end{cases}
$$

where $\Omega \subset \R^d$ is an open bounded subset and $f : \Omega \to \R$ is
$C^2$.

Recall the harmonic mapping $\Phi: \R^n \to \R$ from the definition above. A
potential solution to the Poisson equation is the convolution,

$$
u(x) = (\Phi * f)(x) = \int_{\Omega} \Phi(x - y) f(y) \dd y,
$$

which is harmonic; $\Phi(x-y)f(y)$ is harmonic for every $x \neq y$. However,

$$
\laplace u = \laplace_x (\Phi * f)(x) = \int_{\Omega} \laplace_x \Phi(x - y) f(y)\, \dd y.
$$

$\Phi(x-y)f(y)$ is no longer harmonic when $x = y$, as $\Phi(x)$ blows up as
$x \to 0$. If $n = 2$, $-\frac{1}{2\pi}\log\abs{x} \to \infty$ as $x \to 0$; if
$n \geq 3$, $\frac{1}{\abs{x}^{n-2}} \to \infty$ as $x \to 0$. Thus $D^2 \Phi(x-y)$
would also be singular, so we consider this more carefully.

> **Theorem.** The classical solution to the Poisson equation is $\Phi * f$.

**Proof.** For simplicity, assume $f$ has compact support (otherwise we cannot
assume a bound on $\laplace_x f$), and $0 \in \Omega$.

Since the convolution operator is commutative, consider

$$
\begin{aligned}
Du(x) &= \lim_{h \to 0} \frac{u(x + e_i h) - u(x)}{h}\\
&= \int_{\Omega} \lim_{h \to 0} \pr{\frac{f(x-y + e_i h) - f(x - y)}{h}} \cdot \Phi(y)\, \dd y\\
&= \int_{\Omega} Df(x-y) \cdot \Phi(y)\, \dd y,
\end{aligned}
$$

with $e_i$ the standard unit basis vector at the $i$-th coordinate for all
$i = 1, \ldots, n$. By the same reasoning,

$$
D^2u(x) = \int_{\Omega} D^2f(x-y) \cdot \Phi(y)\, \dd y.
$$

Consider, for $\epsilon > 0$,

$$
\laplace u(x) = \underbrace{\int_{B(0, \epsilon)} \laplace_x f(x-y) \cdot \Phi(y)\, \dd y}_{I_\epsilon\; :=} +
\underbrace{\int_{\Omega \setminus B(0, \epsilon)} \laplace_x f(x-y) \cdot \Phi(y)\, \dd y}_{J_\epsilon\; :=}.
$$

$f$ has compact support, so it is bounded everywhere by the extreme value theorem.
Since $f \in C^2$, there exists $C > 0$ such that $\abs{\laplace_x f} \leq C$. Then

$$
\begin{aligned}
I_\epsilon &= \int_{B(0, \epsilon)} \laplace_x f(x-y) \cdot \Phi(y)\, \dd y\\
&\leq C\int_{B(0, \epsilon)} \abs{\Phi(y)} \, \dd y.
\end{aligned}
$$

For $n = 2$, converting to polar coordinates,

$$
\int_{B(0,\epsilon)} \abs{\Phi(y)} \, \dd y
= \frac{1}{2\pi} \int_{B(0,\epsilon)} \abs{\log \norm{y}_2} \, \dd y
= \frac{1}{2\pi} \int_{0}^{\epsilon} \abs{\log r}\,r \, \dd r
\leq \frac{\epsilon^2}{2}.
$$

For $n = 3$,

$$
\int_{B(0,\epsilon)} \abs{\Phi(y)} \, \dd y
= \frac{1}{n(n-2)\alpha(n)} \int_{B(0,\epsilon)} \frac{1}{\norm{y}_2^{n-2}} \, \dd y
= \frac{1}{n(n-2)\alpha(n)} \int_{0}^{\epsilon} \frac{r^{n-1}}{r^{n-2}} \, \dd r
\leq \frac{\epsilon^2}{2n(n-2)\alpha(n)}.
$$

Thus, taking $\epsilon \to 0$, $I_\epsilon \to 0$ vanishes.

Apply integration by parts on $\R^n$ to $J_\epsilon$; i.e.,
$\int_\Omega u \,\mathrm{div}(V)\, \dd\Omega = -\int_{\Omega} Du \cdot V \dd \Omega + \int_{\partial \Omega} uV \cdot \nu \, \dd S$
where $\nu$ is the outward unit normal along the boundary $\partial \Omega$. Take
$u = \Phi(y)$ and $\mathrm{div}(V) = \laplace_x f(x-y)$:

$$
\begin{aligned}
J_\epsilon &= \int_{\Omega \setminus B(0,\epsilon)} \laplace_x f(x-y) \Phi(y) \, \dd y\\
&= \underbrace{- \int_{\Omega \setminus B(0,\epsilon)} D\Phi(y) \cdot D_y f(x-y) \, \dd y}_{K_\epsilon\; :=}\\
&\quad + \underbrace{\int_{\partial B(0,\epsilon)} \Phi(y) \pr{Df \cdot \nu} \, \dd S(y)}_{L_\epsilon\; :=},
\end{aligned}
$$

where $S(y)$ denotes integration over the surface of the sphere
$\partial B(0,\epsilon)$, and $\nu$ points inwards away from the region
$\Omega \setminus B(0,\epsilon)$ and towards the origin.

The boundary term vanishes, as

$$
L_\epsilon = \int_{\partial B(0,\epsilon)} \Phi(y) \pr{Df \cdot \nu} \, \dd S(y)
\leq \sup_{z \in \partial B(0,\epsilon)} \abs{Df(z)} \int_{\partial B(0,\epsilon)} \abs{\Phi(y)} \, \dd S(y).
$$

By the same argument as before with EVT, $L_\epsilon \to 0$
as $\epsilon \to 0$.

Apply integration by parts again on $K_\epsilon$:

$$
\begin{aligned}
K_\epsilon &= - \int_{\Omega \setminus B(0,\epsilon)} D\Phi(y) \cdot D_y f(x-y) \, \dd y\\
&= \int_{\Omega \setminus B(0,\epsilon)} \laplace \Phi(y)\,f(x-y) \, \dd y\\
&\quad - \int_{\partial B(0,\epsilon)} \left(D\Phi(y) \cdot \nu \right)f(x-y) \, \dd S(y).
\end{aligned}
$$

$\laplace \Phi(z) = 0$ when $z$ is away from the origin, since $\Phi$ satisfies
Laplace's equation. Thus we are left with only the second term. Evans states that
$D\Phi(y) = -\frac{1}{n\alpha(n)} \frac{y}{\norm{y}_2^n}$, and $\nu = -\frac{y}{\epsilon}$
on $\partial B(0,\epsilon)$, so $D\Phi(y) \cdot \nu = (n\alpha(n)\epsilon^{n-1})^{-1}$.
Thus,

$$
\begin{aligned}
K_\epsilon &= - \int_{\partial B(0,\epsilon)} \left(D\Phi(y) \cdot \nu \right)f(x-y) \, \dd S(y) \\
&= - \frac{1}{n\alpha(n) \epsilon^{n-1}} \int_{\partial B(0,\epsilon)} f(x-y) \, \dd S(y) \\
&= - \frac{1}{n\alpha(n) \epsilon^{n-1}} \int_{\partial B(x,\epsilon)} f(y) \, \dd S(y) \to -f(x), \qquad \epsilon \to 0.
\end{aligned}
$$

Thus,

$$
\laplace u(x) = I_\epsilon + J_\epsilon = I_\epsilon + K_\epsilon + L_\epsilon \to -f(x), \qquad \epsilon \to 0,
$$

so

$$
-\laplace u(x) = f,
$$

which is the Poisson equation. $\blacksquare$
