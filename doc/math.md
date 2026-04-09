# Implemented Math

This document describes the math currently implemented in this repository and maps it to the notation used in the TD-Flow paper.

## Paper Notation

The paper uses the following high-level objects:

- $S_t, A_t$: current state and action
- $S_{t+1}, A_{t+1}$: next state and action
- $X_0 \sim m_0$: source sample
- $X_1$: endpoint of the conditional path
- $X_t$: point sampled along a conditional probability path between $X_0$ and $X_1$
- $u_t(X_t \mid X_1)$: target conditional velocity field
- $\psi_t$: neural ODE flow induced by the learned vector field
- $z$: the extra Forward-Backward / task embedding used in the multi-policy setting

In this document, $z$ is reserved for that paper-side task embedding. It is not used for the observation representation.

## Code-to-Paper Mapping

The current code uses:

- current observation $\mathrm{obs}$
- current action $a$
- next observation $\mathrm{obs}'$
- next action $a'$
- optional multi-policy input `policy_embedding`

and learns:

- observation encoder $e_\theta(\mathrm{obs}) \in \mathbb{R}^d$
- encoded state $s = e_\theta(\mathrm{obs})$
- context encoder $c_\theta(s, a)$ in single-policy mode
- context encoder $c_\theta(s, a, z)$ in multi-policy mode
- vector field $v_\theta(x, t, c) \in \mathbb{R}^d$

The mapping to the paper is:

- $S_t \leftrightarrow \mathrm{obs}$
- $A_t \leftrightarrow a$
- $S_{t+1} \leftrightarrow \mathrm{obs}'$
- $A_{t+1} \leftrightarrow a'$
- paper multi-policy embedding $z \leftrightarrow$ code field `policy_embedding`
- current encoded state:
  $$
  s = e_\theta(\mathrm{obs})
  $$
- next encoded state:
  $$
  s' = \bar e(\mathrm{obs}')
  $$

So this repo implements TD-Flow through an encoded state $s = e_\theta(\mathrm{obs})$, while reserving $z$ for the paper's multi-policy task embedding.

From Table 5 in the paper, the confirmed input distinction is:

- single-policy: `s, a`
- multi-policy: `s, a, z`

The current implementation uses:

- single-policy: $(s, a)$
- multi-policy: $(s, a, z)$

When `observation_encoder=identity` and the observation is a flat vector, this reduces to:

$$
s = \mathrm{obs}, \qquad s' = \mathrm{obs}'.
$$

That is why the single-policy flat-state case is the closest code path to the paper's `s, a` view.

The current code also exposes two network families:

- `network_variant=repo`: the repo-default FiLM residual U-Net approximation
- `network_variant=paper`: the paper-width U-Net MLP path using Table 5 widths
  - single-policy width: $512$
  - multi-policy width: $1024$
  - time embedding width: $256$

## Target Networks

The implementation keeps exponential-moving-average target copies:

$$
\bar{\theta} \leftarrow \rho \bar{\theta} + (1 - \rho)\theta
$$

where $\rho$ is the Polyak coefficient.

The training code samples:

$$
t \sim \mathrm{Uniform}(\varepsilon, 1 - \varepsilon), \qquad X_0 := x_0 \sim \mathcal{N}(0, I).
$$

## Direct Flow-Matching Branch

In the direct branch, the code uses a linear conditional probability path from $X_0$ to the target encoded state $s'$:

$$
X_t^{\mathrm{dir}} = t s' + (1 - t)X_0.
$$

The corresponding conditional velocity target is:

$$
u_t^{\mathrm{dir}}(X_t^{\mathrm{dir}} \mid s') = \frac{s' - X_t^{\mathrm{dir}}}{1 - t}.
$$

Since the path is linear, this simplifies to:

$$
u_t^{\mathrm{dir}} = s' - X_0.
$$

The online model predicts:

$$
\hat u_t^{\mathrm{dir}} = v_\theta\!\left(X_t^{\mathrm{dir}}, t, c_\theta(s, a)\right)
$$

in single-policy mode, and

$$
\hat u_t^{\mathrm{dir}} = v_\theta\!\left(X_t^{\mathrm{dir}}, t, c_\theta(s, a, z)\right)
$$

in multi-policy mode.

The implemented direct loss is:

$$
\mathcal{L}_{\mathrm{direct}} = \mathbb{E}\left[\left\|\hat u_t^{\mathrm{dir}} - u_t^{\mathrm{dir}}\right\|_2^2\right].
$$

## Bootstrap TD$^2$ Branch

The bootstrap branch uses the target networks to define a next-step target flow conditioned on $(s', a')$ in single-policy mode, or $(s', a', z)$ in multi-policy mode:

$$
\frac{d x_\tau}{d\tau} = \bar v\!\left(x_\tau, \tau, \bar c(s', a')\right), \qquad x_0 = X_0 \sim \mathcal{N}(0, I)
$$

or

$$
\frac{d x_\tau}{d\tau} = \bar v\!\left(x_\tau, \tau, \bar c(s', a', z)\right), \qquad x_0 = X_0 \sim \mathcal{N}(0, I).
$$

This ODE is integrated from $0$ to $t$ with midpoint integration and a fixed number of steps, producing:

$$
X_t^{\mathrm{boot}}.
$$

The target velocity is then evaluated with the target vector field:

$$
u_t^{\mathrm{boot}} = \bar v\!\left(X_t^{\mathrm{boot}}, t, \bar c(s', a')\right)
$$

or

$$
u_t^{\mathrm{boot}} = \bar v\!\left(X_t^{\mathrm{boot}}, t, \bar c(s', a', z)\right).
$$

The online model predicts a velocity at that same point, but conditioned on the current transition $(s, a)$ or $(s, a, z)$:

$$
\hat u_t^{\mathrm{boot}} = v_\theta\!\left(X_t^{\mathrm{boot}}, t, c_\theta(s, a)\right)
$$

or

$$
\hat u_t^{\mathrm{boot}} = v_\theta\!\left(X_t^{\mathrm{boot}}, t, c_\theta(s, a, z)\right).
$$

The bootstrap loss is:

$$
\mathcal{L}_{\mathrm{boot}} = \mathbb{E}\left[\left\|\hat u_t^{\mathrm{boot}} - u_t^{\mathrm{boot}}\right\|_2^2\right].
$$

## Total Implemented Objective

The final loss is:

$$
\mathcal{L}(\theta) = (1 - \gamma)\mathcal{L}_{\mathrm{direct}} + \gamma \mathcal{L}_{\mathrm{boot}}.
$$

Here $\gamma$ is the TD discount factor.

## Rollout / Learned Flow

For prediction or planning, the learned online flow is:

$$
\frac{d x_t}{dt} = v_\theta(x_t, t, c_\theta(s, a))
$$

or

$$
\frac{d x_t}{dt} = v_\theta(x_t, t, c_\theta(s, a, z)).
$$

Starting from a source representation $x_0$ and integrating to $t_{\mathrm{end}}$, the model outputs a predicted next encoded state:

$$
\hat s_{t+1} = \psi_{t_{\mathrm{end}}}(x_0 \mid s, a)
$$

or

$$
\hat s_{t+1} = \psi_{t_{\mathrm{end}}}(x_0 \mid s, a, z).
$$

In the current code, the default rollout source is sampled from $\mathcal{N}(0, I)$, which matches the training source distribution.

## Important Scope Note

This document is faithful to the current codebase, not a claim that every detail exactly matches the full paper implementation. The strongest paper-supported distinction is between single-policy inputs $s,a$ and multi-policy inputs $s,a,z$. In this document, $z$ is reserved for that paper-side task embedding, while $s = e_\theta(\mathrm{obs})$ denotes the repo's encoded state. The repo's optional identity observation path for flat states is an engineering approximation, not a proven verbatim paper detail. The `network_variant=paper` option aligns widths and time embedding more closely with Table 5, but should still be read as a paper-aligned implementation mode rather than a verified copy of the authors' private code.
