# Equations

This is a list of equations in LaTeX that can be used for presenations.

## VAE

Problem statement

Bayes theorem:

$$
p(z|x)  = \frac{p(x|z)p(z)}{p(x)}
$$

Variational inference, approximate:

$$
p(z|x) = q_\lambda(z|x)
$$

Dataset:

$$
\mathcal{D} = \{x^{(1)}, x^{(2)}, ... , x^{(N)}\}
$$

For iid data, log probabilities add:

$$
\log p_\theta(\mathcal{D}) = \sum_{x\in\mathcal{D}} \log p_\theta(x)
$$

Derivation of elbo

$$
\quad =\underbrace{ \mathbb{E}_{z\sim q_\phi(z|x)}\left[ \log\left(\frac{p_\theta(x,z)}{q_\phi(z|x)} \right)\right]}_{\mathcal{L}_{\theta, \phi}(x)} + \underbrace{\mathbb{E}_{z\sim q_\phi(z|x)}\left[\log \left(  \frac{q_\phi(z|x)}{p_\theta(z|x)} \right)\right] }_{D_{KL}(q_\phi(z|x)|p_\theta(x|z)|)}
$$

Stochastic gradient descent:

$$
\frac{1}{N_\mathcal{D}} \nabla_\theta \log p_\theta(\mathcal{D})
$$

Marginal Likelihood of data in the presence of some latent variables $z$.
3231
$$
p_\theta(x) = \int p_\theta(x, z) dz
$$

Deep latent variable models when $p_\theta(x, z)$ is modelled by neural networks.
Often this is broken down as:

$$
p_\theta(x, z) = p_\theta(z) p_\theta(x|z)
$$

where $p_\theta(z)$ is the *prior*, and can be written simply for certain assumptions since it is not conditioned on any observations.

### Kullback-Leibler Divergence

The KulLback-Leibler Divergence between two distributions of a continuous variable, $q(x)$ and $p(x)$, is defined by:

$$
KL\left[ p(z) || q(x) \right] = \int_{-\inf}^{\inf} p(x) \log\left(\frac{p(x)}{q(x)} \right)
$$

This relationship is fundamentally asymmetric.