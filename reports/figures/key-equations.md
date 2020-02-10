# Equations

This is a list of equations in LaTeX that can be used for presenations.

## VAE

Bayes theorem:

$$
p(z|x)  = \frac{p(x|z)p(z)}{p(x)}
$$

Variational inference, approximate:

$$
p(z|x) = q_\lambda(z|x)
$$

### Kullback-Leibler Divergence

The KulLback-Leibler Divergence between two distributions of a continuous variable, $q(x)$ and $p(x)$, is defined by:

$$
KL\left[ p(z) || q(x) \right] = \int_{-\inf}^{\inf} p(x) \log\left(\frac{p(x)}{q(x)} \right)
$$

This relationship is fundamentally asymmetric.