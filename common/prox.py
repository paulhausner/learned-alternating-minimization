"""Proximal operators and first-order steps shared by both problems."""
import math

import torch


def soft_threshold(u, lam):
    """Proximal operator of lam * ||.||_1 (lam may broadcast)."""
    return torch.sign(u) * torch.relu(torch.abs(u) - lam)


def momentum_coefficient(t):
    """Nesterov/FISTA sequence: returns (t_next, beta) with beta = (t - 1) / t_next."""
    t_next = (1 + math.sqrt(1 + 4 * t ** 2)) / 2
    return t_next, (t - 1) / t_next


def nesterov_step(x, x_extra, grad, L, t):
    """One accelerated gradient step on a smooth function (NAG)."""
    x_prev = x
    x = x_extra - grad / L
    t, beta = momentum_coefficient(t)
    return x, x + beta * (x - x_prev), t


def fista_step(z, z_extra, grad, L, lam, t):
    """One accelerated proximal gradient step for f(z) + lam * ||z||_1 (FISTA)."""
    z_prev = z
    z = torch.nn.functional.softshrink(z_extra - grad / L, lam / L)
    t, beta = momentum_coefficient(t)
    return z, z + beta * (z - z_prev), t


def inv_softplus(y):
    """Inverse of softplus, used to initialize positive learned thresholds."""
    return float(math.log(math.expm1(y)))
