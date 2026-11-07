"""Robust matrix completion (low-rank + sparse decomposition).

Problem:

    min_{x,z}  tau * ||x||_* + 0.5 * ||P_Omega(y - x - z)||_F^2 + lam2 * ||z||_1

P_Omega is elementwise multiplication by a 0/1 mask, so the smooth part of
the x-subproblem has Lipschitz constant 1. The z-subproblem is solved exactly
by soft-thresholding. The x-subproblem has no closed form and is solved
inexactly: by FISTA (baseline) or by one learned proximal gradient step per
outer iteration (`LearnedSVTUpdates`), trained greedily layer by layer.
"""
import math
import time

import torch
import torch.nn as nn

from common.prox import soft_threshold, momentum_coefficient, inv_softplus


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------

def generate_batch(batch_size, n, m, rank, obs_ratio, corruption_frac,
                   corruption_scale=3.0, noise_std=0.0, device="cpu"):
    """Sample a batch of synthetic low-rank + sparse instances.

    Corrupted entries are uniform in [-c*s, c*s] where s is the entrywise
    standard deviation of the low-rank part of the batch, so the
    corruption-to-signal ratio does not depend on n, m or the rank.
    """
    U = torch.randn(batch_size, n, rank, device=device) / math.sqrt(n)
    V = torch.randn(batch_size, rank, m, device=device) / math.sqrt(m)
    x_true = torch.bmm(U, V)

    corruption_magnitude = corruption_scale * x_true.std()
    corrupt_mask = torch.rand(batch_size, n, m, device=device) < corruption_frac
    z_true = torch.zeros(batch_size, n, m, device=device)
    z_true[corrupt_mask] = corruption_magnitude * (
        2 * torch.rand(int(corrupt_mask.sum().item()), device=device) - 1)

    mask = (torch.rand(batch_size, n, m, device=device) < obs_ratio).float()

    y = x_true + z_true
    if noise_std > 0:
        y = y + noise_std * torch.randn(batch_size, n, m, device=device)
    y = mask * y

    return {"x_true": x_true, "z_true": z_true, "mask": mask, "y": y}


def fixed_test_batch(n_test, n, m, rank, obs_ratio, corruption_frac, device, test_seed):
    """Test batch drawn under its own seed without disturbing the global RNG state."""
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(test_seed)

    batch = generate_batch(n_test, n, m, rank, obs_ratio, corruption_frac, device=device)

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return batch


# ----------------------------------------------------------------------
# Objective and subproblems
# ----------------------------------------------------------------------

def objective_value(x, z, y, mask, tau, lam2):
    """Per-instance objective value."""
    resid = mask * (y - x - z)
    nuc = torch.linalg.matrix_norm(x, ord="nuc")
    return (tau * nuc + 0.5 * (resid ** 2).sum(dim=(-2, -1))
            + lam2 * z.abs().sum(dim=(-2, -1)))


def objective_fn(y, mask, tau, lam2):
    """Batch-mean objective as a function of (x, z) for fixed data."""
    def f(x, z):
        return objective_value(x, z, y, mask, tau, lam2).mean().item()
    return f


def rel_err_fn(x_true):
    """Mean relative error ||x - x_true||_F / ||x_true||_F."""
    def f(x_hat):
        return (torch.linalg.matrix_norm(x_hat - x_true, dim=(-2, -1))
                / torch.linalg.matrix_norm(x_true, dim=(-2, -1))).mean().item()
    return f


def solve_z(y, x, mask, lam2):
    """Exact z-subproblem: soft-threshold the observed residual (zero off the mask)."""
    return mask * soft_threshold(y - x, lam2)


def svt(u, tau):
    """Singular value thresholding, the prox of tau * ||.||_*.

    Returns (x, shrunk singular values); their sum is ||x||_*.
    `tau` may be a scalar or a vector indexed by singular value.
    """
    U, S, Vh = torch.linalg.svd(u, full_matrices=False)
    S_shrunk = torch.relu(S - tau)
    return U @ (S_shrunk.unsqueeze(-1) * Vh), S_shrunk


def grad_x(x, s, mask):
    """Gradient of 0.5 * ||P_Omega(s - x)||_F^2 with s = y - z fixed."""
    return mask * (x - s)


def solve_x_classical(x0, s, mask, tau, n_iter):
    """FISTA on the x-subproblem with step size 1/L = 1."""
    x, x_extra, t = x0.clone(), x0.clone(), 1.0
    for _ in range(n_iter):
        x_new, _ = svt(x_extra - grad_x(x_extra, s, mask), tau)
        t, beta = momentum_coefficient(t)
        x_extra = x_new + beta * (x_new - x)
        x = x_new
    return x


def x_subproblem_loss(x, singular_values, s, mask, tau):
    """Unsupervised per-instance loss: the x-dependent part of the objective."""
    resid = mask * (s - x)
    return tau * singular_values.sum(dim=-1) + 0.5 * (resid ** 2).sum(dim=(-2, -1))


# ----------------------------------------------------------------------
# Learned x-update
# ----------------------------------------------------------------------

class LearnedSVTUpdates(nn.Module):
    """One proximal gradient step per outer iteration with a learned scalar
    step size alpha_k and a learned threshold lambda_k per singular-value index.

    At initialization alpha_k = 1 and lambda_k = tau, i.e. one classical
    proximal gradient step. Both are parametrized through softplus.

    `clamp_x` optionally projects the output onto the box {|x_ij| <= clamp_x}
    (part of the learned map; inactive on the trained models in the paper).
    """

    def __init__(self, niter, tau, n, m, alpha_init=1.0, clamp_x=None):
        super().__init__()
        self.niter = niter
        self.tau = tau
        self.clamp_x = clamp_x
        min_dim = min(n, m)

        self._rho_alpha = nn.ParameterList(
            nn.Parameter(torch.tensor(inv_softplus(alpha_init))) for _ in range(niter))
        self._rho_lambda = nn.ParameterList(
            nn.Parameter(torch.full((min_dim,), inv_softplus(tau * alpha_init)))
            for _ in range(niter))

    def forward(self, x, s, mask, layer):
        layer = min(layer, self.niter - 1)
        alpha = torch.nn.functional.softplus(self._rho_alpha[layer])
        lam = torch.nn.functional.softplus(self._rho_lambda[layer])

        x_new, singular_values = svt(x - alpha * grad_x(x, s, mask), lam)

        if self.clamp_x is not None and x_new.detach().abs().max() > self.clamp_x:
            # clamping changes the singular values; recompute so that the
            # loss still evaluates the objective at the returned point
            x_new = x_new.clamp(-self.clamp_x, self.clamp_x)
            singular_values = torch.linalg.svdvals(x_new)

        return x_new, singular_values

    def parameters_for_layer(self, layer):
        return [self._rho_alpha[layer], self._rho_lambda[layer]]

    def init_layer_from_previous(self, layer):
        if layer == 0:
            return
        with torch.no_grad():
            self._rho_alpha[layer].copy_(self._rho_alpha[layer - 1])
            self._rho_lambda[layer].copy_(self._rho_lambda[layer - 1])


def train_layerwise(model_niter, n, m, rank, obs_ratio, corruption_frac,
                    tau, lam2, steps_per_layer=400, batch_size=16, lr=1e-2,
                    device="cpu", log_every=50, log_fn=print, clamp_x=None):
    """Greedy training: layer k is fit to the x-subproblem objective at the
    iterates produced by the already-trained layers 0..k-1, initialized from
    layer k-1. Fresh instances are sampled at every step."""
    
    net = LearnedSVTUpdates(model_niter, tau, n, m, clamp_x=clamp_x).to(device)

    for layer in range(model_niter):
        net.init_layer_from_previous(layer)
        opt = torch.optim.Adam(net.parameters_for_layer(layer), lr=lr)

        for step in range(steps_per_layer):
            batch = generate_batch(batch_size, n, m, rank, obs_ratio,
                                   corruption_frac, device=device)
            y, mask = batch["y"], batch["mask"]

            with torch.no_grad():
                x = torch.zeros_like(y)
                z = torch.zeros_like(y)
                for prev in range(layer):
                    x, _ = net(x, y - z, mask, prev)
                    z = solve_z(y, x, mask, lam2)
                s = y - z

            x_new, sv = net(x, s, mask, layer)
            loss = x_subproblem_loss(x_new, sv, s, mask, tau).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % log_every == 0:
                log_fn(f"[layer {layer + 1}/{model_niter}] step {step:4d}  loss={loss.item():.4f}")

    return net


# ----------------------------------------------------------------------
# Rollouts: {rel_err, objective, wall_time} after each outer iteration
# ----------------------------------------------------------------------

@torch.no_grad()
def learned_rollout(net, y, mask, lam2, max_k, rel_err_of, objective_of):
    x = torch.zeros_like(y)
    z = torch.zeros_like(y)
    out, t_cum = [], 0.0
    for k in range(max_k):
        t0 = time.time()
        x, _ = net(x, y - z, mask, k)
        z = solve_z(y, x, mask, lam2)
        t_cum += time.time() - t0
        out.append({"rel_err": rel_err_of(x), "objective": objective_of(x, z),
                    "wall_time": t_cum})
    return out


def admm_step(x, z, w, u, y, mask, tau, lam2, rho):
    """One ADMM iteration for the splitting x = w with the nuclear norm on w
    (one SVD per iteration, the same cost as one learned iteration)."""
    x = (mask * (y - z) + rho * (w - u)) / (mask + rho)
    z = solve_z(y, x, mask, lam2)
    w, _ = svt(x + u, tau / rho)
    u = u + (x - w)
    return x, z, w, u


@torch.no_grad()
def admm_rollout(y, mask, tau, lam2, rho, max_k, rel_err_of, objective_of):
    """ADMM baseline. The objective is reported at whichever of the two primal
    iterates (x from the quadratic step, or the low-rank w from the SVT step)
    has the lower value; both are kept in the output."""
    x, z, w, u = (torch.zeros_like(y) for _ in range(4))
    out, t_cum = [], 0.0
    for _ in range(max_k):
        t0 = time.time()
        x, z, w, u = admm_step(x, z, w, u, y, mask, tau, lam2, rho)
        t_cum += time.time() - t0

        z_w = solve_z(y, w, mask, lam2)
        obj_x, obj_w = objective_of(x, z), objective_of(w, z_w)
        at_w = obj_w <= obj_x
        out.append({
            "rel_err": rel_err_of(w) if at_w else rel_err_of(x),
            "objective": obj_w if at_w else obj_x,
            "wall_time": t_cum,
            "objective_x": obj_x, "objective_w": obj_w,
            "scored_at": "w" if at_w else "x",
        })
    return out


@torch.no_grad()
def prox_grad_rollout(y, mask, tau, lam2, alpha, max_k, rel_err_of, objective_of):
    """Alternating proximal gradient with a fixed step size alpha and threshold
    alpha * tau: the learned update with hand-set instead of trained scalars."""
    x = torch.zeros_like(y)
    z = torch.zeros_like(y)
    out, t_cum = [], 0.0
    for _ in range(max_k):
        t0 = time.time()
        x, _ = svt(x - alpha * grad_x(x, y - z, mask), alpha * tau)
        z = solve_z(y, x, mask, lam2)
        t_cum += time.time() - t0
        out.append({"rel_err": rel_err_of(x), "objective": objective_of(x, z),
                    "wall_time": t_cum})
    return out


@torch.no_grad()
def classical_solve(y, mask, tau, lam2, outer_iter, x_iter):
    """Inexact alternating minimization from zero: x by FISTA, z exactly."""
    x = torch.zeros_like(y)
    z = torch.zeros_like(y)
    for _ in range(outer_iter):
        x = solve_x_classical(x, y - z, mask, tau, x_iter)
        z = solve_z(y, x, mask, lam2)
    return x, z


@torch.no_grad()
def reference_optimum(y, mask, tau, lam2, outer_iter, x_iter, objective_of,
                      rel_err_of, checkpoint_every=10):
    """High-accuracy reference value f* by warm-started alternating minimization.
    The returned history shows that f* has stopped changing."""
    x = torch.zeros_like(y)
    z = torch.zeros_like(y)
    history = []
    for outer in range(1, outer_iter + 1):
        x = solve_x_classical(x, y - z, mask, tau, x_iter)
        z = solve_z(y, x, mask, lam2)
        if outer % checkpoint_every == 0 or outer == outer_iter:
            history.append({"outer_iter": outer, "total_svds": outer * x_iter,
                            "objective": objective_of(x, z), "rel_err": rel_err_of(x)})
    return history[-1]["objective"], history


class ClassicalCache:
    """Memoizes classical_solve results per (outer_iter, x_iter)."""

    def __init__(self, y, mask, tau, lam2, rel_err_of):
        self.y, self.mask, self.tau, self.lam2 = y, mask, tau, lam2
        self.rel_err_of = rel_err_of
        self._cache = {}

    def solve(self, outer_iter, x_iter):
        key = (outer_iter, x_iter)
        if key not in self._cache:
            t0 = time.time()
            x, z = classical_solve(self.y, self.mask, self.tau, self.lam2, outer_iter, x_iter)
            dt = time.time() - t0
            # objective evaluated outside the timed region (it needs another SVD)
            objective = objective_value(x, z, self.y, self.mask, self.tau, self.lam2).mean().item()
            self._cache[key] = {"rel_err": self.rel_err_of(x), "objective": objective,
                                "wall_time": dt}
        return self._cache[key]


def svds_to_match(cache, outer_iter, target, x_iter_grid, metric="objective", tol=1e-6):
    """Smallest total SVD count (outer_iter * x_iter) at which FISTA reaches
    `target`; (None, point at the largest x_iter) if never reached."""
    for x_iter in sorted(x_iter_grid):
        point = cache.solve(outer_iter, x_iter)
        if point[metric] <= target + tol:
            return outer_iter * x_iter, point
    return None, cache.solve(outer_iter, max(x_iter_grid))
