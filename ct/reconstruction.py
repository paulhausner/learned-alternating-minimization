"""Alternating minimization for dictionary-regularized CT reconstruction.

    min_{x, z}  ||A x - y||_W^2 + lam1 * ||D z - x||^2 + lam2 * ||z||_1

with W = diag(exp(-y)). The low-frequency part of the image (low-pass FBP)
is removed in a pre-processing step and added back at the end, so x lives in
a centred frame.

`Reconstruction` is the inexact baseline: NAG on the x-subproblem and FISTA
on the z-subproblem, each with an inner iteration budget and an early exit
at relative objective change 1e-5. `LearnedReconstruction` replaces each
subproblem solve with one network call per outer iteration.
"""
import time

import torch

from common.prox import nesterov_step, fista_step
from ct.logger import Log
from ct.networks import LearnedUpdatesX, LearnedUpdatesZ, AmortizedXZ
from ct.operators import (LAM1, LAM2, get_dict, get_torch_operators,
                          op_norm_sq)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Reconstruction(torch.nn.Module):

    def __init__(self, operator, dict_stride=2, rel_tol=1e-5):
        super().__init__()
        self.A, self.A_adj, self.fbp, self.fbp_lf = get_torch_operators(operator)
        self.D, self.S = get_dict(stride=dict_stride)
        self.rel_tol = rel_tol

        # the image space has unit cell weighting while the data space does
        # not; rescale the regularization parameters accordingly
        dy = operator.domain.cell_volume
        dx = operator.range.cell_volume
        self.lam1 = (dy / dx) * LAM1
        self.lam2 = (dy / dx) * LAM2

        self.log_error = False   # log objective terms after every step
        self.log_psnr = False

    def enable_logger(self, log_psnr=False):
        self.log_error = True
        self.log_psnr = log_psnr

    def disable_logger(self):
        self.log_error = False
        self.log_psnr = False

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def lipschitz(self, y_weight):
        Lx = 2 * op_norm_sq(self.A, self.A_adj, y_weight, n_iter=10) + 2 * self.lam1
        Lz = 2 * self.lam1 * self.S.norm_sq(self.D)
        return Lx, Lz

    def grad_x(self, x, y, y_weight, Dz):
        """Partial gradient in x, split into data-fidelity and coupling terms."""
        return 2 * self.A_adj((self.A(x) - y) * y_weight), 2 * self.lam1 * (x - Dz)

    def grad_z(self, x, Dz):
        return 2 * self.lam1 * self.S.adjoint(Dz - x, self.D)

    @torch.no_grad()
    def f(self, x, z, y, y_weight):
        return (torch.sum((self.A(x) - y) ** 2 * y_weight)
                + self.lam1 * torch.sum((self.S(z, self.D) - x) ** 2)
                + self.lam2 * torch.sum(torch.abs(z))).item()

    def objective(self, x, z, y, y_weight):
        """Differentiable objective (summed over the batch), used as training loss."""
        return (torch.sum((self.A(x) - y) ** 2 * y_weight)
                + self.lam1 * torch.sum((self.S(z, self.D) - x) ** 2)
                + self.lam2 * torch.sum(torch.abs(z)))

    def _converged(self, f, f_prev):
        return abs((f - f_prev) / f_prev) <= self.rel_tol

    # ------------------------------------------------------------------
    # Subproblem solvers (classical)
    # ------------------------------------------------------------------

    def solve_x(self, x, z, Lx, y, y_weight, x_iter, logs, k=0):
        """Up to x_iter NAG steps on the x-subproblem."""
        Dz = self.S(z, self.D)
        _sync()
        x_extra, t = x.clone(), 1
        elapsed, f_prev = 0.0, None

        for ix in range(x_iter):
            start = time.time()
            g1, g2 = self.grad_x(x_extra, y, y_weight, Dz)
            x, x_extra, t = nesterov_step(x, x_extra, g1 + g2, Lx, t)
            f_ = (torch.sum((self.A(x) - y) ** 2 * y_weight)
                  + self.lam1 * torch.sum((Dz - x) ** 2)).item()
            _sync()
            elapsed += time.time() - start

            if self.log_error and logs is not None:
                logs.update_x(self.A(x), x, Dz, ix)
            if ix > 0 and abs(f_prev - f_) / f_prev <= self.rel_tol:
                break
            f_prev = f_

        if logs is not None:
            logs.log_time(elapsed)
        return x

    def solve_z(self, x, z, Lz, z_iter, logs, k=0):
        """Up to z_iter FISTA steps on the z-subproblem."""
        z_extra, t = z.clone(), 1
        elapsed, f_prev = 0.0, None

        for iz in range(z_iter):
            start = time.time()
            z, z_extra, t = fista_step(z, z_extra, self.grad_z(x, self.S(z_extra, self.D)),
                                       Lz, self.lam2, t)
            f_ = (self.lam1 * torch.sum((self.S(z, self.D) - x) ** 2)
                  + self.lam2 * torch.sum(torch.abs(z))).item()
            _sync()
            elapsed += time.time() - start

            if self.log_error and logs is not None:
                logs.update_z(x, self.S(z, self.D), z, iz)
            if iz > 0 and abs(f_prev - f_) / f_prev <= self.rel_tol:
                break
            f_prev = f_

        if logs is not None:
            logs.log_time(elapsed)
        return z

    # ------------------------------------------------------------------
    # Outer scheme
    # ------------------------------------------------------------------

    @torch.no_grad()
    def pre_solve(self, y, x_true, x_star=None, Dz_star=None, f_star=None, verbose=False):
        """Remove the low-frequency component, compute the weights and Lipschitz
        constants, and the initial point: x^0 = FBP(y) - mean and z^0 from 50
        FISTA steps on the z-subproblem started at zero."""
        mean = self.fbp_lf(y)
        y_weight = torch.exp(-y)
        Lx, Lz = self.lipschitz(y_weight)

        x = self.fbp(y) - mean
        z = torch.zeros_like(self.S.adjoint(x, self.D))
        z = Reconstruction.solve_z(self, x, z, Lz, 50, None)

        y = y - self.A(mean)

        logs = Log(y, y_weight, self.lam1, self.lam2, x_true - mean, mean,
                   log_psnr=self.log_psnr, x_star=x_star, Dz_star=Dz_star,
                   f_star=f_star, verbose=verbose)
        if self.log_error:
            logs.update(self.A(x), x, self.S(z, self.D), z)

        return x, z, y, y_weight, Lx, Lz, mean, logs

    def initialize(self, x, z, Lz, logs):
        """Hook to replace the initial point (used by the learned warm start)."""
        return x, z

    @torch.no_grad()
    def __call__(self, y, x_true, outer_iter=20, x_iter=1, z_iter=1,
                 x_star=None, Dz_star=None, f_star=None, verbose=False):
        """Run the alternating scheme. Returns (reconstruction, z, logs)."""
        start = time.time()
        x, z, y, y_weight, Lx, Lz, mean, logs = self.pre_solve(
            y, x_true, x_star, Dz_star, f_star, verbose)
        _sync()
        logs.log_time(time.time() - start)

        x, z = self.initialize(x, z, Lz, logs)
        f_prev = logs.get_latest(logs.f) if self.log_error else self.f(x, z, y, y_weight)

        for k in range(outer_iter):
            x = self.solve_x(x, z, Lx, y, y_weight, x_iter, logs, k=k)
            z = self.solve_z(x, z, Lz, z_iter, logs, k=k)

            f = logs.get_latest(logs.f) if self.log_error else self.f(x, z, y, y_weight)
            if self._converged(f, f_prev):
                logs.info(f"Converged after {k + 1} outer iterations.")
                break
            f_prev = f

        return x + mean, z, logs

    @torch.no_grad()
    def compute_reference(self, y, x_true, outer_iter=200, x_iter=50, z_iter=50):
        """Approximate minimizer (x*, D z*, F*) in the centred frame."""
        log_error = self.log_error
        self.disable_logger()
        x, z, y_c, y_weight, Lx, Lz, _, _ = self.pre_solve(y, x_true)
        if log_error:
            self.enable_logger(self.log_psnr)

        f_prev = self.f(x, z, y_c, y_weight)
        for _ in range(outer_iter):
            x = Reconstruction.solve_x(self, x, z, Lx, y_c, y_weight, x_iter, None)
            z = Reconstruction.solve_z(self, x, z, Lz, z_iter, None)
            f = self.f(x, z, y_c, y_weight)
            if self._converged(f, f_prev):
                break
            f_prev = f
        return x, self.S(z, self.D), self.f(x, z, y_c, y_weight)


class LearnedReconstruction(Reconstruction):
    """Learned alternating scheme with one (x, z) network pair per outer iteration.

    x-step:  x+ = x - (grad_x F(x, z) + d_x) / Lx,   d_x = phi_k(x, grads)
    z-step:  z+ = soft_threshold(z - (grad_z + d_z) / Lz, theta_k)

    `clamp_x` / `clamp_z` optionally project the learned iterates onto the
    a priori boxes |x| <= clamp_x, |z| <= clamp_z (0.1 and 4 in the paper;
    inactive on the trained models, so the outputs do not change).
    """

    def __init__(self, operator, n_iter, dict_stride=2, clamp_x=None, clamp_z=None, **kwargs):
        super().__init__(operator, dict_stride=dict_stride, **kwargs)
        self.clamp_x, self.clamp_z = clamp_x, clamp_z

        Lz_init = float(2 * self.lam1 * self.S.norm_sq(self.D))
        self.net = LearnedUpdatesX(niter=n_iter).cuda()
        self.net_z = LearnedUpdatesZ(niter=n_iter, theta_init=self.lam2 / Lz_init,
                                     n_atoms=self.D.shape[0]).cuda()

    @property
    def n_iter(self):
        return self.net.niter

    def _project(self, t, radius):
        return t if radius is None else t.clamp(-radius, radius)

    def learned_x_step(self, x, z, Lx, y, y_weight, k):
        Dz = self.S(z, self.D)
        g1, g2 = self.grad_x(x, y, y_weight, Dz)
        d = self.net(torch.cat([x, g1, g2], dim=1), k).reshape_as(x)
        return self._project(x - (g1 + g2 + d) / Lx, self.clamp_x)

    def learned_z_step(self, x, z, Lz, k):
        grad_z = self.grad_z(x, self.S(z, self.D))
        return self._project(self.net_z(z, grad_z, Lz, k), self.clamp_z)

    # losses: the subproblem objectives at the learned point (unsupervised)
    def loss_x(self, x, z, y, y_weight):
        """x-subproblem objective, averaged over the batch."""
        Dz = self.S(z, self.D)
        return (torch.sum((self.A(x) - y) ** 2 * y_weight)
                + self.lam1 * torch.sum((Dz - x) ** 2)) / x.shape[0]

    def loss_z(self, x, z):
        """z-subproblem objective, averaged over the batch."""
        coupling = self.lam1 * (self.S(z, self.D) - x).pow(2).sum(dim=(1, 2, 3))
        sparsity = self.lam2 * z.abs().sum(dim=(1, 2, 3))
        return (coupling + sparsity).mean()

    # used by the outer scheme in Reconstruction.__call__
    def solve_x(self, x, z, Lx, y, y_weight, x_iter, logs, k=0):
        start = time.time()
        x = self.learned_x_step(x, z, Lx, y, y_weight, k)
        _sync()
        if logs is not None:
            logs.log_time(time.time() - start)
            if self.log_error:
                logs.update_x(self.A(x), x, self.S(z, self.D), ix=0)
        return x

    def solve_z(self, x, z, Lz, z_iter, logs, k=0):
        start = time.time()
        z = self.learned_z_step(x, z, Lz, k)
        _sync()
        if logs is not None:
            logs.log_time(time.time() - start)
            if self.log_error:
                logs.update_z(x, self.S(z, self.D), z, iz=0)
        return z

    def unrolled(self, x, z, y, y_weight, Lx, Lz, n_iter):
        """n_iter learned outer iterations with gradients through all of them
        (semi-amortized baseline and end-to-end fine-tuning)."""
        for k in range(n_iter):
            x = self.learned_x_step(x, z, Lx, y, y_weight, k)
            z = self.learned_z_step(x, z, Lz, k)
        return x, z


class AmortizedReconstruction(Reconstruction):
    """Fully-amortized network applied to the initial point. With outer_iter=0
    this is the fully-amortized baseline; with outer_iter > 0 the classical
    inexact scheme continues from the predicted point (learned warm start)."""

    def __init__(self, operator, dict_stride=2, **amortized_kwargs):
        super().__init__(operator, dict_stride=dict_stride)
        Lz_init = float(2 * self.lam1 * self.S.norm_sq(self.D))
        self.fa_net = AmortizedXZ(theta_init=self.lam2 / Lz_init, n_atoms=self.D.shape[0],
                                  **amortized_kwargs).cuda()

    def predict(self, x, z, Lz):
        return self.fa_net(x, z, self.S, self.D, Lz, self.lam1)

    def initialize(self, x, z, Lz, logs):
        _sync()
        start = time.time()
        x, z = self.predict(x, z, Lz)
        _sync()
        logs.log_time(time.time() - start)
        if self.log_error:
            logs.update(self.A(x), x, self.S(z, self.D), z)
        return x, z
