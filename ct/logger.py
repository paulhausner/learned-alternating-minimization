"""Per-iteration log of the objective terms, PSNR, optimality gap and timing."""
import numpy as np
import torch
from odl.contrib import fom


class Log:
    """Records, for every inner step of every subproblem, the objective terms
        ly = ||Ax - y||_w^2,  lx = lam1 * ||Dz - x||^2,  l1 = lam2 * ||z||_1,
    their sum f, the PSNR, and (if a reference solution is given) the gap
    f - f* and the distances to the reference. Values are stored as one list
    per subproblem solve, so the end of outer iteration k is block 2k.
    """

    def __init__(self, y, y_weight, lam1, lam2, x_true, mean,
                 log_psnr=True, x_star=None, Dz_star=None, f_star=None, verbose=False):
        self.y, self.y_weight = y, y_weight
        self.lam1, self.lam2 = lam1, lam2
        self.x_true, self.mean = x_true, mean
        self.log_psnr = log_psnr
        self.verbose = verbose

        self.x_star, self.Dz_star, self.f_star = x_star, Dz_star, f_star
        self.has_ref = x_star is not None

        self.ly, self.lx, self.l1, self.f = [], [], [], []
        self.e, self.p = [], []                         # ||x - x_true||, PSNR
        self.e_star, self.ez_star, self.f_gap = [], [], []
        self.time = []                                  # time[0] is the pre-solve

    @staticmethod
    def get_latest(array):
        return [v for block in array for v in block][-1]

    def _new_block(self, lists):
        for lst in lists:
            lst.append([])

    def _append_f(self):
        self.f[-1].append(self.ly[-1][-1] + self.lx[-1][-1] + self.l1[-1][-1])
        if self.has_ref:
            self.f_gap[-1].append(self.f[-1][-1] - self.f_star)

    def update(self, Ax, x, Dz, z):
        """Log a full (x, z) point as its own block."""
        self._new_block([self.ly, self.lx, self.l1, self.f, self.e, self.p]
                        + ([self.e_star, self.ez_star, self.f_gap] if self.has_ref else []))
        self.ly[-1].append(torch.sum((Ax - self.y) ** 2 * self.y_weight).item())
        self.lx[-1].append(self.lam1 * torch.sum((Dz - x) ** 2).item())
        self.l1[-1].append(self.lam2 * torch.sum(torch.abs(z)).item())
        self._append_f()
        self._log_x(x, Dz)

    def update_x(self, Ax, x, Dz, ix):
        """Log after x-step `ix` of the current x-subproblem (ix == 0 opens a block)."""
        if ix == 0:
            l1 = self.get_latest(self.l1)
            self._new_block([self.ly, self.lx, self.l1, self.f, self.e, self.p]
                            + ([self.e_star, self.ez_star, self.f_gap] if self.has_ref else []))
        else:
            l1 = self.l1[-1][-1]
        self.ly[-1].append(torch.sum((Ax - self.y) ** 2 * self.y_weight).item())
        self.lx[-1].append(self.lam1 * torch.sum((Dz - x) ** 2).item())
        self.l1[-1].append(l1)
        self._append_f()
        self._log_x(x, Dz)

    def update_z(self, x, Dz, z, iz):
        """Log after z-step `iz` of the current z-subproblem (iz == 0 opens a block)."""
        ly = self.get_latest(self.ly)
        if iz == 0:
            self._new_block([self.ly, self.lx, self.l1, self.f]
                            + ([self.ez_star, self.f_gap] if self.has_ref else []))
        self.ly[-1].append(ly)
        self.lx[-1].append(self.lam1 * torch.sum((Dz - x) ** 2).item())
        self.l1[-1].append(self.lam2 * torch.sum(torch.abs(z)).item())
        self._append_f()
        if self.has_ref:
            self.ez_star[-1].append(torch.norm(Dz - self.Dz_star).item())

    def _log_x(self, x, Dz):
        self.e[-1].append(torch.norm(x - self.x_true).item())
        if self.log_psnr:
            self.p[-1].append(self.psnr(x))
        if self.has_ref:
            self.e_star[-1].append(torch.norm(x - self.x_star).item())
            self.ez_star[-1].append(torch.norm(Dz - self.Dz_star).item())

    def psnr(self, x):
        """PSNR of the full reconstruction (low-frequency part added back)."""
        return fom.psnr((x + self.mean).detach().cpu().numpy(),
                        (self.x_true + self.mean).detach().cpu().numpy())

    def log_time(self, t):
        self.time.append(t)

    def info(self, msg):
        if self.verbose:
            print(msg)

    def save(self, filename):
        obj = lambda a: np.asarray(a, dtype=object)
        np.savez(filename + ".npz",
                 ly=obj(self.ly), lx=obj(self.lx), l1=obj(self.l1), f=obj(self.f),
                 e=obj(self.e), p=obj(self.p), e_star=obj(self.e_star),
                 ez_star=obj(self.ez_star), f_gap=obj(self.f_gap), time=obj(self.time))
