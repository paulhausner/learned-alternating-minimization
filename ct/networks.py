"""Network architectures for the CT experiments.

Attribute names are kept stable so that the released checkpoints load.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.prox import soft_threshold, inv_softplus


def _conv_backbone(n_in, n_out, filters, nlayers):
    """GroupNorm -> conv -> ReLU, (nlayers - 1) x [conv -> GroupNorm -> ReLU], conv."""
    layers = [nn.GroupNorm(n_in, n_in), nn.Conv2d(n_in, filters, 3, padding=1), nn.ReLU()]
    for _ in range(nlayers - 1):
        layers += [nn.Conv2d(filters, filters, 3, padding=1), nn.GroupNorm(1, filters), nn.ReLU()]
    layers.append(nn.Conv2d(filters, n_out, 3, padding=1))
    return nn.Sequential(*layers)


class LearnedUpdatesX(nn.Module):
    """One CNN per outer iteration for the x-subproblem.

    Input: [x, data-fidelity gradient, coupling gradient] (3 channels).
    Output: a correction d_x of the image size. Iterations beyond the trained
    depth reuse the last network.
    """

    def __init__(self, niter, nlayers=4, n_in=3, n_out=1, filters=32):
        super().__init__()
        self.niter = niter
        self._iterations = nn.ModuleList(
            _conv_backbone(n_in, n_out, filters, nlayers) for _ in range(niter))

    def forward(self, inp, k):
        return self._iterations[min(k, self.niter - 1)](inp)


class LearnedUpdatesZ(nn.Module):
    """One single-step LISTA network per outer iteration for the z-subproblem.

    Input: [z, grad_z] (2 * n_atoms channels). The update is
        u = z - (grad_z + d_z) / Lz,  z_new = soft_threshold(u, theta)
    with a learned per-atom threshold theta = softplus(rho). The last
    convolution is zero-initialized and theta starts at lam2 / Lz, so each
    network starts as one ISTA step.
    """

    def __init__(self, niter, theta_init, n_atoms=128, nlayers=4, filters=32):
        super().__init__()
        self.niter = niter
        self.iterations = nn.ModuleList()
        self._thetas = nn.ParameterList()

        for _ in range(niter):
            net = _conv_backbone(2 * n_atoms, n_atoms, filters, nlayers)
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.iterations.append(net)
            self._thetas.append(nn.Parameter(torch.full((n_atoms,), inv_softplus(theta_init))))

    def forward(self, z, grad_z, Lz, k):
        k = min(k, self.niter - 1)
        d_z = self.iterations[k](torch.cat([z, grad_z], dim=1))
        theta = F.softplus(self._thetas[k]).view(1, -1, 1, 1)
        return soft_threshold(z - (grad_z + d_z) / Lz, theta)


# ----------------------------------------------------------------------
# Fully-amortized baseline
# ----------------------------------------------------------------------

class SharedTrunk(nn.Module):
    """Image-domain residual trunk: input projection and `nblocks` residual blocks."""

    def __init__(self, in_ch=2, filters=96, nblocks=6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.GroupNorm(in_ch, in_ch), nn.Conv2d(in_ch, filters, 3, padding=1), nn.LeakyReLU())
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.GroupNorm(1, filters), nn.LeakyReLU(),
                          nn.Conv2d(filters, filters, 3, padding=1),
                          nn.GroupNorm(1, filters), nn.LeakyReLU(),
                          nn.Conv2d(filters, filters, 3, padding=1))
            for _ in range(nblocks))
        self.tail = nn.LeakyReLU()

    def forward(self, inp):
        x = self.input_proj(inp)
        for blk in self.blocks:
            x = x + blk(x)
        return self.tail(x)


class ConvHead(nn.Module):
    """Conv -> GroupNorm -> ReLU stack with a zero-initialized last convolution."""

    def __init__(self, in_ch, out_ch, nlayers=2, filters=None):
        super().__init__()
        filters = filters or in_ch
        layers = [nn.Conv2d(in_ch, filters, 3, padding=1), nn.GroupNorm(1, filters), nn.ReLU()]
        for _ in range(nlayers - 1):
            layers += [nn.Conv2d(filters, filters, 3, padding=1), nn.GroupNorm(1, filters), nn.ReLU()]
        last = nn.Conv2d(filters, out_ch, 3, padding=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, inp):
        return self.net(inp)


class AmortizedXZ(nn.Module):
    """Y-shaped fully-amortized network predicting (x, z) in one forward pass
    from the initial point (x^0, z^0).

    A shared trunk processes [x^0, D z^0]; the x-head adds a correction to x^0
    and the z-head predicts a LISTA correction from the trunk features
    (projected to atom space) and [z^0, D^T x, D^T D z^0], followed by one
    soft-thresholding step with a learned per-atom threshold.
    """

    def __init__(self, theta_init, n_atoms=128, trunk_blocks=6, trunk_filters=96,
                 head_x_layers=2, head_z_layers=3):
        super().__init__()
        self.trunk = SharedTrunk(in_ch=2, filters=trunk_filters, nblocks=trunk_blocks)
        self.head_x = ConvHead(trunk_filters, 1, nlayers=head_x_layers, filters=trunk_filters)
        self.proj_z = nn.Conv2d(trunk_filters, n_atoms, 1)
        self.head_z = ConvHead(4 * n_atoms, n_atoms, nlayers=head_z_layers,
                               filters=max(n_atoms, 128))
        self.rho = nn.Parameter(torch.full((n_atoms,), inv_softplus(theta_init)))

    def forward(self, x0, z0, S, D, Lz, lam1):
        Dz0 = S(z0, D)
        feat = self.trunk(torch.cat([x0, Dz0], dim=1))
        x = x0 + self.head_x(feat)

        DTx = S.adjoint(x, D)
        DTDz0 = S.adjoint(Dz0, D)
        z_ctx = self.proj_z(feat)
        if z_ctx.shape[-2:] != z0.shape[-2:]:
            z_ctx = F.adaptive_avg_pool2d(z_ctx, z0.shape[-2:])

        d_z = self.head_z(torch.cat([z0, DTx, DTDz0, z_ctx], dim=1))
        u = z0 - 2.0 * lam1 * (DTDz0 - DTx) / Lz - d_z / Lz
        theta = F.softplus(self.rho).view(1, -1, 1, 1)
        return x, soft_threshold(u, theta)
