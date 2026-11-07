"""CT forward model and dictionary synthesis operator.

The image space is 512 x 512 and the acquisition is a 2D fan beam with 1000
angles and 1000 detector elements (ODL with the ASTRA backend). The
dictionary has 128 atoms of size 16 x 16 and is applied as a (transposed)
convolution with a given stride.
"""
import os

import numpy as np
import odl
import torch
from odl.contrib import torch as odl_torch

DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dics", "dic_128.npy")

# regularization parameters lambda_1 (coupling) and lambda_2 (sparsity)
LAM1 = 50.0
LAM2 = 0.0016


def get_operator():
    """ODL ray transform for the 2D fan-beam geometry."""
    size, xlim = 512, 128
    space = odl.uniform_discr([-xlim] * 2, [xlim] * 2, [size] * 2,
                              dtype="float32", weighting=1.0)
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
    detector_partition = odl.uniform_partition(-360, 360, 1000)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                        src_radius=500, det_radius=500)
    return odl.tomo.RayTransform(space, geometry)


def get_torch_operators(operator):
    """Torch modules for the forward operator, its adjoint, the FBP used for
    the initial guess (Hann, cut-off 0.75) and the low-pass FBP used to
    remove the low-frequency image content (Hann, cut-off 0.1)."""
    fbp = odl.tomo.fbp_op(operator, frequency_scaling=0.75, filter_type="Hann")
    fbp_lf = odl.tomo.fbp_op(operator, frequency_scaling=0.1, filter_type="Hann")
    return (odl_torch.OperatorModule(operator),
            odl_torch.OperatorModule(operator.adjoint),
            odl_torch.OperatorModule(fbp),
            odl_torch.OperatorModule(fbp_lf))


def op_norm_sq(op, op_adj, y_weight=None, n_iter=100, tol=1e-3):
    """Power method estimate of ||op^T W op||, W = diag(y_weight)."""
    with torch.no_grad():
        x = torch.randn_like(op_adj(y_weight))
        n = torch.inf
        for _ in range(n_iter):
            n_new = torch.sqrt(torch.sum(x ** 2))
            if torch.abs(n_new - n) < tol:
                return n_new
            n = n_new
            x = x.mul_(1 / n)
            x = op_adj(y_weight * op(x)) if y_weight is not None else op_adj(op(x))
        return torch.sqrt(torch.sum(x ** 2))


class SynthesisOperator:
    """Convolutional synthesis operator x = D z and its adjoint z = D^T x."""

    def __init__(self, stride, image_shape=(512, 512)):
        self.stride = stride
        self.image_shape = (1, 1, *image_shape)

    def __call__(self, z, D):
        return torch.nn.functional.conv_transpose2d(z, D, stride=self.stride)

    def adjoint(self, x, D):
        x = x.reshape(-1, 1, *self.image_shape[2:])
        return torch.nn.functional.conv2d(x, D, stride=self.stride)

    def norm_sq(self, D, n_iter=100):
        """Power method estimate of ||D||^2."""
        with torch.no_grad():
            x = torch.zeros(self.image_shape).type(D.type())
            z = torch.randn_like(self.adjoint(x, D))
            for _ in range(n_iter):
                z /= torch.norm(z)
                z = self.adjoint(self(z, D), D)
        return torch.norm(z)


def get_dict(stride=2, device="cuda"):
    """Pre-trained dictionary D (128 x 1 x 16 x 16) and the synthesis operator."""
    D = torch.from_numpy(np.load(DICT_PATH)).float().to(device)
    return D, SynthesisOperator(stride)
