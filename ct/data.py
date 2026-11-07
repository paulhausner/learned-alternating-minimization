"""Simulated low-dose CT data from the 2016 AAPM Low Dose CT Grand Challenge.

The data folder is expected to contain one file per axial slice,
`<patient ID, e.g. L067>_<anything>.npy`, holding a 512 x 512 image in Hounsfield units.
Patient L286 is the test patient; every 100th slice of the remaining
(sorted) slices is used for validation and the rest for training.
"""
import os
import re

import numpy as np
import odl
import torch

MU_WATER = 0.0192          # attenuation of water in 1/mm
PHOTONS_PER_PIXEL = 50_000
TEST_PATIENT = "L286"
VAL_RATIO = 0.01
N_VAL, N_TEST = 21, 210


SLICE_PATTERN = re.compile(r"^L\d{3}_.*\.npy$")   # e.g. L067_FD_3_1.CT.0002.0001....npy


def _list_slices(data_dir, test):
    paths = []
    for dirpath, _, filenames in os.walk(data_dir):
        paths.extend(os.path.join(dirpath, f) for f in filenames
                     if SLICE_PATTERN.match(f) and f.startswith(TEST_PATIENT) == test)
    return sorted(paths)


def _split(paths, val):
    """Validation slices are taken uniformly from the sorted non-test slices."""
    n_val = int(VAL_RATIO * len(paths))
    step = (len(paths) + n_val - 1) // n_val
    return [p for i, p in enumerate(paths) if (i % step == 0) == val]


def load_image(path):
    """Slice in Hounsfield units -> attenuation, shape (1, 1, 512, 512)."""
    return (np.load(path) / 1000.0 * MU_WATER).reshape(1, 1, 512, 512)


def simulate_data(image, operator, photons_per_pixel=PHOTONS_PER_PIXEL):
    """Noisy log-transformed sinogram under the Poisson (Beer-Lambert) model;
    counts are clipped from below at 1."""
    projection = operator(operator.domain.element(image.squeeze()))
    counts = odl.phantom.poisson_noise(np.exp(-projection) * photons_per_pixel)
    noisy = np.maximum(counts, 1) / photons_per_pixel
    return -np.log(noisy).asarray()[None, None]


class TrainingDataset:
    """Training slices (shuffled with the global numpy seed). Data are
    simulated on the fly, but only once: the initial states are cached
    (see ct/cache.py)."""

    def __init__(self, operator, data_dir, max_samples=None):
        self.paths = _split(_list_slices(data_dir, test=False), val=False)
        if not self.paths:
            raise FileNotFoundError(f"no training slices found in {data_dir!r}")
        np.random.shuffle(self.paths)
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        self.operator = operator

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)

    def load(self, path):
        x = load_image(path)
        return x, simulate_data(x, self.operator)


def _fixed_set(operator, data_dir, split):
    """Validation or test set, simulated once and stored in data_dir."""
    img_file = os.path.join(data_dir, f"{split}_image.npy")
    data_file = os.path.join(data_dir, f"{split}_data.npy")

    if not os.path.exists(img_file):
        if split == "val":
            paths = _split(_list_slices(data_dir, test=False), val=True)[:N_VAL]
        else:
            paths = _list_slices(data_dir, test=True)[:N_TEST]
        images = np.concatenate([load_image(p) for p in paths])
        data = np.concatenate([simulate_data(x, operator) for x in images])
        np.save(img_file, images)
        np.save(data_file, data)

    to_pt = lambda a: torch.from_numpy(np.load(a)).float().cuda()
    return to_pt(img_file), to_pt(data_file)


def get_val_data(operator, data_dir):
    return _fixed_set(operator, data_dir, "val")


def get_test_data(operator, data_dir):
    return _fixed_set(operator, data_dir, "test")
