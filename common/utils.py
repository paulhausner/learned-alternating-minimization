"""Training and bookkeeping helpers shared by both problems."""
import csv
import logging
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(params, lr, weight_decay=0.01, betas=(0.9, 0.999)):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def make_scheduler(optimizer, kind, t_max):
    """Epoch-level learning-rate schedule ("cosine" or "none")."""
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if kind == "none":
        return None
    raise ValueError(f"unknown lr scheduler {kind!r}")


def setup_logger(log_dir=None, name="lopt"):
    """Logger that prints to stdout and, if log_dir is given, to log_dir/train.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class CSVLog:
    """Append-only CSV file with a fixed header."""

    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)
        self._file.flush()

    def write(self, row):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


class EarlyStopping:
    """Stop when the validation objective has not improved for `patience` checks."""

    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_checks = 0

    def update(self, value):
        """Record a validation value; returns True if it is a new best."""
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_checks = 0
            return True
        self.bad_checks += 1
        return False

    @property
    def should_stop(self):
        return self.bad_checks >= self.patience
