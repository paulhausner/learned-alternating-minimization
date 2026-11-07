"""Greedy (layer-wise) training of the learned alternating scheme.

For each outer iteration k = 1..K:
  1. advance the cached training states through iteration k-1,
  2. train the x-network phi_k on the x-subproblem objective,
  3. update the cached x with phi_k,
  4. train the z-network psi_k on the z-subproblem objective,
  5. initialize (phi_{k+1}, psi_{k+1}) from (phi_k, psi_k).

Usage (from the repository root):
    python -m ct.train_greedy --data_dir data/mayo
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from common.utils import (seed_everything, make_optimizer, make_scheduler,
                          setup_logger, CSVLog)
from ct.cache import (SampleStore, populate_initial_cache, make_loader, to_device)
from ct.data import TrainingDataset, get_val_data
from ct.operators import get_operator
from ct.reconstruction import LearnedReconstruction


def epoch_schedule(layer):
    """(x epochs, z epochs) for 0-based layer index."""
    if layer == 0:
        return 10, 50
    if layer <= 5:
        return 5, 25
    return 2, 10


class GreedyTrainer:

    def __init__(self, model, args, save_dir):
        self.model = model
        self.args = args
        self.save_dir = save_dir
        self.log = setup_logger(save_dir, name="ct.train_greedy")
        self.train_csv = CSVLog(os.path.join(save_dir, "train.csv"),
                                ["layer", "phase", "epoch", "step", "loss"])
        self.val_csv = CSVLog(os.path.join(save_dir, "metrics.csv"),
                              ["layer", "phase", "val_obj", "val_psnr"])
        self.tmp_dir = args.tmp_dir or os.path.join(args.cache_dir, f"tmp_{os.path.basename(save_dir)}")
        self.initial_store = None
        self.tmp_store = None
        self.store = None

    # ------------------------------------------------------------------
    # cached states
    # ------------------------------------------------------------------

    def _ensure_tmp_store(self):
        if self.tmp_store is None:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            self.tmp_store = SampleStore.create(self.tmp_dir, self.initial_store.shapes,
                                                self.initial_store.N)

    @torch.no_grad()
    def _advance(self, layer):
        """Apply the z-step of iteration layer-1 to every cached state (the
        cached x already is the output of x-network layer-1)."""
        self._ensure_tmp_store()
        written = 0
        for s in tqdm(make_loader(self.store, self.args.cache_batch_size, shuffle=False),
                      desc=f"Advancing states through layer {layer}", dynamic_ncols=True):
            s = to_device(s)
            s["z"] = self.model.learned_z_step(s["x"], s["z"], s["Lz"], layer - 1)
            self.tmp_store.save_batch(written, s)
            written += s["x"].shape[0]
        self.store = self.tmp_store

    @torch.no_grad()
    def _refresh_x(self, layer):
        """Replace the cached x by the output of the just-trained x-network.
        The persistent initial cache is never modified."""
        copy = self.store is self.initial_store
        if copy:
            self._ensure_tmp_store()
        written = 0
        for s in tqdm(make_loader(self.store, self.args.cache_batch_size, shuffle=False),
                      desc=f"Refreshing x for layer {layer + 1}", dynamic_ncols=True):
            s = to_device(s)
            x = self.model.learned_x_step(s["x"], s["z"], s["Lx"], s["y"], s["y_weight"], layer)
            if copy:
                s["x"] = x
                self.tmp_store.save_batch(written, s)
            else:
                self.store.update_x_batch(written, x)
            written += x.shape[0]
        self.store = self.tmp_store

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def _train_phase(self, layer, phase, params, epochs):
        model, args = self.model, self.args
        optimizer = make_optimizer(params, lr=args.lr, weight_decay=args.weight_decay,
                                   betas=(0.9, 0.99))
        scheduler = make_scheduler(optimizer, args.lr_scheduler, epochs)
        batch_size = args.x_batch_size if phase == "x" else args.z_batch_size
        keys = None if phase == "x" else ("x", "z", "Lz")

        step = 0
        for epoch in range(epochs):
            start, ema = time.time(), None
            loader = make_loader(self.store, batch_size, keys=keys)
            pbar = tqdm(loader, desc=f"Layer {layer + 1} [{phase}] epoch {epoch + 1}/{epochs}",
                        dynamic_ncols=True)
            for s in pbar:
                s = to_device(s)
                if phase == "x":
                    x = model.learned_x_step(s["x"], s["z"], s["Lx"], s["y"], s["y_weight"], layer)
                    loss = model.loss_x(x, s["z"], s["y"], s["y_weight"])
                else:
                    z = model.learned_z_step(s["x"], s["z"], s["Lz"], layer)
                    loss = model.loss_z(s["x"], z)

                optimizer.zero_grad()
                loss.backward()
                if phase == "z":
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                step += 1
                ema = loss.item() if ema is None else 0.05 * loss.item() + 0.95 * ema
                pbar.set_postfix(loss=f"{ema:.5f}")
                if step % 50 == 0:
                    self.train_csv.write([layer, phase, epoch, step, loss.item()])

            if scheduler is not None:
                scheduler.step()
            self.log.info(f"  layer {layer + 1} [{phase}] epoch {epoch + 1}/{epochs}: "
                          f"{time.time() - start:.0f}s, EMA loss {ema:.5f}")

    @torch.no_grad()
    def _validate(self, layer, phase, val_image, val_data):
        if self.args.val_samples == 0:
            return
        self.model.enable_logger(log_psnr=True)
        obj, psnr = [], []
        for i in range(min(self.args.val_samples, val_image.shape[0])):
            _, _, logs = self.model(val_data[i:i + 1], val_image[i:i + 1], outer_iter=layer + 1)
            obj.append(logs.get_latest(logs.f))
            psnr.append(logs.get_latest(logs.p))
        self.model.disable_logger()
        self.val_csv.write([layer, phase, np.mean(obj), np.mean(psnr)])
        self.log.info(f"  [val] layers={layer + 1} obj={np.mean(obj):.5f} "
                      f"PSNR={np.mean(psnr):.2f} dB")

    def _init_next_layer(self, layer):
        if layer + 1 >= self.model.n_iter:
            return
        net, net_z = self.model.net, self.model.net_z
        net._iterations[layer + 1].load_state_dict(net._iterations[layer].state_dict())
        net_z.iterations[layer + 1].load_state_dict(net_z.iterations[layer].state_dict())
        with torch.no_grad():
            net_z._thetas[layer + 1].copy_(net_z._thetas[layer])

    def fit(self, dataset, val_image, val_data):
        self.initial_store = populate_initial_cache(
            self.model, dataset, os.path.join(self.args.cache_dir, f"initial_s{self.args.dict_stride}"))
        self.store = self.initial_store
        try:
            for layer in range(self.model.n_iter):
                epochs_x, epochs_z = epoch_schedule(layer)
                self.log.info(f"===== layer {layer + 1}/{self.model.n_iter} "
                              f"(x: {epochs_x} epochs, z: {epochs_z} epochs) =====")
                if layer > 0:
                    self._advance(layer)

                self._train_phase(layer, "x", list(self.model.net._iterations[layer].parameters()),
                                  epochs_x)
                self._refresh_x(layer)

                z_params = (list(self.model.net_z.iterations[layer].parameters())
                            + [self.model.net_z._thetas[layer]])
                self._train_phase(layer, "z", z_params, epochs_z)
                self._validate(layer, "z", val_image, val_data)

                self._init_next_layer(layer)
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model.pt"))
        finally:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)


def main(args):
    seed_everything(args.seed)
    save_dir = args.save_dir or os.path.join("results", "greedy_" + time.strftime("%Y-%m-%d-%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump({"model": "learned", "n_iter": args.n_iter,
                   "dict_stride": args.dict_stride, "train_args": vars(args)}, f, indent=2)

    operator = get_operator()
    model = LearnedReconstruction(operator, n_iter=args.n_iter, dict_stride=args.dict_stride)
    model.disable_logger()

    dataset = TrainingDataset(operator, args.data_dir, max_samples=args.max_samples)
    val_image, val_data = get_val_data(operator, args.data_dir)

    trainer = GreedyTrainer(model, args, save_dir)
    trainer.log.info(f"{len(dataset)} training samples, K={args.n_iter}, saving to {save_dir}")
    trainer.fit(dataset, val_image, val_data)
    trainer.log.info("Training complete.")


def argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/mayo", help="folder with the .npy slices")
    p.add_argument("--save_dir", default=None, help="default: results/greedy_<timestamp>")
    p.add_argument("--cache_dir", default=".cache",
                   help="initial-state cache (reused across runs) and per-run temporary cache")
    p.add_argument("--tmp_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_iter", type=int, default=20, help="number of outer iterations K")
    p.add_argument("--dict_stride", type=int, default=2)
    p.add_argument("--x_batch_size", type=int, default=4)
    p.add_argument("--z_batch_size", type=int, default=8)
    p.add_argument("--cache_batch_size", type=int, default=32,
                   help="batch size of the (gradient-free) cache updates between layers")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--lr_scheduler", default="cosine", choices=["cosine", "none"])
    p.add_argument("--val_samples", type=int, default=20,
                   help="validation samples evaluated after each layer (0 disables)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="use only the first N training samples (smoke tests); use a "
                        "separate --cache_dir when changing N")
    return p.parse_args()


if __name__ == "__main__":
    main(argparser())
