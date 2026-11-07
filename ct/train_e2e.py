"""End-to-end training on the final objective value.

Modes:
  finetune        fine-tune the first K=10 network pairs of a greedily
                  trained model through the unrolled scheme
  semi_amortized  same architecture as the greedy model (K=10), trained
                  from scratch through the unrolled scheme
  full_amortized  one forward pass predicting (x, z) from the initial point

All modes minimize the objective F(x^K, z^K) on the training set and use
early stopping on the validation objective.

Usage (from the repository root):
    python -m ct.train_e2e --mode finetune --input_dir results/greedy_<...>
    python -m ct.train_e2e --mode semi_amortized
    python -m ct.train_e2e --mode full_amortized
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from odl.contrib import fom
from tqdm import tqdm

from common.utils import (seed_everything, make_optimizer, make_scheduler,
                          setup_logger, CSVLog, EarlyStopping)
from ct.cache import populate_initial_cache, make_loader, to_device
from ct.data import TrainingDataset, get_val_data
from ct.operators import get_operator
from ct.reconstruction import LearnedReconstruction, AmortizedReconstruction

_LAYER_PREFIXES = ("net._iterations.", "net_z.iterations.", "net_z._thetas.")


def truncate_state_dict(state_dict, n_keep):
    """Drop the per-iteration parameters of iterations >= n_keep."""
    out = {}
    for key, value in state_dict.items():
        prefix = next((p for p in _LAYER_PREFIXES if key.startswith(p)), None)
        if prefix is None or int(key[len(prefix):].split(".", 1)[0]) < n_keep:
            out[key] = value
    return out


def build(args, operator):
    """Model, trainable parameters and forward map (cached state -> (x^K, z^K))."""
    if args.mode == "full_amortized":
        model = AmortizedReconstruction(operator, dict_stride=args.dict_stride)
        params = list(model.fa_net.parameters())

        def forward(s):
            return model.predict(s["x"], s["z"], s["Lz"])
        return model, params, forward

    model = LearnedReconstruction(operator, n_iter=args.n_iter, dict_stride=args.dict_stride)
    if args.mode == "finetune":
        sd = torch.load(os.path.join(args.input_dir, "model.pt"), map_location="cpu",
                        weights_only=True)
        missing, unexpected = model.load_state_dict(truncate_state_dict(sd, args.n_iter),
                                                    strict=False)
        if unexpected or missing:
            raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]}, "
                               f"unexpected={unexpected[:5]}")
    params = list(model.net.parameters()) + list(model.net_z.parameters())

    def forward(s):
        return model.unrolled(s["x"], s["z"], s["y"], s["y_weight"], s["Lx"], s["Lz"],
                              args.n_iter)
    return model, params, forward


@torch.no_grad()
def validate(model, forward, val_image, val_data):
    obj, psnr = [], []
    for i in range(val_image.shape[0]):
        x_true, y = val_image[i:i + 1], val_data[i:i + 1]
        x0, z0, y_c, y_weight, Lx, Lz, mean, _ = model.pre_solve(y, x_true)
        x, z = forward({"x": x0, "z": z0, "y": y_c, "y_weight": y_weight, "Lx": Lx, "Lz": Lz})
        obj.append(model.objective(x, z, y_c, y_weight).item())
        psnr.append(fom.psnr((x + mean).cpu().numpy(), x_true.cpu().numpy()))
    return float(np.mean(obj)), float(np.mean(psnr))


def main(args):
    seed_everything(args.seed)
    if args.mode == "finetune":
        if args.input_dir is None:
            raise SystemExit("--input_dir is required for --mode finetune")
        with open(os.path.join(args.input_dir, "params.json")) as f:
            args.dict_stride = json.load(f).get("dict_stride", args.dict_stride)
        save_dir = args.save_dir or args.input_dir.rstrip("/") + "_fine_tuned"
    else:
        save_dir = args.save_dir or os.path.join(
            "results", f"{args.mode}_{time.strftime('%Y-%m-%d-%H%M%S')}")
    if os.path.exists(os.path.join(save_dir, "model.pt")):
        raise FileExistsError(f"{save_dir} already contains a model")
    os.makedirs(save_dir, exist_ok=True)

    log = setup_logger(save_dir, name="ct.train_e2e")
    train_csv = CSVLog(os.path.join(save_dir, "train.csv"), ["epoch", "step", "loss"])
    val_csv = CSVLog(os.path.join(save_dir, "metrics.csv"), ["epoch", "step", "val_obj", "val_psnr"])

    operator = get_operator()
    model, params, forward = build(args, operator)
    model.disable_logger()
    log.info(f"mode={args.mode}: {sum(p.numel() for p in params):,} trainable parameters")

    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump({"model": "amortized" if args.mode == "full_amortized" else "learned",
                   "n_iter": 0 if args.mode == "full_amortized" else args.n_iter,
                   "dict_stride": args.dict_stride, "train_args": vars(args)}, f, indent=2)

    dataset = TrainingDataset(operator, args.data_dir, max_samples=args.max_samples)
    val_image, val_data = get_val_data(operator, args.data_dir)
    store = populate_initial_cache(model, dataset, os.path.join(args.cache_dir,
                                                                f"initial_s{args.dict_stride}"))

    optimizer = make_optimizer(params, args.lr, args.weight_decay)
    scheduler = make_scheduler(optimizer, args.lr_scheduler, args.epochs)
    stopper = EarlyStopping(args.val_patience, args.min_delta)
    n_batches = (store.N + args.batch_size - 1) // args.batch_size
    eval_every = max(1, n_batches // args.vals_per_epoch)

    step = 0
    for epoch in range(args.epochs):
        start, ema = time.time(), None
        pbar = tqdm(make_loader(store, args.batch_size), total=n_batches,
                    desc=f"{args.mode} epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for s in pbar:
            s = to_device(s)
            x, z = forward(s)
            loss = model.objective(x, z, s["y"], s["y_weight"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            step += 1
            ema = loss.item() if ema is None else 0.05 * loss.item() + 0.95 * ema
            pbar.set_postfix(loss=f"{ema:.5f}")
            if step % 50 == 0:
                train_csv.write([epoch, step, loss.item()])

            if step % eval_every == 0:
                val_obj, val_psnr = validate(model, forward, val_image, val_data)
                val_csv.write([epoch, step, val_obj, val_psnr])
                log.info(f"  [val step {step}] obj={val_obj:.5f} PSNR={val_psnr:.2f} dB")
                if stopper.update(val_obj):
                    torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
                if stopper.should_stop:
                    break

        if scheduler is not None:
            scheduler.step()
        log.info(f"  epoch {epoch + 1}/{args.epochs}: {time.time() - start:.0f}s, "
                 f"EMA loss {ema:.5f}, lr {optimizer.param_groups[0]['lr']:.2e}")
        if stopper.should_stop:
            log.info(f"  early stopping (best val obj {stopper.best:.5f})")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    log.info(f"Training complete. Best val obj {stopper.best:.5f}; "
             f"use {save_dir}/model_best.pt for evaluation.")


def argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["finetune", "semi_amortized", "full_amortized"])
    p.add_argument("--input_dir", default=None, help="[finetune] greedy run with model.pt")
    p.add_argument("--data_dir", default="data/mayo")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--cache_dir", default=".cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_iter", type=int, default=10, help="outer iterations K (not used by full_amortized)")
    p.add_argument("--dict_stride", type=int, default=2)
    p.add_argument("--epochs", type=int, default=None, help="default: 20 (finetune), 50 (otherwise)")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-5 (finetune), 1e-4 (otherwise)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lr_scheduler", default="cosine", choices=["cosine", "none"])
    p.add_argument("--vals_per_epoch", type=int, default=4)
    p.add_argument("--val_patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = 20 if args.mode == "finetune" else 50
    if args.lr is None:
        args.lr = 1e-5 if args.mode == "finetune" else 1e-4
    return args


if __name__ == "__main__":
    main(argparser())
