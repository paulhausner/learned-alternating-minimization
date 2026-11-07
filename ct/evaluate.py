"""Evaluate learned and baseline methods on the CT test set.

For every test slice, a reference solution (x*, z*, F*) is computed (or read
from --ref_cache) with the inexact scheme run for up to 200 outer and 50
inner iterations. Every selected method is then run and its per-iteration
log (objective terms, F - F*, PSNR, time) is saved to
<save_dir>/<method>/test<i>.npz.

Methods:
  learned          learned scheme (greedy or fine-tuned checkpoint, --model)
  semi_amortized   unrolled baseline (--semi_amortized_model)
  inexact          inexact AM with N inner iterations, for each N in --inexact_iters
  full_amortized   fully-amortized baseline (--full_amortized_model)
  warm_start       fully-amortized prediction followed by inexact AM

With --timing, each method is additionally run --timing_repeats times
without logging and the fastest run is stored in test<i>_timing.npz
(wall-clock time excludes the shared pre-solve).

Usage (from the repository root):
    python -m ct.evaluate --model results/greedy/model.pt --methods learned inexact \
        --inexact_iters 1 5 10 20 50 100 --timing
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from ct.data import get_test_data, get_val_data
from ct.operators import get_operator
from ct.reconstruction import (Reconstruction, LearnedReconstruction,
                               AmortizedReconstruction)

torch.backends.cudnn.benchmark = True


def load_config(checkpoint):
    with open(os.path.join(os.path.dirname(checkpoint), "params.json")) as f:
        return json.load(f)


def load_learned(operator, checkpoint, dict_stride, clamp_x=None, clamp_z=None):
    cfg = load_config(checkpoint)
    model = LearnedReconstruction(operator, n_iter=cfg["n_iter"], dict_stride=dict_stride,
                                  clamp_x=clamp_x, clamp_z=clamp_z)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


def load_amortized(operator, checkpoint, dict_stride):
    model = AmortizedReconstruction(operator, dict_stride=dict_stride)
    model.fa_net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


def build_methods(args, operator):
    """List of (name, model, call kwargs)."""
    s, K = args.dict_stride, args.outer_iters
    methods = []

    if "learned" in args.methods:
        m = load_learned(operator, args.model, s, args.clamp_x, args.clamp_z)
        methods.append(("learned", m, dict(outer_iter=m.n_iter)))

    if "semi_amortized" in args.methods:
        m = load_learned(operator, args.semi_amortized_model, s)
        methods.append(("semi_amortized", m, dict(outer_iter=m.n_iter)))

    if "inexact" in args.methods:
        m = Reconstruction(operator, dict_stride=s)
        for n in args.inexact_iters:
            methods.append((f"inexact{n}", m, dict(outer_iter=K, x_iter=n, z_iter=n)))

    if "full_amortized" in args.methods or "warm_start" in args.methods:
        m = load_amortized(operator, args.full_amortized_model, s)
        if "full_amortized" in args.methods:
            methods.append(("full_amortized", m, dict(outer_iter=0)))
        if "warm_start" in args.methods:
            for n in args.inexact_iters:
                methods.append((f"warm_start{n}", m, dict(outer_iter=K, x_iter=n, z_iter=n)))

    return methods


def reference(ref_model, y, x_true, path, ref_iters):
    if path is not None and os.path.exists(path):
        ref = torch.load(path, weights_only=False)
        return ref["x_star"], ref["Dz_star"], ref["f_star"]
    x_star, Dz_star, f_star = ref_model.compute_reference(y, x_true, outer_iter=ref_iters)
    if path is not None:
        torch.save({"x_star": x_star, "Dz_star": Dz_star, "f_star": f_star}, path)
    return x_star, Dz_star, f_star


def time_method(model, y, x_true, kwargs, repeats):
    """Fastest of `repeats` runs without logging; times exclude the pre-solve."""
    model.disable_logger()
    runs = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _, logs = model(y, x_true, **kwargs)
        torch.cuda.synchronize()
        presolve = float(logs.time[0])
        runs.append((time.perf_counter() - t0 - presolve, presolve, logs.time[1:]))
    return min(runs, key=lambda r: r[0])


def main(args):
    needs = {"learned": "model", "semi_amortized": "semi_amortized_model",
             "full_amortized": "full_amortized_model", "warm_start": "full_amortized_model"}
    for method, arg in needs.items():
        if method in args.methods and getattr(args, arg) is None:
            raise SystemExit(f"--{arg} is required for method {method!r}")

    operator = get_operator()
    methods = build_methods(args, operator)
    ref_model = Reconstruction(operator, dict_stride=args.dict_stride)

    if args.val:
        images, data = get_val_data(operator, args.data_dir)
    else:
        images, data = get_test_data(operator, args.data_dir)
    split = "val" if args.val else "test"

    ref_dir = None
    if args.ref_cache:
        ref_dir = os.path.join(args.ref_cache, f"{split}_ref{args.ref_iters}_s{args.dict_stride}")
        os.makedirs(ref_dir, exist_ok=True)
    for name, *_ in methods:
        os.makedirs(os.path.join(args.save_dir, name), exist_ok=True)

    for i in range(images.shape[0]):
        x_true, y = images[i:i + 1], data[i:i + 1]
        ref_path = os.path.join(ref_dir, f"sample_{i:03d}.pt") if ref_dir else None
        x_star, Dz_star, f_star = reference(ref_model, y, x_true, ref_path, args.ref_iters)

        for name, model, kwargs in methods:
            out_dir = os.path.join(args.save_dir, name)
            model.enable_logger(log_psnr=True)
            _, _, logs = model(y, x_true, x_star=x_star, Dz_star=Dz_star, f_star=f_star, **kwargs)
            logs.save(os.path.join(out_dir, f"test{i}"))
            f = logs.get_latest(logs.f)
            print(f"[{i + 1}/{images.shape[0]}] {name:16s} F={f:.4f}  "
                  f"rel. gap={(f - f_star) / f_star:.4f}  PSNR={logs.get_latest(logs.p):.2f} dB")

            if args.timing:
                wall, presolve, step_times = time_method(model, y, x_true, kwargs,
                                                         args.timing_repeats)
                np.savez(os.path.join(out_dir, f"test{i}_timing.npz"),
                         wall=wall, presolve=presolve,
                         step_times=np.asarray(step_times, dtype=np.float64),
                         inner_iters=kwargs.get("x_iter", 0))
                print(f"    wall={wall:.3f}s (pre-solve {presolve:.3f}s excluded)")


def argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["learned", "inexact"],
                   choices=["learned", "semi_amortized", "inexact", "full_amortized", "warm_start"])
    p.add_argument("--model", default=None, help="learned checkpoint (params.json next to it)")
    p.add_argument("--semi_amortized_model", default=None)
    p.add_argument("--full_amortized_model", default=None)
    p.add_argument("--data_dir", default="data/mayo")
    p.add_argument("--save_dir", default="test_results")
    p.add_argument("--val", action="store_true", help="evaluate on the validation set")
    p.add_argument("--dict_stride", type=int, default=2)
    p.add_argument("--outer_iters", type=int, default=20,
                   help="outer iterations of inexact / warm-start runs")
    p.add_argument("--inexact_iters", type=int, nargs="+", default=[10],
                   help="inner iteration budgets of the inexact (and warm-start) baselines")
    p.add_argument("--ref_iters", type=int, default=200,
                   help="outer iterations of the reference solve")
    p.add_argument("--ref_cache", default=".cache/reference",
                   help="folder to store/reuse reference solutions ('' disables)")
    p.add_argument("--clamp_x", type=float, default=None,
                   help="a priori box for the learned x-iterates (paper: 0.1)")
    p.add_argument("--clamp_z", type=float, default=None,
                   help="a priori box for the learned z-iterates (paper: 4)")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--timing_repeats", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    main(argparser())
