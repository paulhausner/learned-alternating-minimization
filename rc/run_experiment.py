"""Robust matrix completion experiment.

Trains the learned scheme for several seeds, runs the baselines (FISTA with
varying inner budgets, a tuned fixed-step proximal gradient method, and ADMM
with a tuned penalty), and writes plain JSON results to --out_dir:

    config.json           resolved configuration, f*, selected rho and alpha
    reference.json        reference optimum f* and its convergence history
    seed_*/               model.pt, params.json, learned_curve.json per seed
    admm_sweep.json       ADMM curves for every rho
    admm_curve.json       ADMM curve at the selected rho
    prox_grad_sweep.json  proximal gradient curves for every step size alpha
    fista.json            FISTA at equal SVD budget and at matched objective

All comparisons use the relative optimality gap (f - f*) / f*.

Usage (from the repository root):
    python -m rc.run_experiment
"""
import argparse
import json
import os
import statistics
import time

import torch

from rc.robust_completion import (
    train_layerwise, LearnedSVTUpdates, rel_err_fn, objective_fn,
    fixed_test_batch, learned_rollout, admm_rollout, prox_grad_rollout,
    reference_optimum, ClassicalCache, svds_to_match, svt,
)

PROBLEM = dict(n=100, m=100, rank=5, obs_ratio=0.4, corruption_frac=0.05,
               tau=0.05, lam2=0.01)
TRAIN = dict(n_iter=8, steps_per_layer=400, batch_size=16, lr=1e-2)
SEEDS = [42, 43, 44, 45, 46]
TEST_SEED = 12345
N_TEST = 128
EXTRAPOLATE_FACTOR = 4                 # evaluate up to 4x the trained depth
CHECKPOINTS = [1, 4, 8, 16, 32]        # outer iterations at which FISTA is run
RHO_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
ALPHA_GRID = [1.0, 1.5, 1.8, 2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0]
REFERENCE = dict(outer_iter=60, x_iter=30)
X_ITER_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]


def train_one_seed(seed, out_dir, device):
    """Train (or load, if trained under the same config) the model for one seed."""
    seed_dir = os.path.join(out_dir, f"seed_{seed}")
    model_path = os.path.join(seed_dir, "model.pt")
    params_path = os.path.join(seed_dir, "params.json")
    params = {**PROBLEM, **TRAIN, "seed": seed, "test_seed": TEST_SEED, "device": device}

    if os.path.exists(model_path) and os.path.exists(params_path):
        with open(params_path) as f:
            if json.load(f) == params:
                net = LearnedSVTUpdates(TRAIN["n_iter"], PROBLEM["tau"],
                                        PROBLEM["n"], PROBLEM["m"])
                net.load_state_dict(torch.load(model_path, weights_only=True))
                return net.eval(), seed_dir, False
        print(f"[seed {seed}] saved model was trained with a different config; retraining")

    torch.manual_seed(seed)
    net = train_layerwise(model_niter=TRAIN["n_iter"],
                          steps_per_layer=TRAIN["steps_per_layer"],
                          batch_size=TRAIN["batch_size"], lr=TRAIN["lr"],
                          device=device, log_fn=lambda msg: None, **PROBLEM)

    os.makedirs(seed_dir, exist_ok=True)
    torch.save(net.state_dict(), model_path)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    return net.eval(), seed_dir, True


def curve_to_json(curve):
    return [{"k": k + 1, **point} for k, point in enumerate(curve)]


def dump(obj, out_dir, name):
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=2)


def main(args):
    out_dir, device = args.out_dir, args.device
    os.makedirs(out_dir, exist_ok=True)
    tau, lam2 = PROBLEM["tau"], PROBLEM["lam2"]
    trained_k = TRAIN["n_iter"]
    max_k = EXTRAPOLATE_FACTOR * trained_k

    batch = fixed_test_batch(N_TEST, PROBLEM["n"], PROBLEM["m"], PROBLEM["rank"],
                             PROBLEM["obs_ratio"], PROBLEM["corruption_frac"],
                             device, TEST_SEED)
    y, mask, x_true = batch["y"], batch["mask"], batch["x_true"]
    rel_err_of = rel_err_fn(x_true)
    objective_of = objective_fn(y, mask, tau, lam2)

    # one throwaway SVD so that library setup is not charged to the first timed solver
    svt(torch.randn(8, PROBLEM["n"], PROBLEM["m"], device=device), 0.1)

    # reference optimum
    print("Computing reference optimum f*...", flush=True)
    f_star, history = reference_optimum(y, mask, tau, lam2, REFERENCE["outer_iter"],
                                        REFERENCE["x_iter"], objective_of, rel_err_of)
    dump({"f_star": f_star, "history": history, **REFERENCE}, out_dir, "reference.json")
    print(f"  f* = {f_star:.8f} after {history[-1]['total_svds']} SVDs", flush=True)

    # learned scheme
    print("\nTraining learned scheme (existing models with the same config are reused)...",
          flush=True)
    curves = {}
    for seed in SEEDS:
        t0 = time.time()
        net, seed_dir, trained = train_one_seed(seed, out_dir, device)
        curves[seed] = learned_rollout(net, y, mask, lam2, max_k, rel_err_of, objective_of)
        dump(curve_to_json(curves[seed]), seed_dir, "learned_curve.json")
        print(f"[seed {seed}] {'trained' if trained else 'loaded'} in {time.time() - t0:.1f}s, "
              f"objective at k={trained_k}: {curves[seed][trained_k - 1]['objective']:.5f}",
              flush=True)

    # ADMM, penalty rho selected once at the trained depth
    print("\nSweeping ADMM penalty rho...", flush=True)
    admm = {rho: admm_rollout(y, mask, tau, lam2, rho, max_k, rel_err_of, objective_of)
            for rho in RHO_GRID}
    best_rho = min(RHO_GRID, key=lambda r: admm[r][trained_k - 1]["objective"])
    print(f"  selected rho = {best_rho}", flush=True)
    dump({"best_rho": best_rho, "selected_at_k": trained_k,
          "curves": {str(r): curve_to_json(c) for r, c in admm.items()}},
         out_dir, "admm_sweep.json")
    dump(curve_to_json(admm[best_rho]), out_dir, "admm_curve.json")

    # proximal gradient with a hand-tuned step size
    print("\nSweeping proximal gradient step size alpha...", flush=True)
    prox = {a: prox_grad_rollout(y, mask, tau, lam2, a, max_k, rel_err_of, objective_of)
            for a in ALPHA_GRID}
    best_alpha = min(ALPHA_GRID, key=lambda a: prox[a][trained_k - 1]["objective"])
    print(f"  selected alpha = {best_alpha} (1/L = 1)", flush=True)
    dump({"best_alpha": best_alpha, "selected_at_k": trained_k,
          "curves": {str(a): curve_to_json(c) for a, c in prox.items()}},
         out_dir, "prox_grad_sweep.json")

    # FISTA: one inner iteration (equal SVD budget) and the inner budget
    # needed to match the learned objective
    print("\nRunning FISTA at the checkpoints...", flush=True)
    cache = ClassicalCache(y, mask, tau, lam2, rel_err_of)
    equal_budget, matched = {}, {}
    for k in CHECKPOINTS:
        equal_budget[k] = cache.solve(k, 1)
        target = statistics.mean(curves[s][k - 1]["objective"] for s in SEEDS)
        n_svd, point = svds_to_match(cache, k, target, X_ITER_GRID)
        matched[k] = {"target": target, "matched_svd": n_svd, **point}
        print(f"  k={k}: SVDs to match learned objective = {n_svd}", flush=True)
    dump({"equal_budget": equal_budget, "matched": matched}, out_dir, "fista.json")

    dump({**PROBLEM, **TRAIN, "seeds": SEEDS, "test_seed": TEST_SEED, "n_test": N_TEST,
          "max_k": max_k, "checkpoints": CHECKPOINTS, "rho_grid": RHO_GRID,
          "alpha_grid": ALPHA_GRID, "reference": REFERENCE, "f_star": f_star,
          "x_iter_grid": X_ITER_GRID, "rho": best_rho, "best_alpha": best_alpha,
          "device": device}, out_dir, "config.json")

    # summary
    def gap(f):
        return (f - f_star) / f_star

    print("\nRelative optimality gap (f - f*)/f*")
    print(f"{'k':>4} | {'learned':>10} | {'ADMM':>10} | {'prox-grad':>10} | {'FISTA':>10}")
    for k in CHECKPOINTS:
        learned = statistics.mean(curves[s][k - 1]["objective"] for s in SEEDS)
        print(f"{k:>4} | {gap(learned):10.2e} | {gap(admm[best_rho][k - 1]['objective']):10.2e} | "
              f"{gap(prox[best_alpha][k - 1]['objective']):10.2e} | "
              f"{gap(equal_budget[k]['objective']):10.2e}")
    print(f"\nResults written to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/robust_completion")
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
