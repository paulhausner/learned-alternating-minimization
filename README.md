# Learning inexact alternating minimization for convex problems with quadratic coupling

This is the official repository for the NeurIPS paper _Learning inexact alternating minimization for convex problems with quadratic coupling_ by Paul Häusner, Jevgenija Rudzusika, Jens Sjölund, and Ozan Öktem.

We replace the inexact subproblem solves of alternating minimization with one learned update per outer iteration. The networks are trained greedily, one iteration at a time, on the objective of the corresponding subproblem.

## Repository layout

```
common/                 shared by both experiments
  prox.py               soft-thresholding, NAG / FISTA steps
  utils.py              seeding, optimizer, LR schedule, logging, early stopping
ct/                     dictionary-regularized low-dose CT reconstruction
  operators.py          fan-beam ray transform (ODL + ASTRA), FBPs, dictionary operator
  data.py               Mayo slices -> simulated noisy sinograms
  networks.py           x- and z-networks, fully-amortized baseline network
  reconstruction.py     inexact AM baseline, learned scheme, amortized / warm-start
  cache.py              on-disk cache of optimization states used during training
  logger.py             per-iteration objective, gap, PSNR and timing logs
  train_greedy.py       greedy layer-wise training (proposed method)
  train_e2e.py          end-to-end fine-tuning and the semi-/fully-amortized baselines
  evaluate.py           test-set evaluation of all methods, reference solutions, timing
  dics/dic_128.npy      pre-trained dictionary (128 atoms of size 16 x 16)
rc/                     robust matrix completion
  robust_completion.py  problem, learned SVT update, FISTA / proximal gradient / ADMM
  run_experiment.py     training over 5 seeds and all baselines
```

All commands are run from the repository root.

## Installation

```bash
pip install -r requirements.txt
```

The robust matrix completion experiment only needs PyTorch and runs on the CPU.
The CT experiments need a CUDA GPU and the ASTRA toolbox with CUDA support, which is easiest to install with conda (`conda install -c astra-toolbox astra-toolbox`). We used Python 3.12 and PyTorch 2.

## Robust matrix completion

```bash
python -m rc.run_experiment --out_dir results/robust_completion
```

This trains the learned scheme (K = 8) for 5 seeds, computes the reference optimum, and runs FISTA with varying inner budgets, the proximal gradient method with a tuned step size, and ADMM with a tuned penalty. All curves are written as JSON to `--out_dir` and a summary of the relative optimality gaps is printed. Training a seed takes a few minutes on a CPU; models trained with the same configuration are reused on reruns.

## CT reconstruction

### Data

We use the 2016 AAPM Low Dose CT Grand Challenge data. The license does not allow redistribution.
The data can be obtained from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) after registration.

Store the full-dose image volumes as axial 512 x 512 slices in Hounsfield units, one `.npy` file per slice, named with the patient ID as prefix (e.g. `data/mayo/L067_0001.npy`). Patient L286 is the test patient and every 100th of the remaining slices is used for validation (21 slices), which leaves 2147 training slices. The noisy sinograms (fan beam, 1000 angles, 1000 detector elements, 50 000 photons per pixel) are simulated from these slices. The validation and test sets are simulated once and stored in the data folder as `{val,test}_{image,data}.npy`.

### Training

```bash
# proposed method: greedy training, K = 20
python -m ct.train_greedy --data_dir data/mayo --save_dir results/greedy

# end-to-end fine-tuning of the first 10 iterations
python -m ct.train_e2e --mode finetune --input_dir results/greedy

# learned baselines
python -m ct.train_e2e --mode semi_amortized --save_dir results/semi_amortized
python -m ct.train_e2e --mode full_amortized --save_dir results/full_amortized
```

The first run computes and caches the initial states (FBP initialization and warm-started sparse codes) of all training slices in `.cache/` (about 55 GB for the stride-2 dictionary). Later runs reuse the cache. Greedy training additionally writes a temporary cache of the same size, which is deleted at the end of the run.

### Evaluation

```bash
# learned scheme and the inexact baseline with different inner budgets (Pareto front)
python -m ct.evaluate --model results/greedy/model.pt --methods learned inexact \
    --inexact_iters 1 2 5 10 20 50 100 --timing --save_dir test_results/greedy

# fine-tuned and learned baselines
python -m ct.evaluate --model results/greedy_fine_tuned/model_best.pt --methods learned \
    --save_dir test_results/fine_tuned
python -m ct.evaluate --methods semi_amortized full_amortized \
    --semi_amortized_model results/semi_amortized/model_best.pt \
    --full_amortized_model results/full_amortized/model_best.pt --save_dir test_results/baselines

# learned warm start followed by the inexact baseline
python -m ct.evaluate --methods warm_start --full_amortized_model results/full_amortized/model_best.pt \
    --inexact_iters 1 5 10 50 --save_dir test_results/warm_start
```

The reference solution of each test slice is computed once and cached in `.cache/reference/`. For each method and test slice, `ct.evaluate` saves the per-iteration objective terms, relative gap to the reference, PSNR and subproblem times in `<save_dir>/<method>/test<i>.npz` and, with `--timing`, the fastest of three timed runs in `test<i>_timing.npz`.

## Citation

```bibtex
@inproceedings{hausner2026learning,
  title     = {Learning inexact alternating minimization for convex problems with quadratic coupling},
  author    = {H{\"a}usner, Paul and Rudzusika, Jevgenija and Sj{\"o}lund, Jens and {\"O}ktem, Ozan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
