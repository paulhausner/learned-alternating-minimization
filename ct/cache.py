"""On-disk cache of optimization states for training.

The initial states (x^0, z^0, centred data, weights, Lipschitz constants) of
all training samples are computed once and stored as memory-mapped arrays.
The greedy trainer advances this cache through each trained layer, so layer
k is trained on the iterates produced by layers 0..k-1 without recomputing
them. z is stored in float16 to keep the cache small.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


class SampleStore:
    """Memory-mapped arrays indexed by sample: one file per tensor plus (Lx, Lz)."""

    TENSOR_KEYS = ("x", "y", "y_weight", "z")
    TENSOR_DTYPES = {"x": "float32", "y": "float32", "y_weight": "float32", "z": "float16"}

    def __init__(self, folder):
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        self.folder = folder
        self.N = meta["N"]
        self.shapes = {k: tuple(v) for k, v in meta["shapes"].items()}
        self.dtypes = meta["dtypes"]
        self._mm = None  # opened lazily (also in each DataLoader worker)

    @classmethod
    def create(cls, folder, shapes, N):
        os.makedirs(folder, exist_ok=True)
        for k in cls.TENSOR_KEYS:
            nbytes = N * int(np.prod(shapes[k])) * np.dtype(cls.TENSOR_DTYPES[k]).itemsize
            with open(os.path.join(folder, f"{k}.bin"), "wb") as f:
                f.truncate(nbytes)
        with open(os.path.join(folder, "scalars.bin"), "wb") as f:
            f.truncate(N * 2 * 4)
        with open(os.path.join(folder, "done.bin"), "wb") as f:
            f.truncate(N)
        # meta.json last: its presence marks a fully allocated store
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump({"N": int(N), "shapes": {k: list(shapes[k]) for k in cls.TENSOR_KEYS},
                       "dtypes": dict(cls.TENSOR_DTYPES)}, f)
        return cls(folder)

    @classmethod
    def exists(cls, folder):
        return os.path.exists(os.path.join(folder, "meta.json"))

    def _open(self):
        if self._mm is None:
            mm = {k: np.memmap(os.path.join(self.folder, f"{k}.bin"),
                               dtype=np.dtype(self.dtypes[k]),
                               shape=(self.N,) + self.shapes[k], mode="r+")
                  for k in self.TENSOR_KEYS}
            mm["scalars"] = np.memmap(os.path.join(self.folder, "scalars.bin"),
                                      dtype="float32", shape=(self.N, 2), mode="r+")
            mm["done"] = np.memmap(os.path.join(self.folder, "done.bin"),
                                   dtype="uint8", shape=(self.N,), mode="r+")
            self._mm = mm
        return self._mm

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    @staticmethod
    def _scalars(v, B):
        if isinstance(v, torch.Tensor):
            return v.detach().reshape(-1).cpu().numpy().astype(np.float32)
        return np.full((B,), float(v), dtype=np.float32)

    def save_batch(self, start, state):
        """Write a batch (leading dim B) to the contiguous slots start..start+B."""
        mm = self._open()
        B = state["x"].shape[0]
        end = start + B
        for k in self.TENSOR_KEYS:
            t = state[k].detach().cpu()
            if k == "z":
                t = t.to(torch.float16)
            mm[k][start:end] = t.numpy().reshape((B,) + self.shapes[k])
        mm["scalars"][start:end, 0] = self._scalars(state["Lx"], B)
        mm["scalars"][start:end, 1] = self._scalars(state["Lz"], B)
        mm["done"][start:end] = 1

    def update_x_batch(self, start, x):
        mm = self._open()
        B = x.shape[0]
        mm["x"][start:start + B] = x.detach().cpu().numpy().reshape((B,) + self.shapes["x"])

    def load_sample(self, idx, keys=None):
        mm = self._open()
        keys = set(keys) if keys is not None else set(self.TENSOR_KEYS) | {"Lx", "Lz"}
        out = {k: torch.from_numpy(np.array(mm[k][idx], dtype=np.float32))
               for k in self.TENSOR_KEYS if k in keys}
        if "Lx" in keys:
            out["Lx"] = float(mm["scalars"][idx, 0])
        if "Lz" in keys:
            out["Lz"] = float(mm["scalars"][idx, 1])
        return out

    def is_done(self, idx):
        return bool(self._open()["done"][idx])


class StateDataset(Dataset):
    def __init__(self, store, keys=None):
        self.store, self.keys = store, keys

    def __len__(self):
        return self.store.N

    def __getitem__(self, idx):
        return self.store.load_sample(idx, self.keys)


def collate(batch):
    """Stack samples along dim 0; Lx/Lz become (B, 1, 1, 1) tensors."""
    out = {k: torch.cat([s[k] for s in batch], dim=0)
           for k in batch[0] if k not in ("Lx", "Lz")}
    for k in ("Lx", "Lz"):
        if k in batch[0]:
            out[k] = torch.tensor([float(s[k]) for s in batch]).view(-1, 1, 1, 1)
    return out


def to_device(state, device="cuda"):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in state.items()}


class ChunkShuffleSampler(Sampler):
    """Shuffle chunks of consecutive indices, then within each chunk; keeps
    reads local so the memmap pages stay in the OS page cache."""

    def __init__(self, n, chunk_size=200):
        self.n, self.chunk_size = n, chunk_size

    def __len__(self):
        return self.n

    def __iter__(self):
        chunks = [list(range(i, min(i + self.chunk_size, self.n)))
                  for i in range(0, self.n, self.chunk_size)]
        np.random.shuffle(chunks)
        for chunk in chunks:
            np.random.shuffle(chunk)
            yield from chunk


def make_loader(store, batch_size, shuffle=True, keys=None, num_workers=2):
    """Shuffled loader for training, or a sequential one (contiguous batches)
    for advancing the cache in place."""
    dataset = StateDataset(store, keys)
    sampler = ChunkShuffleSampler(len(dataset)) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, collate_fn=collate, pin_memory=True)


@torch.no_grad()
def populate_initial_cache(model, dataset, folder):
    """Fill `folder` with the pre-solve output of every training sample
    (resumable). Delete the folder when the data, the problem parameters or
    the dictionary stride change."""
    paths = list(dataset)
    N = len(paths)

    store = SampleStore(folder) if SampleStore.exists(folder) else None
    if store is not None and store.N != N:
        raise RuntimeError(f"cache {folder!r} has N={store.N} but the training set has "
                           f"N={N}; delete it and re-run")
    pending = [i for i in range(N) if store is None or not store.is_done(i)]

    for i in tqdm(pending, desc="Caching initial states", unit="sample", dynamic_ncols=True):
        image, data = dataset.load(paths[i])
        y = torch.from_numpy(data).float().cuda()
        x_true = torch.from_numpy(image).float().cuda()
        x, z, y_c, y_weight, Lx, Lz, _, _ = model.pre_solve(y, x_true)

        state = {"x": x, "y": y_c, "z": z, "y_weight": y_weight, "Lx": Lx, "Lz": Lz}
        if store is None:
            store = SampleStore.create(
                folder, {k: tuple(state[k].shape) for k in SampleStore.TENSOR_KEYS}, N)
        store.save_batch(i, state)

    return store
