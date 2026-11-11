"""
Dataset utilities for the SymbTr makam classification experiments.

Changes w.r.t. the original submission:
  * stratified (by makam) train/val/test split, controlled by a single seed;
  * split file lists are written to <split_dir>/seed<k>_{train,val,test}.txt
    so that every experiment can be reproduced exactly;
  * optional test-time perturbations needed by the revision experiments
    (note-order shuffling, komma transposition) are implemented here so that
    the model code stays unchanged;
  * helper to convert (pitch class, accidental, octave) to an absolute
    53-TET komma index, used by the histogram baselines and the
    transposition / interval experiments.
"""

import os
import pickle
import random
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------
# Vocabularies (fixed, identical to vocab.pkl)
# ----------------------------------------------------------------------------
PC_VOCAB = {"PAD": 0, "Rest": 1, "G": 2, "A": 3, "B": 4, "C": 5, "D": 6, "E": 7, "F": 8}
ACC_VOCAB = {"PAD": 0, None: 1, "#1": 2, "#2": 3, "#3": 4, "#4": 5, "#5": 6, "#6": 7, "#7": 8, "#8": 9}
OCT_VOCAB = {"PAD": 0, None: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
MEAS_VOCAB = {"PAD": 0, "start": 1, "middle": 2, "end": 3}
MAKAM_VOCAB = {
    "hicaz": 0, "nihavent": 1, "ussak": 2, "rast": 3,
    "huzzam": 4, "segah": 5, "huseyni": 6, "mahur": 7,
    "hicazkar": 8, "kurdilihicazkar": 9, "muhayyer": 10, "saba": 11,
}
NUM_CLASSES = len(MAKAM_VOCAB)

# Interval representation (transposition-invariant pitch feature):
# komma difference to the previous pitched note, clipped to +-INT_CLIP.
#   id 0 = PAD, 1 = rest, 2 = first pitched note (no predecessor),
#   3 .. 3+2*INT_CLIP  = interval -INT_CLIP .. +INT_CLIP
INT_CLIP = 40
INT_VOCAB_SIZE = 3 + 2 * INT_CLIP + 1

# komma offset of each natural pitch class from C within an octave
# (C-D 9, D-E 9, E-F 4, F-G 9, G-A 9, A-B 9, B-C 4; octave number increments at C,
#  as in SymbTr pitch names, so C4 < G4). Verified against SymbTr's KomaAE column
#  (theoretical AEU komma index; the practice-adjusted Koma53 column differs).
_PC_KOMMA_BASE = {"C": 0, "D": 9, "E": 18, "F": 22, "G": 31, "A": 40, "B": 49}
_PC_ID_TO_NAME = {v: k for k, v in PC_VOCAB.items()}
_OCT_ID_TO_VAL = {v: k for k, v in OCT_VOCAB.items()}


def komma_index(pc_id, acc_id, oct_id):
    """
    Absolute 53-TET komma index of a note (None for rests).
    pc_id/acc_id/oct_id are the integer ids used in dataset.pkl.
    acc id k (>=2) means k-1 kommas sharp (e.g. id 6 == '#5').
    """
    pc = _PC_ID_TO_NAME[pc_id]
    if pc == "Rest":
        return None
    octave = _OCT_ID_TO_VAL[oct_id]
    acc_komma = 0 if acc_id <= 1 else acc_id - 1
    return octave * 53 + _PC_KOMMA_BASE[pc] + acc_komma


def komma_to_ids(k):
    """Inverse of komma_index: komma index -> (pc_id, acc_id, oct_id)."""
    octave, rem = divmod(k, 53)
    # find the natural pitch class with the largest base <= rem
    pc, base = max(((p, b) for p, b in _PC_KOMMA_BASE.items() if b <= rem),
                   key=lambda x: x[1])
    acc = rem - base
    pc_id = PC_VOCAB[pc]
    acc_id = 1 if acc == 0 else acc + 1
    oct_id = OCT_VOCAB.get(octave)
    if oct_id is None:
        # octave out of vocabulary (can happen after transposition) -> clip
        oct_id = OCT_VOCAB[min(max(octave, 3), 7)]
    return pc_id, acc_id, oct_id


def interval_ids(notes):
    """Interval token ids for a list of (pc, acc, oct, dur, meas) notes."""
    out, prev = [], None
    for pc, acc, octv, dur, meas in notes:
        k = komma_index(pc, acc, octv)
        if k is None:
            out.append(1)
            continue
        if prev is None:
            out.append(2)
        else:
            d = max(-INT_CLIP, min(INT_CLIP, k - prev))
            out.append(3 + d + INT_CLIP)
        prev = k
    return out


# ----------------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Loading and splitting
# ----------------------------------------------------------------------------
def load_pieces(vocab_file="vocab.pkl", dataset_file="dataset.pkl"):
    """Returns (pieces, labels, vocab). Pieces with zero notes are dropped."""
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    pieces, labels = [], []
    for makam, makam_pieces in dataset.items():
        label = vocab["makam"][makam]
        for piece in makam_pieces:
            if len(piece["notes"]) == 0:
                print(f"[dataset] skipping empty piece {piece['f_name']}")
                continue
            pieces.append(piece)
            labels.append(label)
    return pieces, labels, vocab


def make_split_indices(labels, seed, stratify=True, val_frac=0.15, test_frac=0.15):
    """Stratified 70/15/15 split (by default) of indices, reproducible by seed."""
    idx = np.arange(len(labels))
    labels = np.asarray(labels)
    strat = labels if stratify else None
    tr_idx, rest_idx = train_test_split(
        idx, test_size=val_frac + test_frac, random_state=seed, stratify=strat)
    strat_rest = labels[rest_idx] if stratify else None
    val_idx, te_idx = train_test_split(
        rest_idx, test_size=test_frac / (val_frac + test_frac),
        random_state=seed, stratify=strat_rest)
    return np.sort(tr_idx), np.sort(val_idx), np.sort(te_idx)


def write_split_files(pieces, split, seed, split_dir="splits"):
    os.makedirs(split_dir, exist_ok=True)
    for name, idx in zip(("train", "val", "test"), split):
        path = os.path.join(split_dir, f"seed{seed}_{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for i in idx:
                f.write(pieces[i]["f_name"] + "\n")


def read_split_files(pieces, seed, split_dir="splits"):
    """Rebuild index split from the saved file lists (returns None if absent)."""
    name_to_idx = {p["f_name"]: i for i, p in enumerate(pieces)}
    out = []
    for name in ("train", "val", "test"):
        path = os.path.join(split_dir, f"seed{seed}_{name}.txt")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            out.append(np.array([name_to_idx[l.strip()] for l in f if l.strip()]))
    return tuple(out)


def compute_class_weights(labels, num_classes=NUM_CLASSES):
    """w_c = N / (K * n_c)"""
    counts = Counter(labels)
    total = sum(counts.values())
    return torch.tensor([total / (num_classes * counts[c]) for c in range(num_classes)],
                        dtype=torch.float)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class MakamDataset(Dataset):
    """
    pieces: list of {"f_name": str, "notes": [(pc, acc, oct, dur, meas), ...]}
    labels: list of int

    Perturbations (all off by default; used in the control experiments):
      shuffle_order : permute the note order of every piece (pitch inventory kept)
      transpose     : shift every pitched note by this many kommas (53-TET)
      transpose_aug : if > 0, shift each piece by a random integer in
                      [-transpose_aug, +transpose_aug] kommas every time it is
                      drawn (training-time augmentation)
      drop_rests    : remove rest events
    The item also carries "ints", the interval representation (see interval_ids).
    """

    def __init__(self, pieces, labels, shuffle_order=False, transpose=0,
                 transpose_aug=0, drop_rests=False, seed=0):
        assert len(pieces) == len(labels)
        self.pieces = pieces
        self.labels = labels
        self.shuffle_order = shuffle_order
        self.transpose = transpose
        self.transpose_aug = transpose_aug
        self.drop_rests = drop_rests
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.pieces)

    def _perturb(self, notes):
        if self.drop_rests:
            notes = [n for n in notes if n[0] != PC_VOCAB["Rest"]] or notes
        shift = self.transpose
        if self.transpose_aug:
            shift += int(self.rng.randint(-self.transpose_aug, self.transpose_aug + 1))
        if shift:
            new = []
            for pc, acc, octv, dur, meas in notes:
                k = komma_index(pc, acc, octv)
                if k is None:
                    new.append((pc, acc, octv, dur, meas))
                else:
                    pc2, acc2, oct2 = komma_to_ids(k + shift)
                    new.append((pc2, acc2, oct2, dur, meas))
            notes = new
        if self.shuffle_order:
            notes = list(notes)
            self.rng.shuffle(notes)
        return notes

    def __getitem__(self, idx):
        piece = self.pieces[idx]
        notes = self._perturb(piece["notes"])
        cols = list(zip(*notes))
        return {
            "pcs": torch.tensor(cols[0], dtype=torch.long),
            "accs": torch.tensor(cols[1], dtype=torch.long),
            "octs": torch.tensor(cols[2], dtype=torch.long),
            "durs": torch.tensor(cols[3], dtype=torch.long),
            "meas": torch.tensor(cols[4], dtype=torch.long),
            "ints": torch.tensor(interval_ids(notes), dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "length": len(notes),
            "f_name": piece["f_name"],
        }


def collate_fn(batch):
    # sort by length (descending) -- required by pack_padded_sequence
    batch.sort(key=lambda x: x["length"], reverse=True)
    out = {}
    for key in ("pcs", "accs", "octs", "durs", "meas", "ints"):
        out[key] = pad_sequence([b[key] for b in batch], batch_first=True, padding_value=0)
    out["mask"] = (out["pcs"] != 0).long()
    out["lengths"] = torch.tensor([b["length"] for b in batch])
    out["labels"] = torch.stack([b["label"] for b in batch])
    out["f_names"] = np.array([b["f_name"] for b in batch])
    return out


class BucketBatchSampler(Sampler):
    """
    Groups pieces of similar length into the same mini-batch to reduce padding
    and the number of recurrent steps per epoch. Pieces are shuffled, cut into
    pools of `pool_factor` batches, sorted by length inside each pool, split into
    batches, and the batch order is shuffled again. Model-agnostic: it only
    changes which pieces share a batch, not the optimisation objective.
    """

    def __init__(self, lengths, batch_size, seed=0, pool_factor=20, shuffle=True):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.pool = batch_size * pool_factor
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths)) if self.shuffle else np.argsort(-self.lengths)
        batches = []
        for start in range(0, len(idx), self.pool):
            pool = idx[start:start + self.pool]
            pool = pool[np.argsort(-self.lengths[pool], kind="stable")]
            batches += [pool[i:i + self.batch_size].tolist() for i in range(0, len(pool), self.batch_size)]
        if self.shuffle:
            order = self.rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        return iter(batches)

    def __len__(self):
        return int(np.ceil(len(self.lengths) / self.batch_size))


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def build_dataloaders(vocab_file="vocab.pkl", dataset_file="dataset.pkl",
                      batch_size=8, seed=0, stratify=True, split_dir="splits",
                      test_perturbation=None, train_perturbation=None,
                      num_workers=0, bucket=True):
    """
    Returns train_loader, class_weights, val_loader, test_loader
    (same order as the original implementation).

    test_perturbation : optional dict of MakamDataset kwargs applied to the
                        TEST set only, e.g. {"shuffle_order": True} or {"transpose": 9}.
    train_perturbation: optional dict applied to the TRAIN set only, e.g.
                        {"shuffle_order": True} (bag-of-notes upper bound) or
                        {"transpose_aug": 13} (transposition augmentation).
    """
    pieces, labels, _ = load_pieces(vocab_file, dataset_file)

    split = read_split_files(pieces, seed, split_dir) if split_dir else None
    if split is None:
        split = make_split_indices(labels, seed, stratify=stratify)
        if split_dir:
            write_split_files(pieces, split, seed, split_dir)
    tr_idx, va_idx, te_idx = split

    def subset(idx, **kw):
        return MakamDataset([pieces[i] for i in idx], [labels[i] for i in idx], **kw)

    ds_train = subset(tr_idx, **(train_perturbation or {}), seed=seed + 1000)
    ds_val = subset(va_idx)
    ds_test = subset(te_idx, **(test_perturbation or {}), seed=seed)

    class_weights = compute_class_weights([labels[i] for i in tr_idx])

    if bucket:
        def lens(idx):
            return [len(pieces[i]["notes"]) for i in idx]
        train_loader = DataLoader(ds_train, collate_fn=collate_fn, num_workers=num_workers,
                                  batch_sampler=BucketBatchSampler(lens(tr_idx), batch_size, seed))
        val_loader = DataLoader(ds_val, collate_fn=collate_fn, num_workers=num_workers,
                                batch_sampler=BucketBatchSampler(lens(va_idx), batch_size, shuffle=False))
        test_loader = DataLoader(ds_test, collate_fn=collate_fn, num_workers=num_workers,
                                 batch_sampler=BucketBatchSampler(lens(te_idx), batch_size, shuffle=False))
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, generator=g, num_workers=num_workers)
        val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers)
        test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, class_weights, val_loader, test_loader


def split_summary(vocab_file="vocab.pkl", dataset_file="dataset.pkl", seeds=range(5),
                  split_dir="splits"):
    """Per-makam counts in train/val/test for each seed (for the appendix table)."""
    import pandas as pd
    pieces, labels, vocab = load_pieces(vocab_file, dataset_file)
    inv = {v: k for k, v in vocab["makam"].items()}
    rows = []
    for seed in seeds:
        split = read_split_files(pieces, seed, split_dir)
        if split is None:
            split = make_split_indices(labels, seed)
            write_split_files(pieces, split, seed, split_dir)
        for name, idx in zip(("train", "val", "test"), split):
            c = Counter(labels[i] for i in idx)
            for lab, n in c.items():
                rows.append({"seed": seed, "subset": name, "makam": inv[lab], "n": n})
    df = pd.DataFrame(rows)
    return df.pivot_table(index="makam", columns=["seed", "subset"], values="n", fill_value=0)


if __name__ == "__main__":
    tr, cw, va, te = build_dataloaders(batch_size=4, seed=0)
    print(len(tr.dataset), len(va.dataset), len(te.dataset))
    print("class weights:", cw)
    b = next(iter(tr))
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in b.items() if k != "f_names"})
    print(split_summary(seeds=[0]))
