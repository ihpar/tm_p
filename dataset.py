import os
import torch
import pickle
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from constants import *


PC_ID_TO_NAME = {v: k for k, v in PC_VOCAB.items()}
OCT_ID_TO_VALUE = {v: k for k, v in OCT_VOCAB.items()}


# ---------------------------------------------------------------------------
# Komma arithmetic
# ---------------------------------------------------------------------------
def komma_index(pc_id, acc_id, oct_id):
    """
    Absolute komma index of a note (53 kommas per octave).
    Returns None for a rest.

    pc_id, acc_id, oct_id: the integer ids used in dataset.pkl.
    acc_id k (k >= 2) means k - 1 kommas sharp (e.g. id 6 is "#5").
    """
    pc_name = PC_ID_TO_NAME[pc_id]
    if pc_name == "Rest":
        return None

    octave = OCT_ID_TO_VALUE[oct_id]

    if acc_id <= 1:
        acc_kommas = 0
    else:
        acc_kommas = acc_id - 1

    return octave * KOMMAS_PER_OCTAVE + PC_KOMMA_BASE[pc_name] + acc_kommas


def komma_to_ids(komma):
    """
    Inverse of komma_index.
    komma index -> (pc_id, acc_id, oct_id)
    """
    octave = komma // KOMMAS_PER_OCTAVE
    remainder = komma % KOMMAS_PER_OCTAVE

    # the natural pitch class with the largest komma offset <= remainder
    pc_name = None
    pc_base = -1
    for name, base in PC_KOMMA_BASE.items():
        if base <= remainder and base > pc_base:
            pc_name = name
            pc_base = base

    acc_kommas = remainder - pc_base

    pc_id = PC_VOCAB[pc_name]

    if acc_kommas == 0:
        acc_id = 1
    else:
        acc_id = acc_kommas + 1

    if octave in OCT_VOCAB:
        oct_id = OCT_VOCAB[octave]
    else:
        # the octave is outside the vocabulary
        # (this can happen after a transposition) -> clip it
        clipped_octave = min(max(octave, LOWEST_OCTAVE), HIGHEST_OCTAVE)
        oct_id = OCT_VOCAB[clipped_octave]

    return pc_id, acc_id, oct_id


def interval_ids(notes):
    """
    Interval token ids of a piece.

    notes: List[Tuple[int,int,int,int,int]]  (pc, acc, oct, dur, meas)
    returns: List[int], one interval token per note
    """
    ids = []
    prev_komma = None

    for pc, acc, octave, dur, meas in notes:
        komma = komma_index(pc, acc, octave)

        if komma is None:
            ids.append(INTERVAL_REST_ID)
            continue

        if prev_komma is None:
            ids.append(INTERVAL_FIRST_NOTE_ID)
        else:
            interval = komma - prev_komma
            # clip the interval to [-INTERVAL_CLIP, +INTERVAL_CLIP]
            interval = max(-INTERVAL_CLIP, min(INTERVAL_CLIP, interval))
            ids.append(INTERVAL_OFFSET + interval + INTERVAL_CLIP)

        prev_komma = komma

    return ids


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------
def load_pieces(vocab_file=VOCAB_FILE, dataset_file=DATASET_FILE):
    """
    returns pieces, labels, vocab

    pieces: List[ {"f_name": str, "notes": List[Tuple[int,int,int,int,int]]} ]
    labels: List[int]  (makam labels 0-11)
    """
    vocab, dataset = None, None
    with open(vocab_file, "rb") as in_file:
        vocab = pickle.load(in_file)
    with open(dataset_file, "rb") as in_file:
        dataset = pickle.load(in_file)

    pieces, labels = [], []
    for makam, makam_pieces in dataset.items():
        label = vocab["makam"][makam]
        for makam_piece in makam_pieces:
            if len(makam_piece["notes"]) == 0:
                print(f"Skipping empty piece {makam_piece['f_name']}")
                continue
            pieces.append(makam_piece)
            labels.append(label)

    return pieces, labels, vocab


def get_duration_values(vocab):
    """
    returns {duration id: relative duration}, with 0.0 for the padding id
    """
    dur_vocab_inv = {v: k for k, v in vocab["dur"].items()}
    dur_vocab_inv[PAD_ID] = 0.0
    return dur_vocab_inv


def make_split_indices(labels, seed):
    """
    Stratified 70/15/15 split of the piece indices, reproducible by seed.
    returns train_idx, val_idx, test_idx
    """
    indices = np.arange(len(labels))
    labels = np.asarray(labels)

    rest_fraction = VAL_FRACTION + TEST_FRACTION

    train_idx, rest_idx = train_test_split(
        indices,
        test_size=rest_fraction,
        random_state=seed,
        stratify=labels
    )

    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=TEST_FRACTION / rest_fraction,
        random_state=seed,
        stratify=labels[rest_idx]
    )

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def write_split_files(pieces, split, seed, split_dir=SPLIT_DIR):
    """
    Writes the file names of the train, val and test pieces to
    <split_dir>/seed<seed>_train.txt, _val.txt and _test.txt
    """
    os.makedirs(split_dir, exist_ok=True)

    subset_names = ["train", "val", "test"]
    for subset_name, subset_idx in zip(subset_names, split):
        path = os.path.join(split_dir, f"seed{seed}_{subset_name}.txt")
        with open(path, "w", encoding="utf-8") as out_file:
            for i in subset_idx:
                out_file.write(pieces[i]["f_name"] + "\n")


def read_split_files(pieces, seed, split_dir=SPLIT_DIR):
    """
    Rebuilds the split from the saved file lists.
    returns (train_idx, val_idx, test_idx), or None if the files are absent
    """
    f_name_to_idx = {}
    for i, piece in enumerate(pieces):
        f_name_to_idx[piece["f_name"]] = i

    split = []
    for subset_name in ["train", "val", "test"]:
        path = os.path.join(split_dir, f"seed{seed}_{subset_name}.txt")
        if not os.path.exists(path):
            return None

        subset_idx = []
        with open(path, encoding="utf-8") as in_file:
            for line in in_file:
                f_name = line.strip()
                if f_name:
                    subset_idx.append(f_name_to_idx[f_name])

        split.append(np.array(subset_idx))

    return tuple(split)


def get_split(pieces, labels, seed, split_dir=SPLIT_DIR):
    """
    Reads the split of this seed from <split_dir>.
    If it does not exist yet, makes the split and writes it there.
    returns train_idx, val_idx, test_idx
    """
    split = read_split_files(pieces, seed, split_dir)

    if split is None:
        split = make_split_indices(labels, seed)
        write_split_files(pieces, split, seed, split_dir)

    return split


def compute_class_weights(labels, num_classes=NUM_CLASSES):
    """
    weight of class c = N / (K * n_c)
    N: number of pieces, K: number of classes, n_c: pieces of class c
    """
    counts = Counter(labels)
    total = sum(counts.values())

    weights = []
    for c in range(num_classes):
        weights.append(total / (num_classes * counts[c]))

    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MakamDataset(Dataset):
    def __init__(
        self,
        pieces,
        labels,
        shuffle_order=False,
        transpose=0,
        transpose_aug=0,
        seed=0
    ):
        """
        pieces: List[ {"f_name": str, "notes": List[Tuple[int,int,int,int,int]]} ]
                Each piece = [(pc, acc, oct, dur, meas), ...]
        labels: List[int]  (makam labels 0-11)

        Perturbations for the control experiments (all off by default):
        shuffle_order: permute the note order of every piece
        transpose:     shift every pitched note by this many kommas
        transpose_aug: if > 0, shift each piece by a random number of kommas
                       in [-transpose_aug, +transpose_aug] every time it is
                       drawn (augmentation of the training pieces)
        seed:          seed of the random perturbations
        """
        assert len(pieces) == len(labels)

        self.pieces = pieces
        self.labels = labels
        self.shuffle_order = shuffle_order
        self.transpose = transpose
        self.transpose_aug = transpose_aug
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.pieces)

    def _transpose_notes(self, notes, shift):
        """
        Shifts every pitched note by `shift` kommas. Rests are not changed.
        """
        transposed_notes = []
        for pc, acc, octave, dur, meas in notes:
            komma = komma_index(pc, acc, octave)
            if komma is None:
                transposed_notes.append((pc, acc, octave, dur, meas))
            else:
                new_pc, new_acc, new_octave = komma_to_ids(komma + shift)
                transposed_notes.append(
                    (new_pc, new_acc, new_octave, dur, meas))

        return transposed_notes

    def _perturb(self, notes):
        shift = self.transpose
        if self.transpose_aug:
            random_shift = self.rng.randint(
                -self.transpose_aug, self.transpose_aug + 1)
            shift += int(random_shift)

        if shift:
            notes = self._transpose_notes(notes, shift)

        if self.shuffle_order:
            notes = list(notes)
            self.rng.shuffle(notes)

        return notes

    def __getitem__(self, idx):
        piece = self.pieces[idx]
        label = self.labels[idx]

        notes = self._perturb(piece["notes"])

        # Split features into separate tensors
        pcs = torch.tensor([n[0] for n in notes], dtype=torch.long)
        accs = torch.tensor([n[1] for n in notes], dtype=torch.long)
        octs = torch.tensor([n[2] for n in notes], dtype=torch.long)
        durs = torch.tensor([n[3] for n in notes], dtype=torch.long)
        meas = torch.tensor([n[4] for n in notes], dtype=torch.long)
        ints = torch.tensor(interval_ids(notes), dtype=torch.long)

        return {
            "pcs": pcs,
            "accs": accs,
            "octs": octs,
            "durs": durs,
            "meas": meas,
            "ints": ints,
            "label": torch.tensor(label, dtype=torch.long),
            "length": len(notes),
            "f_name": piece["f_name"]
        }


def collate_fn(batch):
    # sort the batch w.r.t. piece lengths, desc
    batch.sort(key=lambda x: x["length"], reverse=True)

    pcs = [b["pcs"] for b in batch]
    accs = [b["accs"] for b in batch]
    octs = [b["octs"] for b in batch]
    durs = [b["durs"] for b in batch]
    meas = [b["meas"] for b in batch]
    ints = [b["ints"] for b in batch]

    labels = torch.stack([b["label"] for b in batch])
    lengths = torch.tensor([b["length"] for b in batch])
    f_names = np.array([b["f_name"] for b in batch])

    pcs_pad = pad_sequence(pcs, batch_first=True, padding_value=PAD_ID)
    accs_pad = pad_sequence(accs, batch_first=True, padding_value=PAD_ID)
    octs_pad = pad_sequence(octs, batch_first=True, padding_value=PAD_ID)
    durs_pad = pad_sequence(durs, batch_first=True, padding_value=PAD_ID)
    meas_pad = pad_sequence(meas, batch_first=True, padding_value=PAD_ID)
    ints_pad = pad_sequence(ints, batch_first=True, padding_value=PAD_ID)

    # Mask: 1 = real token, 0 = padding (fake token)
    mask = (pcs_pad != PAD_ID).long()

    return {
        "pcs": pcs_pad,
        "accs": accs_pad,
        "octs": octs_pad,
        "durs": durs_pad,
        "meas": meas_pad,
        "ints": ints_pad,
        "mask": mask,
        "lengths": lengths,
        "labels": labels,
        "f_names": f_names
    }


def make_batch(pieces, labels):
    """
    Builds one batch from a list of pieces (no perturbation).
    """
    ds = MakamDataset(pieces, labels)
    items = [ds[i] for i in range(len(ds))]
    return collate_fn(items)


class BucketBatchSampler(Sampler):
    """
    Puts pieces of similar length into the same mini-batch, which reduces
    the padding and the number of recurrent steps per epoch.

    The pieces are shuffled, cut into pools of `pool_factor` batches,
    sorted by length inside each pool and split into batches. Then the
    order of the batches is shuffled.

    It only changes which pieces share a batch.
    """

    def __init__(self, lengths, batch_size, seed=0, pool_factor=20, shuffle=True):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_factor
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        if self.shuffle:
            indices = self.rng.permutation(len(self.lengths))
        else:
            # longest pieces first
            indices = np.argsort(-self.lengths)

        batches = []
        for pool_start in range(0, len(indices), self.pool_size):
            pool = indices[pool_start:pool_start + self.pool_size]

            # sort the pool w.r.t. piece lengths, desc
            order = np.argsort(-self.lengths[pool], kind="stable")
            pool = pool[order]

            for batch_start in range(0, len(pool), self.batch_size):
                batch = pool[batch_start:batch_start + self.batch_size]
                batches.append(batch.tolist())

        if self.shuffle:
            batch_order = self.rng.permutation(len(batches))
            batches = [batches[i] for i in batch_order]

        return iter(batches)

    def __len__(self):
        return int(np.ceil(len(self.lengths) / self.batch_size))


def build_dataloaders(
    vocab_file=VOCAB_FILE,
    dataset_file=DATASET_FILE,
    batch_size=BATCH_SIZE,
    seed=0,
    split_dir=SPLIT_DIR,
    train_shuffle=False,
    train_transpose_aug=0,
    test_shuffle=False,
    test_transpose=0
):
    """
    returns train_loader, class_weights, val_loader, test_loader

    seed: selects the stratified split (and seeds the batch order)

    Perturbations of the training pieces:
    train_shuffle:       shuffle the note order of every training piece
    train_transpose_aug: random transposition in [-k, +k] kommas

    Perturbations of the test pieces:
    test_shuffle:        shuffle the note order of every test piece
    test_transpose:      transpose every test piece by this many kommas

    The validation pieces are never perturbed.
    """
    pieces, labels, _ = load_pieces(vocab_file, dataset_file)
    train_idx, val_idx, test_idx = get_split(pieces, labels, seed, split_dir)

    train_pieces = [pieces[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_pieces = [pieces[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_pieces = [pieces[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    ds_train = MakamDataset(
        train_pieces,
        train_labels,
        shuffle_order=train_shuffle,
        transpose_aug=train_transpose_aug,
        seed=seed + 1000
    )
    ds_val = MakamDataset(val_pieces, val_labels)
    ds_test = MakamDataset(
        test_pieces,
        test_labels,
        shuffle_order=test_shuffle,
        transpose=test_transpose,
        seed=seed
    )

    class_weights = compute_class_weights(train_labels)

    train_lengths = [len(piece["notes"]) for piece in train_pieces]
    val_lengths = [len(piece["notes"]) for piece in val_pieces]
    test_lengths = [len(piece["notes"]) for piece in test_pieces]

    train_loader = DataLoader(
        ds_train,
        batch_sampler=BucketBatchSampler(
            train_lengths, batch_size, seed=seed, shuffle=True),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        ds_val,
        batch_sampler=BucketBatchSampler(
            val_lengths, batch_size, shuffle=False),
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        ds_test,
        batch_sampler=BucketBatchSampler(
            test_lengths, batch_size, shuffle=False),
        collate_fn=collate_fn
    )

    return train_loader, class_weights, val_loader, test_loader


def main():
    train_loader, class_weights, val_loader, test_loader = build_dataloaders(
        batch_size=4)
    print(len(train_loader))
    print(class_weights)
    print(len(val_loader))
    print(len(test_loader))

    loaders = [train_loader, val_loader, test_loader]
    for loader in loaders:
        batch = next(iter(loader))
        print(batch["pcs"].shape,
              batch["accs"].shape,
              batch["octs"].shape,
              batch["durs"].shape,
              batch["meas"].shape,
              batch["ints"].shape,
              batch["mask"].shape,
              batch["lengths"].shape,
              batch["labels"],
              batch["f_names"]
              )
        print("-" * 50)


if __name__ == "__main__":
    main()
