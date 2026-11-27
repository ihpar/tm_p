import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

from constants import *
from experiments import load_trained_model
from utils import seed_everything, move_batch_to_device
from disk_utils import save_json
from dataset import (
    load_pieces,
    read_split_files,
    MakamDataset,
    collate_fn
)

# Deletion orders of the faithfulness test
#   attn_top     notes with the highest attention first
#   attn_bottom  notes with the lowest attention first
#   grad_top     notes with the highest gradient x input saliency first
#   random       random notes (averaged over several draws)
DELETION_ORDERS = ["attn_top", "attn_bottom", "grad_top", "random"]


@torch.no_grad()
def predict_pieces(model, pieces, labels, device, batch_size=32):
    """
    returns {f_name: predicted label} for a list of pieces
    """
    ds = MakamDataset(pieces, labels)
    preds_by_f_name = {}

    for batch_start in range(0, len(ds), batch_size):
        batch_end = min(batch_start + batch_size, len(ds))
        items = [ds[i] for i in range(batch_start, batch_end)]
        batch = collate_fn(items)

        logits, _ = model(move_batch_to_device(batch, device))
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        for f_name, pred in zip(batch["f_names"].tolist(), preds):
            preds_by_f_name[f_name] = pred

    return preds_by_f_name


def compute_attention_and_saliency(model, pieces, labels, device):
    """
    For every piece: the predicted label, the attention weight of every
    note and the saliency of every note.

    The saliency of a note is |gradient x input|: the gradient of the
    predicted class score w.r.t. the embedded note, multiplied by the
    embedded note and summed over the embedding dimensions.

    returns {f_name: {"pred": int, "attn": array (T,), "sal": array (T,)}}
    """
    ds = MakamDataset(pieces, labels)
    info = {}

    model.keep_input_grad = True
    # cuDNN can only run the backward pass of an LSTM in training mode.
    # The LSTM has a single layer and therefore no dropout, so switching
    # only this module to training mode changes nothing numerically.
    model.bilstm.train()

    for i in range(len(ds)):
        batch = collate_fn([ds[i]])
        f_name = batch["f_names"][0]

        model.zero_grad()
        logits, attn_weights = model(move_batch_to_device(batch, device))
        pred = torch.argmax(logits, dim=1)

        # gradient of the predicted class score w.r.t. the embedded input
        logits[0, pred].sum().backward()
        x = model.last_input
        saliency = (x.grad * x).sum(-1).abs()[0]

        info[f_name] = {
            "pred": int(pred.item()),
            "attn": attn_weights[0].detach().cpu().numpy(),
            "sal": saliency.detach().cpu().numpy()
        }

    model.keep_input_grad = False
    model.bilstm.eval()
    model.zero_grad()

    return info


def delete_notes(piece, order, fraction, rng=None):
    """
    Returns a copy of the piece without `fraction` of its notes.

    order: note indices, the notes to delete first come first.
           None deletes random notes, drawn with rng.

    At least one note is kept.
    """
    T = len(piece["notes"])
    num_delete = int(round(fraction * T))
    num_delete = min(num_delete, T - 1)

    if order is None:
        deleted = rng.choice(T, size=num_delete, replace=False).tolist()
    else:
        deleted = order[:num_delete].tolist()
    deleted = set(deleted)

    kept_notes = []
    for t, note in enumerate(piece["notes"]):
        if t not in deleted:
            kept_notes.append(note)

    return {"f_name": piece["f_name"], "notes": kept_notes}


def run_deletion_test(
    model,
    pieces,
    labels,
    deletion_orders,
    order_name,
    fraction,
    num_draws,
    rng,
    base_preds,
    device
):
    """
    Deletes `fraction` of the notes of every piece in the given order and
    classifies the shortened pieces.

    returns accuracy, flip_rate (averaged over num_draws)
    flip_rate: share of pieces whose prediction changed
    """
    y = np.array(labels)
    accuracies = []
    flip_rates = []

    for _ in range(num_draws):
        shortened_pieces = []
        for piece in pieces:
            if order_name == "random":
                order = None
            else:
                order = deletion_orders[order_name][piece["f_name"]]
            shortened_pieces.append(delete_notes(piece, order, fraction, rng))

        preds_by_f_name = predict_pieces(
            model, shortened_pieces, labels, device)
        preds = np.array([preds_by_f_name[p["f_name"]] for p in pieces])

        accuracies.append((preds == y).mean())
        flip_rates.append((preds != base_preds).mean())

    return float(np.mean(accuracies)), float(np.mean(flip_rates))


def run_deletion_tests(
    tag,
    device,
    seeds=SEEDS,
    fractions=DELETION_FRACTIONS,
    num_random_draws=NUM_RANDOM_DRAWS,
    results_dir=RESULTS_DIR,
    split_dir=SPLIT_DIR
):
    """
    Faithfulness test of the attention weights of the trained models
    <results_dir>/<tag>/seed<seed>.pt

    If attention points at the notes a prediction depends on, deleting the
    most attended notes should hurt the accuracy much more than deleting
    random or least attended notes.

    Writes to <results_dir>/faithfulness/
        <tag>_seed<seed>.json   results of one seed
        <tag>_curves.csv        accuracy per seed, order and fraction
        <tag>_summary.csv       mean and SD over the seeds

    returns curves, summary (DataFrames)
    """
    pieces_all, labels_all, _ = load_pieces()
    out_dir = os.path.join(results_dir, "faithfulness")
    os.makedirs(out_dir, exist_ok=True)

    rows = []

    for seed in seeds:
        seed_everything(seed)

        model, _ = load_trained_model(tag, seed, device, results_dir)

        _, _, test_idx = read_split_files(pieces_all, seed, split_dir)
        pieces = [pieces_all[i] for i in test_idx]
        labels = [labels_all[i] for i in test_idx]
        y = np.array(labels)

        info = compute_attention_and_saliency(model, pieces, labels, device)

        base_preds = np.array([info[p["f_name"]]["pred"] for p in pieces])
        base_accuracy = (base_preds == y).mean()

        # rank correlation between attention and saliency, per piece
        correlations = []
        for piece in pieces:
            if len(piece["notes"]) > 2:
                piece_info = info[piece["f_name"]]
                correlation = spearmanr(
                    piece_info["attn"], piece_info["sal"]).correlation
                correlations.append(correlation)

        correlation_mean = float(np.nanmean(correlations))
        correlation_median = float(np.nanmedian(correlations))

        print(
            f"[{tag} s{seed}] "
            f"Base Acc: {base_accuracy:.4f} | "
            f"Spearman(attention, saliency): mean {correlation_mean:.3f}, "
            f"median {correlation_median:.3f}"
        )

        # note indices of every piece, the notes to delete first come first
        deletion_orders = {
            "attn_top": {},
            "attn_bottom": {},
            "grad_top": {}
        }
        for piece in pieces:
            f_name = piece["f_name"]
            deletion_orders["attn_top"][f_name] = np.argsort(-info[f_name]["attn"])
            deletion_orders["attn_bottom"][f_name] = np.argsort(info[f_name]["attn"])
            deletion_orders["grad_top"][f_name] = np.argsort(-info[f_name]["sal"])

        rng = np.random.RandomState(seed)

        result = {
            "tag": tag,
            "seed": seed,
            "base_accuracy": base_accuracy,
            "spearman_attn_saliency_mean": correlation_mean,
            "spearman_attn_saliency_median": correlation_median,
            "spearman_per_piece": [float(c) for c in correlations],
            "curves": {}
        }

        for fraction in fractions:
            for order_name in DELETION_ORDERS:
                if order_name == "random":
                    num_draws = num_random_draws
                else:
                    num_draws = 1

                accuracy, flip_rate = run_deletion_test(
                    model,
                    pieces,
                    labels,
                    deletion_orders,
                    order_name,
                    fraction,
                    num_draws,
                    rng,
                    base_preds,
                    device
                )

                result["curves"][f"{order_name}@{fraction}"] = {
                    "accuracy": accuracy,
                    "flip_rate": flip_rate
                }
                rows.append({
                    "tag": tag,
                    "seed": seed,
                    "order": order_name,
                    "frac": fraction,
                    "accuracy": accuracy,
                    "flip_rate": flip_rate,
                    "base_accuracy": base_accuracy
                })

                print(
                    f"    delete {int(fraction * 100):2d}% {order_name:12s} "
                    f"Acc: {accuracy:.4f} | Flip rate: {flip_rate:.3f}"
                )

        save_json(result, os.path.join(out_dir, f"{tag}_seed{seed}.json"))

    curves = pd.DataFrame(rows)
    curves.to_csv(os.path.join(out_dir, f"{tag}_curves.csv"), index=False)

    summary = curves.groupby(["order", "frac"]).agg(
        acc_mean=("accuracy", "mean"),
        acc_sd=("accuracy", "std"),
        flip_mean=("flip_rate", "mean"),
        flip_sd=("flip_rate", "std"),
        n=("seed", "count")
    ).reset_index()
    summary.to_csv(os.path.join(out_dir, f"{tag}_summary.csv"), index=False)

    return curves, summary


def pitch_matched_order(piece, attention_order, fraction, rng):
    """
    Control for the deletion of the most attended notes.

    The most attended notes of a piece are mostly notes of a few pitches,
    so deleting them also removes these pitches from the piece. This
    control deletes the same number of notes of the same pitches, but at
    random positions: for every pitch, as many notes are deleted as the
    attention order would delete, drawn at random from all notes of that
    pitch in the piece.

    A pitch is the note letter, the accidental and the octave of a note.

    returns the note indices to delete (array)
    """
    T = len(piece["notes"])
    num_delete = int(round(fraction * T))
    num_delete = min(num_delete, T - 1)

    # how many notes of every pitch the attention order deletes
    num_by_pitch = {}
    for t in attention_order[:num_delete].tolist():
        pitch = piece["notes"][t][:3]
        num_by_pitch[pitch] = num_by_pitch.get(pitch, 0) + 1

    # positions of all notes of every pitch
    positions_by_pitch = {}
    for t, note in enumerate(piece["notes"]):
        positions_by_pitch.setdefault(note[:3], []).append(t)

    deleted = []
    for pitch in sorted(num_by_pitch):
        chosen = rng.choice(
            positions_by_pitch[pitch],
            size=num_by_pitch[pitch],
            replace=False
        )
        deleted.extend(chosen.tolist())

    return np.array(deleted, dtype=int)


def run_pitch_matched_deletion_tests(
    tag,
    device,
    seeds=SEEDS,
    fractions=DELETION_FRACTIONS,
    num_random_draws=NUM_RANDOM_DRAWS,
    results_dir=RESULTS_DIR,
    split_dir=SPLIT_DIR
):
    """
    Deletion of the most attended notes (attn_top) against the deletion of
    the same number of notes of the same pitches at random positions
    (pitch_matched, averaged over num_random_draws draws).

    If the two hurt the accuracy equally, the attention weights point at
    pitches. If attn_top hurts more, they also point at the particular
    notes of these pitches that the prediction depends on.

    Writes to <results_dir>/faithfulness/
        <tag>_pitch_matched_curves.csv    accuracy per seed, order, fraction
        <tag>_pitch_matched_summary.csv   mean and SD over the seeds

    returns curves, summary (DataFrames)
    """
    pieces_all, labels_all, _ = load_pieces()
    out_dir = os.path.join(results_dir, "faithfulness")
    os.makedirs(out_dir, exist_ok=True)

    rows = []

    for seed in seeds:
        seed_everything(seed)

        model, _ = load_trained_model(tag, seed, device, results_dir)

        _, _, test_idx = read_split_files(pieces_all, seed, split_dir)
        pieces = [pieces_all[i] for i in test_idx]
        labels = [labels_all[i] for i in test_idx]
        y = np.array(labels)

        info = compute_attention_and_saliency(model, pieces, labels, device)

        base_preds = np.array([info[p["f_name"]]["pred"] for p in pieces])
        base_accuracy = (base_preds == y).mean()
        print(f"[{tag} s{seed}] Base Acc: {base_accuracy:.4f}")

        attention_orders = {}
        for piece in pieces:
            f_name = piece["f_name"]
            attention_orders[f_name] = np.argsort(-info[f_name]["attn"])

        rng = np.random.RandomState(seed)

        for fraction in fractions:
            for order_name in ["attn_top", "pitch_matched"]:
                if order_name == "pitch_matched":
                    num_draws = num_random_draws
                else:
                    num_draws = 1

                accuracies = []
                flip_rates = []

                for _ in range(num_draws):
                    shortened_pieces = []
                    for piece in pieces:
                        order = attention_orders[piece["f_name"]]
                        if order_name == "pitch_matched":
                            order = pitch_matched_order(
                                piece, order, fraction, rng)
                        shortened_pieces.append(
                            delete_notes(piece, order, fraction))

                    preds_by_f_name = predict_pieces(
                        model, shortened_pieces, labels, device)
                    preds = np.array(
                        [preds_by_f_name[p["f_name"]] for p in pieces])

                    accuracies.append((preds == y).mean())
                    flip_rates.append((preds != base_preds).mean())

                accuracy = float(np.mean(accuracies))
                flip_rate = float(np.mean(flip_rates))

                rows.append({
                    "tag": tag,
                    "seed": seed,
                    "order": order_name,
                    "frac": fraction,
                    "accuracy": accuracy,
                    "flip_rate": flip_rate,
                    "base_accuracy": base_accuracy
                })

                print(
                    f"    delete {int(fraction * 100):2d}% {order_name:13s} "
                    f"Acc: {accuracy:.4f} | Flip rate: {flip_rate:.3f}"
                )

    curves = pd.DataFrame(rows)
    curves.to_csv(
        os.path.join(out_dir, f"{tag}_pitch_matched_curves.csv"), index=False)

    summary = curves.groupby(["order", "frac"]).agg(
        acc_mean=("accuracy", "mean"),
        acc_sd=("accuracy", "std"),
        flip_mean=("flip_rate", "mean"),
        flip_sd=("flip_rate", "std"),
        n=("seed", "count")
    ).reset_index()
    summary.to_csv(
        os.path.join(out_dir, f"{tag}_pitch_matched_summary.csv"), index=False)

    return curves, summary


def load_deletion_tests(tag, results_dir=RESULTS_DIR):
    """
    Loads the tables written by run_deletion_tests.
    returns curves, summary (DataFrames)
    """
    out_dir = os.path.join(results_dir, "faithfulness")
    curves = pd.read_csv(os.path.join(out_dir, f"{tag}_curves.csv"))
    summary = pd.read_csv(os.path.join(out_dir, f"{tag}_summary.csv"))

    return curves, summary


def load_pitch_matched_deletion_tests(tag, results_dir=RESULTS_DIR):
    """
    Loads the tables written by run_pitch_matched_deletion_tests.
    returns curves, summary (DataFrames)
    """
    out_dir = os.path.join(results_dir, "faithfulness")
    curves = pd.read_csv(
        os.path.join(out_dir, f"{tag}_pitch_matched_curves.csv"))
    summary = pd.read_csv(
        os.path.join(out_dir, f"{tag}_pitch_matched_summary.csv"))

    return curves, summary


def compare_with_pitch_matched(pitch_matched_curves):
    """
    Deletion of the most attended notes against the pitch-matched control,
    for every fraction of deleted notes:

    drop_attn_top:       accuracy lost by deleting the most attended notes
    drop_pitch_matched:  accuracy lost by the control
    share_reproduced:    drop_pitch_matched / drop_attn_top
    difference:          accuracy of the control minus accuracy after
                         deleting the most attended notes (mean over seeds)
    p_paired_t:          paired t-test of the two over the seeds
    n_seeds_attn_lower:  seeds in which deleting the most attended notes
                         hurts more than the control

    All accuracies are in percentage points.
    """
    table = pitch_matched_curves.pivot_table(
        index=["frac", "seed"], columns="order", values="accuracy")
    base = pitch_matched_curves.groupby(["frac", "seed"])["base_accuracy"].first()

    rows = []
    for fraction in sorted(pitch_matched_curves["frac"].unique()):
        attn_top = 100 * table.loc[fraction, "attn_top"]
        pitch_matched = 100 * table.loc[fraction, "pitch_matched"]
        base_accuracy = 100 * base.loc[fraction]

        drop_attn_top = (base_accuracy - attn_top).mean()
        drop_pitch_matched = (base_accuracy - pitch_matched).mean()
        difference = pitch_matched - attn_top

        rows.append({
            "frac": fraction,
            "acc_attn_top": attn_top.mean(),
            "acc_pitch_matched": pitch_matched.mean(),
            "drop_attn_top": drop_attn_top,
            "drop_pitch_matched": drop_pitch_matched,
            "share_reproduced": drop_pitch_matched / drop_attn_top,
            "difference": difference.mean(),
            "p_paired_t": float(
                ttest_rel(pitch_matched.values, attn_top.values).pvalue),
            "n_seeds_attn_lower": int((difference > 0).sum())
        })

    return pd.DataFrame(rows)


def pitch_matched_overlap(note_table, fractions=DELETION_FRACTIONS):
    """
    How similar are the two sets of deleted notes?

    The control draws, for every pitch, as many notes as the attention order
    deletes, from all notes of that pitch in the piece. The most attended
    notes are among them, so part of the control's notes are the most
    attended notes themselves. When a large fraction is deleted this part is
    large, and the two deletions must give similar results.

    For a pitch with n notes in the piece of which k are among the most
    attended notes, the control is expected to hit k * k / n of them.

    note_table: note table with the rests (AttentionInspector.load_note_table)

    returns a DataFrame with one row per fraction:
    expected_overlap: expected share of the control's notes that are among
                      the most attended notes (mean over the units)
    forced_identical: share of the deleted notes that belong to a pitch of
                      which every note is deleted (control = attention order)
    """
    pieces, _, _ = load_pieces()
    notes_by_f_name = {}
    for piece in pieces:
        notes_by_f_name[piece["f_name"]] = piece["notes"]

    rows = []
    for fraction in fractions:
        overlaps = []
        forced = []

        for (seed, f_name), unit_notes in note_table.groupby(["seed", "f_name"]):
            unit_notes = unit_notes.sort_values("t")
            alpha = unit_notes["alpha"].values
            notes = notes_by_f_name[f_name]

            T = len(notes)
            num_delete = int(round(fraction * T))
            num_delete = min(num_delete, T - 1)
            if num_delete == 0:
                continue

            attention_order = np.argsort(-alpha)

            num_in_piece = {}
            for note in notes:
                pitch = note[:3]
                num_in_piece[pitch] = num_in_piece.get(pitch, 0) + 1

            num_deleted = {}
            for t in attention_order[:num_delete].tolist():
                pitch = notes[t][:3]
                num_deleted[pitch] = num_deleted.get(pitch, 0) + 1

            expected_hits = 0.0
            num_forced = 0
            for pitch, k in num_deleted.items():
                n = num_in_piece[pitch]
                expected_hits += k * k / n
                if k == n:
                    num_forced += k

            overlaps.append(expected_hits / num_delete)
            forced.append(num_forced / num_delete)

        rows.append({
            "frac": fraction,
            "expected_overlap": float(np.mean(overlaps)),
            "forced_identical": float(np.mean(forced))
        })

    return pd.DataFrame(rows)
