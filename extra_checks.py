import os
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter, defaultdict

from constants import *
from disk_utils import load_json
from dataset import load_pieces, read_split_files, komma_index

# Smaller checks that are reported in the paper and are computed from the
# stored result files (no model is needed).

EARLIER_TRAININGS_DIR = os.path.join(RESULTS_DIR, "earlier_trainings")


# ---------------------------------------------------------------------------
# Pieces with the same opening
# ---------------------------------------------------------------------------
def find_same_opening_groups(pieces, num_pitched_notes=30):
    """
    Groups of pieces whose first num_pitched_notes pitched notes (rests left
    out) have the same pitches. Most of them are an instrumental interlude
    (aranagme) that SymbTr holds as a file of its own and also at the
    beginning of a song.

    returns List[List[int]]: piece indices of every group with 2+ pieces
    """
    pieces_by_opening = defaultdict(list)

    for i, piece in enumerate(pieces):
        pitches = []
        for note in piece["notes"]:
            if note[0] == REST_ID:
                continue
            pitches.append(komma_index(note[0], note[1], note[2]))

        opening = tuple(pitches[:num_pitched_notes])
        pieces_by_opening[opening].append(i)

    groups = []
    for piece_indices in pieces_by_opening.values():
        if len(piece_indices) > 1:
            groups.append(piece_indices)

    return groups


def near_duplicate_check(
    tags,
    seeds=SEEDS,
    num_pitched_notes=30,
    results_dir=RESULTS_DIR,
    split_dir=SPLIT_DIR
):
    """
    How many test classifications concern a piece that has a piece with the
    same opening in the training subset of the same split, and what is the
    accuracy without them?

    returns a DataFrame with one row per tag
    """
    pieces, _, _ = load_pieces()
    groups = find_same_opening_groups(pieces, num_pitched_notes)

    rows = []
    for tag in tags:
        num_affected = 0
        num_affected_correct = 0
        num_other = 0
        num_other_correct = 0

        for seed in seeds:
            train_idx, _, _ = read_split_files(pieces, seed, split_dir)
            train_idx = set(train_idx)

            # pieces that have a group member in the training subset
            affected_f_names = set()
            for group in groups:
                for i in group:
                    for j in group:
                        if j != i and j in train_idx:
                            affected_f_names.add(pieces[i]["f_name"])

            result = load_json(
                os.path.join(results_dir, tag, f"seed{seed}.json"))

            for f_name, label, pred in zip(
                    result["test_files"],
                    result["test_labels"],
                    result["test_preds"]):
                if f_name in affected_f_names:
                    num_affected += 1
                    num_affected_correct += int(label == pred)
                else:
                    num_other += 1
                    num_other_correct += int(label == pred)

        num_all = num_affected + num_other
        num_correct = num_affected_correct + num_other_correct

        rows.append({
            "tag": tag,
            "n_groups": len(groups),
            "n_affected": num_affected,
            "accuracy_all": 100 * num_correct / num_all,
            "accuracy_without_affected": 100 * num_other_correct / num_other
        })

    return pd.DataFrame(rows)


def count_test_occurrences(seeds=SEEDS, split_dir=SPLIT_DIR):
    """
    In how many splits is a piece in the test subset?
    returns {number of splits: number of pieces}
    """
    pieces, _, _ = load_pieces()

    num_splits_by_piece = Counter()
    for seed in seeds:
        _, _, test_idx = read_split_files(pieces, seed, split_dir)
        for i in test_idx:
            num_splits_by_piece[i] += 1

    return dict(sorted(Counter(num_splits_by_piece.values()).items()))


# ---------------------------------------------------------------------------
# Predictions for transposed test pieces
# ---------------------------------------------------------------------------
def transposition_predictions(
    tag,
    kommas,
    seeds=SEEDS,
    results_dir=RESULTS_DIR
):
    """
    Predictions of a model for the test pieces transposed by `kommas`,
    pooled over the splits.

    returns a DataFrame: rows = true makam, columns = predicted makam, counts
    """
    labels = []
    preds = []

    for seed in seeds:
        result = load_json(os.path.join(
            results_dir, tag, f"seed{seed}__transpose{kommas:+d}.json"))
        labels.extend(result["test_labels"])
        preds.extend(result["test_preds"])

    table = pd.crosstab(
        pd.Series([MAKAM_NAMES[i] for i in labels], name="true"),
        pd.Series([MAKAM_NAMES[i] for i in preds], name="predicted")
    )

    return table


def prediction_shares(prediction_table):
    """
    Share of all predictions that every makam receives, largest first.
    A few makams with almost all predictions = the model has collapsed.
    """
    counts = prediction_table.sum(axis=0).sort_values(ascending=False)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Pitch enrichment in earlier trainings
# ---------------------------------------------------------------------------
def compare_with_earlier_trainings(enrichment, min_occ_share=0.05):
    """
    Are the pitch findings the same in the two earlier trainings of the
    same model (other software environments)?

    For the pitches with at least min_occ_share of the notes of their makam:
    Spearman rank correlation of the enrichment values of the present
    training with those of each earlier training.

    enrichment: the pitch enrichment table of the present training
    returns a DataFrame with one row per earlier training
    """
    current = enrichment.set_index(["makam", "pitch"])
    current = current[current["occ_share"] >= min_occ_share]

    rows = []
    for name in ["prelim", "first_deterministic"]:
        earlier = pd.read_csv(os.path.join(
            EARLIER_TRAININGS_DIR, f"{name}_pitch_enrichment.csv"))
        earlier = earlier.set_index(["makam", "pitch"])

        both = current.join(
            earlier[["enrichment", "attn_share"]],
            rsuffix="_earlier",
            how="inner"
        )

        report = load_json(os.path.join(
            EARLIER_TRAININGS_DIR, f"{name}_report.json"))
        duration = report["duration_all"]

        rows.append({
            "training": name,
            "n_pitches": len(both),
            "spearman_enrichment": stats.spearmanr(
                both["enrichment"], both["enrichment_earlier"]).correlation,
            "max_attn_share_difference": float(
                (both["attn_share"] - both["attn_share_earlier"]).abs().max()),
            "duration_kendall_W_five_bins": duration["kendall_W"],
            "duration_friedman_p_five_bins": duration["friedman_p"]
        })

    return pd.DataFrame(rows)
