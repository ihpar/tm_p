import os
import json
import torch
import itertools
import numpy as np
import pandas as pd
from scipy import stats

from constants import *
from experiments import load_trained_model
from utils import seed_everything, move_batch_to_device
from pitch_names import (
    get_sharp_name,
    get_symbtr_name,
    build_symbtr_spelling_map
)
from stat_utils import (
    holm_correction,
    cliffs_delta,
    paired_cliffs_delta,
    friedman_test
)
from dataset import (
    load_pieces,
    read_split_files,
    get_duration_values,
    MakamDataset,
    collate_fn,
    make_batch,
    komma_index
)

# The unit of analysis is one test classification: a piece within a split.
# Notes of the same piece are not independent, so every statistic is first
# computed per unit and the units are compared.
UNIT_KEY = ["seed", "f_name"]


class AttentionInspector:
    def __init__(self, device, results_dir=RESULTS_DIR, split_dir=SPLIT_DIR):
        self.device = device
        self.results_dir = results_dir
        self.split_dir = split_dir
        self.out_dir = os.path.join(results_dir, "attention")

        self.pieces, self.labels, vocab = load_pieces()
        self.dur_values = get_duration_values(vocab)
        self.spelling_map = build_symbtr_spelling_map()

    # -----------------------------------------------------------------------
    # Note table: one row per note of every test piece
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def collect_test_notes(self, model, pieces, labels, batch_size=32):
        """
        Runs the model on the pieces and returns one row per note:

        f_name, makam, correct  the piece, its makam, was it classified correctly
        T, t                    length of the piece, index of the note
        alpha                   attention weight of the note
        norm_attn               T * alpha (1 = average attention in the piece)
        high                    alpha is in the top 5% of its piece
        pitch, pitch_aeu        pitch name as stored / as written in SymbTr
        is_rest, komma_class    rest flag, pitch class in kommas (0-52)
        dur, dur_bin            relative duration and its bin
        meas_pos                position in the measure (start, middle, end)
        """
        ds = MakamDataset(pieces, labels)
        rows = []

        for batch_start in range(0, len(ds), batch_size):
            batch_end = min(batch_start + batch_size, len(ds))
            items = [ds[i] for i in range(batch_start, batch_end)]
            batch = collate_fn(items)

            batch_on_device = move_batch_to_device(batch, self.device)
            logits, attn_weights = model(batch_on_device)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            attn_weights = attn_weights.cpu().numpy()

            for b, f_name in enumerate(batch["f_names"].tolist()):
                T = int(batch["lengths"][b])
                label = int(batch["labels"][b])

                attn = attn_weights[b, :T]
                high_threshold = np.percentile(attn, HIGH_ATTENTION_PERCENTILE)

                pcs = batch["pcs"][b, :T].numpy()
                accs = batch["accs"][b, :T].numpy()
                octs = batch["octs"][b, :T].numpy()
                durs = batch["durs"][b, :T].numpy()
                meas = batch["meas"][b, :T].numpy()

                for t in range(T):
                    pc = int(pcs[t])
                    acc = int(accs[t])
                    octave = int(octs[t])

                    komma = komma_index(pc, acc, octave)
                    if komma is None:
                        komma_class = 0
                    else:
                        komma_class = komma % KOMMAS_PER_OCTAVE

                    rows.append({
                        "f_name": f_name,
                        "makam": MAKAM_NAMES[label],
                        "correct": bool(preds[b] == label),
                        "T": T,
                        "t": t,
                        "alpha": float(attn[t]),
                        "norm_attn": float(attn[t] * T),
                        "high": bool(attn[t] >= high_threshold),
                        "pitch": get_sharp_name(pc, acc),
                        "pitch_aeu": get_symbtr_name(pc, acc, self.spelling_map),
                        "is_rest": pc == REST_ID,
                        "komma_class": komma_class,
                        "dur": self.dur_values[int(durs[t])],
                        "meas_pos": MEAS_ID_TO_NAME.get(int(meas[t]), "?")
                    })

        note_table = pd.DataFrame(rows)
        note_table["dur_bin"] = pd.cut(
            note_table["dur"],
            bins=DURATION_BIN_EDGES,
            labels=DURATION_BIN_LABELS,
            include_lowest=True
        )

        return note_table

    def collect_note_table(self, tag, seeds=SEEDS):
        """
        Collects the notes of the test pieces of every seed with the trained
        models of `tag` and writes them to
        <results_dir>/attention/<tag>_notes.csv.gz

        returns the note table
        """
        seed_tables = []

        for seed in seeds:
            seed_everything(seed)

            model, _ = load_trained_model(
                tag, seed, self.device, self.results_dir)

            _, _, test_idx = read_split_files(
                self.pieces, seed, self.split_dir)
            test_pieces = [self.pieces[i] for i in test_idx]
            test_labels = [self.labels[i] for i in test_idx]

            seed_table = self.collect_test_notes(model, test_pieces, test_labels)
            seed_table["seed"] = seed
            seed_tables.append(seed_table)

            print(
                f"[{tag} s{seed}] collected {len(seed_table)} notes "
                f"from {seed_table['f_name'].nunique()} pieces"
            )

        note_table = pd.concat(seed_tables, ignore_index=True)

        os.makedirs(self.out_dir, exist_ok=True)
        note_table.to_csv(self.get_note_table_path(tag), index=False)

        return note_table

    def get_note_table_path(self, tag):
        return os.path.join(self.out_dir, f"{tag}_notes.csv.gz")

    def load_note_table(self, tag):
        """
        Loads the note table written by collect_note_table.
        """
        note_table = pd.read_csv(self.get_note_table_path(tag))
        note_table["dur_bin"] = pd.Categorical(
            note_table["dur_bin"], categories=DURATION_BIN_LABELS)

        return note_table

    @torch.no_grad()
    def get_piece_attention(self, tag, seed, piece_name):
        """
        Attention weights of one test piece of the split of `seed`.

        piece_name: part of the file name of the piece
        returns piece, attn (array of length T), predicted makam
        """
        model, _ = load_trained_model(
            tag, seed, self.device, self.results_dir)

        _, _, test_idx = read_split_files(self.pieces, seed, self.split_dir)

        piece_idx = None
        for i in test_idx:
            if piece_name in self.pieces[i]["f_name"]:
                piece_idx = i
                break

        if piece_idx is None:
            raise ValueError(
                f"No test piece of seed {seed} matches '{piece_name}'")

        piece = self.pieces[piece_idx]
        label = self.labels[piece_idx]

        batch = make_batch([piece], [label])
        logits, attn_weights = model(move_batch_to_device(batch, self.device))

        attn = attn_weights[0].cpu().numpy()
        predicted_makam = MAKAM_NAMES[int(torch.argmax(logits))]

        return piece, attn, predicted_makam

    # -----------------------------------------------------------------------
    # Pitch enrichment
    # -----------------------------------------------------------------------
    def pitch_enrichment(self, note_table):
        """
        For every makam and pitch (rests left out):

        occ_share:   share of the pitch among the notes of the makam
        attn_share:  share of the attention that falls on the pitch
        enrichment:  attn_share / occ_share
                     (1 = attended in proportion to its frequency)
        attn_entropy_bits: entropy of the attention shares of the makam
                     (low = attention concentrated on few pitches)
        """
        pitched_notes = note_table[~note_table["is_rest"]]
        rows = []

        for makam, makam_notes in pitched_notes.groupby("makam"):
            by_pitch = makam_notes.groupby("pitch")

            occurrences = by_pitch.size()
            attention = by_pitch["alpha"].sum()
            mean_norm_attn = by_pitch["norm_attn"].mean()

            occ_share = occurrences / occurrences.sum()
            attn_share = attention / attention.sum()

            p = attn_share.values
            p = p[p > 0]
            entropy = float(-(p * np.log2(p)).sum())

            for pitch in occurrences.index:
                first_note = makam_notes.loc[makam_notes["pitch"] == pitch].iloc[0]

                rows.append({
                    "makam": makam,
                    "pitch": pitch,
                    "pitch_aeu": first_note["pitch_aeu"],
                    "komma_class": int(first_note["komma_class"]),
                    "n_occ": int(occurrences[pitch]),
                    "occ_share": occ_share[pitch],
                    "attn_share": attn_share[pitch],
                    "enrichment": attn_share[pitch] / occ_share[pitch],
                    "mean_norm_attn": mean_norm_attn[pitch],
                    "attn_entropy_bits": entropy,
                    "n_pitches": len(occurrences)
                })

        enrichment = pd.DataFrame(rows)
        enrichment = enrichment.sort_values(
            ["makam", "occ_share"], ascending=[True, False])

        return enrichment

    # -----------------------------------------------------------------------
    # Duration
    # -----------------------------------------------------------------------
    def unit_by_duration_bin(self, note_table, value="norm_attn"):
        """
        One row per unit (test classification), one column per duration bin.

        value="norm_attn":  median T * alpha of the unit's notes in the bin
        value="high_share": share of the bin's notes that are high-attention
                            notes (0.05 expected if there is no association)

        A cell is empty if the unit has no note in the bin.
        """
        grouped = note_table.groupby(UNIT_KEY + ["dur_bin"], observed=False)

        if value == "high_share":
            per_unit = grouped["high"].mean()
        else:
            per_unit = grouped[value].agg("median")

        return per_unit.unstack("dur_bin")

    def pairwise_duration_tests(self, unit_table):
        """
        Wilcoxon signed-rank test and paired Cliff's delta for every pair of
        duration bins, on the units that have notes in both bins.
        The p-values are Holm-corrected.
        """
        pairs = []
        p_values = []

        for bin_a, bin_b in itertools.combinations(DURATION_BIN_LABELS, 2):
            both = unit_table[[bin_a, bin_b]].dropna()
            if len(both) < 10:
                continue

            try:
                p_value = float(stats.wilcoxon(both[bin_a], both[bin_b]).pvalue)
            except ValueError:
                p_value = 1.0

            delta = paired_cliffs_delta(both[bin_a], both[bin_b])

            pairs.append({
                "bin_a": bin_a,
                "bin_b": bin_b,
                "n_pieces": int(len(both)),
                "median_a": float(both[bin_a].median()),
                "median_b": float(both[bin_b].median()),
                "paired_cliffs_delta": float(delta),
                "p_raw": p_value
            })
            p_values.append(p_value)

        if pairs:
            adjusted = holm_correction(p_values)
            for pair, p_holm in zip(pairs, adjusted):
                pair["p_holm"] = float(p_holm)

        return pairs

    def note_level_duration_statistics(self, note_table):
        """
        Statistics with the single note as the unit. Notes of one piece are
        not independent, so these are reported for reference only.
        """
        by_bin = note_table.groupby("dur_bin", observed=False)

        groups = []
        for _, bin_notes in by_bin:
            if len(bin_notes):
                groups.append(bin_notes["norm_attn"].values)

        kruskal_H, kruskal_p = stats.kruskal(*groups)
        num_notes = sum(len(group) for group in groups)

        n_by_bin = {}
        for dur_bin, count in note_table["dur_bin"].value_counts().items():
            n_by_bin[str(dur_bin)] = int(count)

        median_by_bin = {}
        for dur_bin, median in by_bin["norm_attn"].median().items():
            median_by_bin[str(dur_bin)] = float(median)

        very_short = note_table.loc[
            note_table["dur_bin"] == DURATION_BIN_LABELS[0], "norm_attn"]
        very_long = note_table.loc[
            note_table["dur_bin"] == DURATION_BIN_LABELS[4], "norm_attn"]

        statistics = {
            "kruskal_H": float(kruskal_H),
            "p": float(kruskal_p),
            "epsilon2": float(kruskal_H / (num_notes - 1)),
            "n_notes": int(num_notes),
            "n_by_bin": n_by_bin,
            "median_norm_attn_by_bin": median_by_bin,
            "cliffs_delta_veryshort_vs_verylong": float(
                cliffs_delta(very_short, very_long))
        }

        # 5 x 2 table: duration bin x (high attention or not)
        contingency = pd.crosstab(note_table["dur_bin"], note_table["high"])
        chi2, chi2_p, dof, _ = stats.chi2_contingency(contingency)

        num_cells = contingency.values.sum()
        cramers_V = np.sqrt(chi2 / (num_cells * (min(contingency.shape) - 1)))

        high_rate_by_bin = {}
        for dur_bin, rate in by_bin["high"].mean().items():
            high_rate_by_bin[str(dur_bin)] = float(rate)

        statistics["chi2_5x2"] = {
            "chi2": float(chi2),
            "dof": int(dof),
            "p": float(chi2_p),
            "cramers_V": float(cramers_V),
            "high_rate_by_bin": high_rate_by_bin
        }

        return statistics

    def duration_statistics(self, note_table, label):
        """
        Attention by relative duration, with the test classification as the
        unit of analysis.

        note_table: note table without the rests
        label:      description of the note table, kept in the output
        """
        statistics = {"label": label}

        unit_table = self.unit_by_duration_bin(note_table, "norm_attn")

        # --- Friedman test on the units that have notes in all five bins
        complete_units = unit_table.dropna()
        statistics["n_pieces_all_bins"] = int(len(complete_units))

        if len(complete_units) >= 5:
            chi2, p, kendall_w = friedman_test(complete_units)
            statistics["friedman_chi2"] = chi2
            statistics["friedman_p"] = p
            statistics["kendall_W"] = kendall_w

        median_complete = {}
        if len(complete_units):
            for dur_bin in complete_units.columns:
                median_complete[dur_bin] = float(
                    np.median(complete_units[dur_bin]))
        statistics["piece_median_by_bin_complete_cases"] = median_complete

        # --- per bin: median over all units that have notes in the bin
        median_by_bin = {}
        units_by_bin = {}
        for dur_bin in unit_table.columns:
            median_by_bin[dur_bin] = float(unit_table[dur_bin].dropna().median())
            units_by_bin[dur_bin] = int(unit_table[dur_bin].notna().sum())
        statistics["piece_median_by_bin"] = median_by_bin
        statistics["n_pieces_by_bin"] = units_by_bin

        # --- pairs of bins
        statistics["pairwise"] = self.pairwise_duration_tests(unit_table)

        # --- share of high-attention notes per bin, unit level
        high_share_table = self.unit_by_duration_bin(note_table, "high_share")
        high_share_table = high_share_table.dropna()

        if len(high_share_table) >= 5:
            chi2, p, kendall_w = friedman_test(high_share_table)
            statistics["high_share_friedman_chi2"] = chi2
            statistics["high_share_friedman_p"] = p
            statistics["high_share_kendall_W"] = kendall_w

            high_share_mean = {}
            for dur_bin in high_share_table.columns:
                high_share_mean[dur_bin] = float(high_share_table[dur_bin].mean())
            statistics["high_share_mean_by_bin"] = high_share_mean

        # --- note level (reference only)
        statistics["note_level"] = self.note_level_duration_statistics(note_table)

        return statistics

    # -----------------------------------------------------------------------
    # Rests and measure position
    # -----------------------------------------------------------------------
    def unit_by_measure_position(self, note_table, value="norm_attn"):
        """
        One row per unit, one column per measure position (end, middle, start)
        with the unit's mean of `value` at that position.
        """
        grouped = note_table.groupby(UNIT_KEY + ["meas_pos"])
        return grouped[value].mean().unstack()

    def rest_and_measure_statistics(self, note_table):
        """
        note_table: note table with the rests
        """
        statistics = {}

        rests = note_table[note_table["is_rest"]]
        pitched = note_table[~note_table["is_rest"]]

        statistics["rest_share_of_notes"] = float(note_table["is_rest"].mean())
        statistics["rest_share_of_attention"] = float(
            rests["alpha"].sum() / note_table["alpha"].sum())
        statistics["mean_norm_attn_rests"] = float(rests["norm_attn"].mean())
        statistics["mean_norm_attn_pitched"] = float(pitched["norm_attn"].mean())

        # mean T * alpha per unit and measure position
        attn_table = self.unit_by_measure_position(note_table, "norm_attn")

        position_mean = {}
        position_median = {}
        for position in attn_table.columns:
            position_mean[position] = float(attn_table[position].mean())
            position_median[position] = float(attn_table[position].median())
        statistics["measure_position_piece_mean"] = position_mean
        statistics["measure_position_piece_median"] = position_median

        # share of high-attention notes per unit and measure position
        high_table = self.unit_by_measure_position(note_table, "high")

        high_share_mean = {}
        for position in high_table.columns:
            high_share_mean[position] = float(high_table[position].mean())
        statistics["measure_position_high_share_mean"] = high_share_mean

        complete_high = high_table.dropna()
        if len(complete_high) >= 5 and complete_high.shape[1] == 3:
            chi2, p, kendall_w = friedman_test(complete_high)
            statistics["measure_position_high_share_friedman"] = {
                "chi2": chi2,
                "p": p,
                "kendall_W": kendall_w,
                "n_pieces": int(len(complete_high))
            }

        complete_attn = attn_table.dropna()
        if len(complete_attn) >= 5 and complete_attn.shape[1] == 3:
            chi2, p, kendall_w = friedman_test(complete_attn)
            statistics["measure_position_friedman"] = {
                "chi2": chi2,
                "p": p,
                "kendall_W": kendall_w,
                "n_pieces": int(len(complete_attn))
            }

        return statistics

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    def build_report(self, tag, note_table):
        """
        Computes every attention statistic of the paper and writes

        <results_dir>/attention/<tag>_pitch_enrichment.csv
        <results_dir>/attention/<tag>_report.json

        returns report (dict), enrichment (DataFrame)
        """
        os.makedirs(self.out_dir, exist_ok=True)

        enrichment = self.pitch_enrichment(note_table)
        enrichment.to_csv(
            os.path.join(self.out_dir, f"{tag}_pitch_enrichment.csv"),
            index=False
        )

        # rests have no pitch and a different role: they are left out of
        # the duration statistics
        pitched_notes = note_table[~note_table["is_rest"]]
        correct_pitched_notes = pitched_notes[pitched_notes["correct"]]

        by_makam = enrichment.groupby("makam")

        report = {
            "tag": tag,
            "seeds": sorted(note_table["seed"].unique().tolist()),
            "n_notes": int(len(note_table)),
            "n_test_classifications": int(note_table.groupby(UNIT_KEY).ngroups),
            "n_distinct_pieces": int(note_table["f_name"].nunique()),
            "duration_all": self.duration_statistics(
                pitched_notes, "all test pieces"),
            "duration_correct_only": self.duration_statistics(
                correct_pitched_notes, "correctly classified pieces"),
            "rest_and_measure": self.rest_and_measure_statistics(note_table),
            "attention_entropy_bits_by_makam":
                by_makam["attn_entropy_bits"].first().to_dict(),
            "n_pitches_by_makam": by_makam["n_pitches"].first().to_dict()
        }

        report_path = os.path.join(self.out_dir, f"{tag}_report.json")
        with open(report_path, "w") as out_file:
            json.dump(report, out_file, indent=1, default=str)

        return report, enrichment

    def print_report(self, report, enrichment):
        duration = report["duration_all"]

        print("=== Duration bins (unit level, rests excluded) ===")
        print("units by bin:", duration["n_pieces_by_bin"])
        print(
            f"units with all 5 bins: {duration['n_pieces_all_bins']} | "
            f"Friedman chi2: {duration.get('friedman_chi2', float('nan')):.2f} | "
            f"p: {duration.get('friedman_p', float('nan')):.2g} | "
            f"Kendall W: {duration.get('kendall_W', float('nan')):.3f}"
        )

        print("median of the per-unit median T*alpha, by bin:")
        for dur_bin, median in duration["piece_median_by_bin"].items():
            print(f"  {dur_bin:>20s}: {median:.3f}")

        print("pairs of bins:")
        for pair in duration["pairwise"]:
            print(
                f"  {pair['bin_a']:>20s} vs {pair['bin_b']:<20s} "
                f"n={pair['n_pieces']:3d} "
                f"delta={pair['paired_cliffs_delta']:+.3f} "
                f"p_holm={pair['p_holm']:.3g}"
            )

        note_level = duration["note_level"]
        chi2_5x2 = note_level["chi2_5x2"]
        print(
            f"note level (reference): "
            f"Kruskal-Wallis H={note_level['kruskal_H']:.1f} "
            f"eps2={note_level['epsilon2']:.4f} | "
            f"chi2(5x2)={chi2_5x2['chi2']:.1f} dof={chi2_5x2['dof']} "
            f"p={chi2_5x2['p']:.2g} V={chi2_5x2['cramers_V']:.3f} | "
            f"Cliff's delta (very short vs very long)="
            f"{note_level['cliffs_delta_veryshort_vs_verylong']:+.3f}"
        )

        print()
        print("=== Rests and measure position ===")
        for name, value in report["rest_and_measure"].items():
            if isinstance(value, float):
                value = round(value, 3)
            print(f"  {name}: {value}")

        print()
        print("=== Attention entropy (bits) by makam ===")
        for makam, entropy in report["attention_entropy_bits_by_makam"].items():
            print(f"  {makam:>16s}: {entropy:.2f}")

        print()
        print("=== Top-3 enriched pitches per makam (at least 1% of the notes) ===")
        frequent = enrichment[enrichment["occ_share"] >= 0.01]
        for makam, makam_rows in frequent.groupby("makam"):
            top = makam_rows.sort_values("enrichment", ascending=False).head(3)
            entries = []
            for row in top.itertuples():
                entries.append(
                    f"{row.pitch_aeu} ({row.enrichment:.2f}x, "
                    f"{row.occ_share * 100:.1f}% of notes)"
                )
            print(f"  {makam:>16s}: " + ", ".join(entries))
