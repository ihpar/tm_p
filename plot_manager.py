import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.patches import Patch

from constants import *
from pitch_names import get_symbtr_name, with_accidental_symbols

OCT_ID_TO_VALUE = {v: k for k, v in OCT_VOCAB.items()}

# Rows of the results overview figure: (tag, perturbation, label, group)
OVERVIEW_ROWS = [
    ("full", "", "BiLSTM-attention (proposed)", "model"),
    ("full_unpacked", "", "  same, unpacked LSTM (preliminary version)", "model"),
    ("max_pool", "", "BiLSTM, max pooling", "model"),
    ("mean_pool", "", "BiLSTM, mean pooling", "model"),
    ("last_pool", "", "BiLSTM, last states", "model"),
    ("hist_abs_dur__svm_rbf", "", "Abs.-pitch histogram (dur.-weighted) + SVM", "baseline"),
    ("hist53_dur__logreg", "", "53-bin pitch-class histogram (dur.-weighted) + LR", "baseline"),
    ("hist53_dur__svm_rbf", "", "53-bin pitch-class histogram (dur.-weighted) + SVM", "baseline"),
    ("hist53_cnt__logreg", "", "53-bin pitch-class histogram (counts) + LR", "baseline"),
    ("bigram__svm_rbf", "", "Pitch-bigram TF-IDF + SVM", "baseline"),
    ("no_dur", "", "− duration", "ablation"),
    ("no_meas", "", "− measure position", "ablation"),
    ("no_rhythm", "", "− duration − measure position", "ablation"),
    ("no_oct", "", "− octave", "ablation"),
    ("pc_acc", "", "note letter + accidental only", "ablation"),
    ("pc_only", "", "note letter only", "ablation"),
    ("interval", "", "interval representation (+dur, meas)", "invariance"),
    ("interval_only", "", "interval representation only", "invariance"),
    ("transpose_aug_long", "", "abs. pitch + transposition augmentation", "invariance"),
    ("transpose_aug_long", "transpose+4", "  same, tested transposed +4 kommas", "invariance"),
    ("transpose_aug_long", "transpose+9", "  same, tested transposed +9 kommas", "invariance"),
    ("full", "transpose+4", "proposed, tested transposed +4 kommas", "control"),
    ("full", "transpose+9", "proposed, tested transposed +9 kommas", "control"),
    ("full", "transpose-9", "proposed, tested transposed −9 kommas", "control"),
    ("full", "shuffle", "proposed, tested on shuffled note order", "control"),
    ("train_shuffle", "shuffle", "trained and tested on shuffled order", "control")
]

OVERVIEW_GROUP_COLORS = {
    "model": "#3182bd",
    "baseline": "#636363",
    "ablation": "#e6550d",
    "invariance": "#31a354",
    "control": "#756bb1"
}

DELETION_ORDER_LABELS = {
    "attn_top": "highest attention first",
    "grad_top": "highest saliency first",
    "random": "random",
    "attn_bottom": "lowest attention first",
    "pitch_matched": "same pitches as highest attention,\nrandom positions"
}

# deletion order -> (marker, line style)
DELETION_ORDER_STYLES = {
    "attn_top": ("o", "-"),
    "grad_top": ("s", "-"),
    "random": ("^", "--"),
    "attn_bottom": ("v", ":"),
    "pitch_matched": ("D", "-.")
}


class PlotManager:
    def __init__(self, figs_dir=FIGS_DIR):
        self.figs_dir = figs_dir

        # the same sans-serif font and font sizes in every figure,
        # fonts embedded as TrueType in the PDF files
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8
        })

    def _finish(self, fig, f_name, save_fig):
        """
        Saves the figure as <figs_dir>/<f_name>.pdf and .png if save_fig is
        True, then shows it.
        """
        if save_fig:
            os.makedirs(self.figs_dir, exist_ok=True)
            fig.savefig(os.path.join(self.figs_dir, f"{f_name}.pdf"))
            fig.savefig(os.path.join(self.figs_dir, f"{f_name}.png"), dpi=200)
        plt.show()
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    def plot_makam_distribution(self, labels, save_fig=False):
        """
        Number of pieces per makam.
        labels: List[int], the makam label of every piece
        """
        counts = Counter(labels)
        # most frequent makam first
        order = sorted(counts, key=lambda label: -counts[label])

        names = [MAKAM_DISPLAY_NAMES[MAKAM_NAMES[label]] for label in order]
        values = [counts[label] for label in order]

        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(names, values, color="#6baed6", edgecolor="white")

        for i, value in enumerate(values):
            ax.text(i, value + 2, str(value), ha="center", fontsize=8)

        ax.set_ylabel("Number of pieces")
        ax.set_ylim(0, max(values) * 1.12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

        fig.tight_layout()
        self._finish(fig, "distribution", save_fig)

    def plot_length_histogram(self, pieces, save_fig=False):
        """
        Number of notes per piece, on a logarithmic axis.
        Median (dashed line) and 95th percentile (dotted line).
        """
        lengths = np.array([len(piece["notes"]) for piece in pieces])

        # The axis is logarithmic, so the bins are logarithmically spaced.
        bins = np.logspace(
            np.log10(lengths.min()), np.log10(lengths.max()), 36)

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.hist(lengths, bins=bins, color="#9ecae1", edgecolor="white")

        ax.axvline(np.median(lengths), color="k", ls="--", lw=1)
        ax.axvline(np.percentile(lengths, 95), color="#d62728", ls=":", lw=1.5)

        ax.set_xlabel("Notes per piece")
        ax.set_ylabel("Pieces")
        ax.set_xscale("log")
        ax.set_xlim(lengths.min() * 0.9, lengths.max() * 1.1)

        fig.tight_layout()
        self._finish(fig, "length_hist", save_fig)

        print(
            f"Lengths | min: {lengths.min()}, "
            f"median: {np.median(lengths):.1f}, "
            f"mean: {lengths.mean():.1f}, "
            f"95th percentile: {np.percentile(lengths, 95):.1f}, "
            f"max: {lengths.max()}"
        )

    # -----------------------------------------------------------------------
    # Classification results
    # -----------------------------------------------------------------------
    def plot_results_overview(self, summary, save_fig=False):
        """
        Mean test accuracy with 95% CI of every model and control.
        summary: DataFrame from result_tables.summary_table
        """
        summary = summary.set_index(["tag", "perturbation"])
        chance = 100 / NUM_CLASSES

        # (label, group, mean accuracy %, CI half-width %, number of splits)
        items = []
        for tag, perturbation, label, group in OVERVIEW_ROWS:
            if (tag, perturbation) not in summary.index:
                continue

            row = summary.loc[(tag, perturbation)]
            accuracy = 100 * row["acc_mean"]
            if np.isnan(row["acc_ci"]):
                ci = 0
            else:
                ci = 100 * row["acc_ci"]

            items.append((label, group, accuracy, ci, int(row["n"])))

        if len(items) == 0:
            print("Nothing to plot")
            return

        fig, ax = plt.subplots(figsize=(8, 0.32 * len(items) + 1.2))

        # first row at the top
        y_positions = np.arange(len(items))[::-1]

        for y, (label, group, accuracy, ci, n) in zip(y_positions, items):
            ax.barh(y, accuracy, color=OVERVIEW_GROUP_COLORS[group], alpha=0.85)
            ax.errorbar(accuracy, y, xerr=ci, fmt="none",
                        ecolor="k", capsize=2, lw=0.8)

            text = f"{accuracy:.1f}"
            if ci:
                text += f" ± {ci:.1f}"
            if n < 5:
                text += f"  (n={n})"
            ax.text(min(accuracy + ci + 0.8, 99), y, text,
                    va="center", fontsize=7.5)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([item[0] for item in items], fontsize=8.5)

        ax.axvline(chance, color="k", ls=":", lw=0.8)
        ax.text(chance + 0.5, y_positions.min() - 0.6, "chance", fontsize=7)

        ax.set_xlim(0, 112)
        ax.set_xlabel(
            "Test accuracy (%), mean ± 95% CI over 5 stratified splits")

        groups_present = []
        for group in OVERVIEW_GROUP_COLORS:
            if any(item[1] == group for item in items):
                groups_present.append(group)

        legend_handles = []
        for group in groups_present:
            legend_handles.append(
                Patch(color=OVERVIEW_GROUP_COLORS[group], label=group))

        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            fontsize=8,
            ncol=len(groups_present),
            frameon=False
        )
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        self._finish(fig, "results_overview", save_fig)

    def plot_confusion_matrix(self, cm, save_fig=False):
        """
        cm: (12, 12) array of counts, rows = true makam, columns = predicted
        """
        cm = np.asarray(cm)
        names = [MAKAM_DISPLAY_NAMES[makam] for makam in MAKAM_NAMES]

        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        im = ax.imshow(cm, cmap="Blues")

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i, j]:
                    if cm[i, j] > cm.max() / 2:
                        text_color = "white"
                    else:
                        text_color = "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=8, color=text_color)

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted makam")
        ax.set_ylabel("True makam")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        self._finish(fig, "confusion_pooled", save_fig)

    # -----------------------------------------------------------------------
    # Faithfulness
    # -----------------------------------------------------------------------
    def plot_deletion_curves(
        self,
        curves,
        summary,
        tag,
        pitch_matched_summary=None,
        save_fig=False
    ):
        """
        Test accuracy after deleting a fraction of the notes of every piece,
        for the four deletion orders (mean ± SD over the seeds).

        curves, summary: DataFrames from faithfulness.run_deletion_tests
        pitch_matched_summary: summary from
            faithfulness.run_pitch_matched_deletion_tests. If given, the
            control (same pitches, random positions) is drawn as a fifth curve.
        """
        order_names = ["attn_top", "grad_top", "random", "attn_bottom"]

        if pitch_matched_summary is not None:
            control_rows = pitch_matched_summary[
                pitch_matched_summary["order"] == "pitch_matched"]
            summary = pd.concat([summary, control_rows], ignore_index=True)
            order_names = [
                "attn_top", "grad_top", "random", "attn_bottom", "pitch_matched"]

        base_accuracy = curves["base_accuracy"].mean()

        fig, ax = plt.subplots(figsize=(5.5, 3.8))

        for order_name in order_names:
            rows = summary[summary["order"] == order_name].sort_values("frac")

            # start every curve at 0% deleted = accuracy of the full pieces
            x = [0] + (rows["frac"] * 100).tolist()
            y = [base_accuracy] + rows["acc_mean"].tolist()
            y_err = [0] + rows["acc_sd"].fillna(0).tolist()

            marker, line_style = DELETION_ORDER_STYLES[order_name]
            ax.errorbar(
                x,
                np.array(y) * 100,
                yerr=np.array(y_err) * 100,
                marker=marker,
                ls=line_style,
                capsize=3,
                label=DELETION_ORDER_LABELS[order_name]
            )

        ax.set_xlabel("Notes deleted per piece (%)")
        ax.set_ylabel("Test accuracy (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        self._finish(fig, f"faithfulness_{tag}", save_fig)

    # -----------------------------------------------------------------------
    # Attention
    # -----------------------------------------------------------------------
    def plot_pitch_enrichment(
        self,
        enrichment,
        tag,
        pitch_order=False,
        save_fig=False
    ):
        """
        One panel per makam with its ten most frequent pitches:
        share of notes (light bars), share of attention (dark bars) and
        their ratio, the enrichment (line, right axis).

        enrichment:  DataFrame from AttentionInspector.pitch_enrichment
        pitch_order: False -> pitches ordered by frequency
                     True  -> pitches ordered by pitch
        """
        # the makams in the order of the other figures and tables
        # (most frequent first)
        makams = [m for m in MAKAM_NAMES if m in set(enrichment["makam"])]

        # The figure is as wide as a page, so that the labels keep a
        # readable size in print.
        fig, axes = plt.subplots(4, 3, figsize=(7.6, 10.2), sharey=False)

        for ax, makam in zip(axes.ravel(), makams):
            rows = enrichment[enrichment["makam"] == makam]
            rows = rows.sort_values("occ_share", ascending=False).head(10)
            if pitch_order:
                rows = rows.sort_values("komma_class")

            x = np.arange(len(rows))
            ax.bar(x - 0.2, rows["occ_share"] * 100, 0.4,
                   label="share of notes", color="#9ecae1")
            ax.bar(x + 0.2, rows["attn_share"] * 100, 0.4,
                   label="share of attention", color="#3182bd")

            ax2 = ax.twinx()
            ax2.plot(x, rows["enrichment"], "k.-", lw=1, label="enrichment")
            ax2.axhline(1, color="k", ls=":", lw=0.8)
            ax2.set_ylim(0, max(2.5, rows["enrichment"].max() * 1.1))

            pitch_labels = [
                with_accidental_symbols(pitch) for pitch in rows["pitch_aeu"]]
            ax.set_xticks(x)
            ax.set_xticklabels(pitch_labels, rotation=60, fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax2.tick_params(axis="y", labelsize=8)

            makam_name = MAKAM_DISPLAY_NAMES.get(makam, makam)
            entropy = rows["attn_entropy_bits"].iloc[0]
            ax.set_title(f"{makam_name}  (H = {entropy:.2f} bits)", fontsize=9)
            ax.set_ylabel("%", fontsize=8)
            ax2.set_ylabel("enrichment", fontsize=8)

        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color="k", marker=".", lw=1))
        legend_labels.append("enrichment (attention/occurrence)")
        fig.legend(handles, legend_labels,
                   loc="lower center", ncol=3, fontsize=8)

        fig.tight_layout(rect=(0, 0.03, 1, 1))

        f_name = f"enrichment_{tag}"
        if pitch_order:
            f_name += "_pitchorder"
        self._finish(fig, f_name, save_fig)

    def plot_attention_by_duration(self, unit_table, tag, save_fig=False):
        """
        Box plot of the per-unit median T * alpha in every duration bin.

        unit_table: DataFrame from AttentionInspector.unit_by_duration_bin
                    (one row per unit, one column per duration bin)
        """
        data = []
        tick_labels = []
        for dur_bin in DURATION_BIN_LABELS:
            values = unit_table[dur_bin].dropna().values
            data.append(values)
            tick_labels.append(f"{dur_bin}\n(n={len(values)})")

        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.boxplot(data, showfliers=False)
        ax.axhline(1, color="k", ls=":", lw=0.8)

        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel("Per-unit median of T·α")

        fig.tight_layout()
        self._finish(fig, f"duration_bins_{tag}", save_fig)

    def plot_attention_by_measure_position(self, unit_table, tag, save_fig=False):
        """
        Box plot of the per-unit mean T * alpha at the start, in the middle
        and at the end of the measure.

        unit_table: DataFrame from AttentionInspector.unit_by_measure_position
                    (one row per unit, one column per measure position)
        """
        positions = []
        for position in ["start", "middle", "end"]:
            if position in unit_table.columns:
                positions.append(position)

        data = []
        tick_labels = []
        for position in positions:
            data.append(unit_table[position].dropna().values)
            tick_labels.append(
                f"{position}\n(n={unit_table[position].notna().sum()})")

        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.boxplot(data, showfliers=False)
        ax.axhline(1, color="k", ls=":", lw=0.8)

        ax.set_xticks(range(1, len(positions) + 1))
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Per-unit mean of T·α")

        fig.tight_layout()
        self._finish(fig, f"measure_position_{tag}", save_fig)

    def _describe_note(self, t, note, dur_values, spelling_map,
                       use_symbols=True):
        """
        "290: B♭4, octave 4, 0.125, middle"
        (note index, pitch, octave, relative duration, position in measure)

        use_symbols: True writes the accidental as a flat or sharp symbol
                     (for a figure), False writes "Bb4" (for printing)
        """
        pc, acc, octave, dur, meas = note

        pitch = get_symbtr_name(pc, acc, spelling_map)
        if use_symbols:
            pitch = with_accidental_symbols(pitch)

        if pc == REST_ID:
            octave_text = ""
        else:
            octave_text = f", octave {OCT_ID_TO_VALUE[octave]}"

        return (
            f"{t}: {pitch}{octave_text}, "
            f"{dur_values[dur]:.3g}, {MEAS_ID_TO_NAME[meas]}"
        )

    def plot_example_piece(
        self,
        notes,
        attn,
        dur_values,
        spelling_map,
        num_peaks=EXAMPLE_PIECE_NUM_PEAKS,
        save_fig=False
    ):
        """
        Normalised attention T * alpha over the notes of one piece. Red
        lines mark the measure starts. The highest peaks are annotated
        with note index, pitch, octave, relative duration and position in
        the measure.

        notes: List[Tuple[int,int,int,int,int]]  (pc, acc, oct, dur, meas)
        attn:  array (T,), attention weights of the piece
        """
        T = len(attn)
        norm_attn = attn * T

        measure_starts = []
        for t, note in enumerate(notes):
            if note[4] == MEAS_VOCAB["start"]:
                measure_starts.append(t)

        fig, ax = plt.subplots(figsize=(10, 4.2))

        for t in measure_starts:
            ax.axvline(t - 0.5, color="#d62728", lw=0.6, alpha=0.5)

        ax.plot(range(T), norm_attn, color="#1f77b4", lw=1)
        ax.axhline(1, color="k", ls=":", lw=0.8)

        # the highest peaks, in the order of the piece
        peaks = sorted(np.argsort(-attn)[:num_peaks])
        y_max = norm_attn.max()

        # neighbouring peaks share one label
        max_gap = max(3, T * 0.02)
        clusters = []
        current_cluster = [peaks[0]]
        for t in peaks[1:]:
            if t - current_cluster[-1] <= max_gap:
                current_cluster.append(t)
            else:
                clusters.append(current_cluster)
                current_cluster = [t]
        clusters.append(current_cluster)

        prev_center = -1e9
        level = 0

        for cluster in clusters:
            lines = []
            for t in cluster:
                lines.append(
                    self._describe_note(t, notes[t], dur_values, spelling_map))

            center = cluster[len(cluster) // 2]
            y_top = max(norm_attn[t] for t in cluster)

            # labels that are close to each other alternate between two heights
            if center - prev_center < 0.15 * T:
                level = (level + 1) % 2
            else:
                level = 0
            prev_center = center

            if center > 0.85 * T:
                h_align = "right"
            elif center < 0.15 * T:
                h_align = "left"
            else:
                h_align = "center"

            y_text = min(y_top + y_max * (0.1 + 0.22 * level), y_max * 1.2)

            ax.annotate(
                "\n".join(lines),
                (center, y_top),
                xytext=(center, y_text),
                textcoords="data",
                ha=h_align,
                va="bottom",
                fontsize=8.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="gray")
            )

        ax.set_xlabel("Note index")
        ax.set_ylabel("Normalised attention  $T\\alpha_t$")
        ax.set_ylim(0, norm_attn.max() * 1.6)
        ax.tick_params(labelsize=9)

        fig.tight_layout()
        self._finish(fig, "example_piece", save_fig)

        print("Highest peaks "
              "(index: pitch, octave, relative duration, measure position | T*alpha)")
        for t in peaks:
            description = self._describe_note(
                t, notes[t], dur_values, spelling_map, use_symbols=False)
            print(f"  {description} | {norm_attn[t]:.2f}")
