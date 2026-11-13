"""
Overview figure of all classification results (mean accuracy with 95% CI over seeds).

python results_figure.py                      # -> figs_rev/results_overview.pdf/.png
python results_figure.py --tags full max_pool hist53_dur__logreg ...   # custom subset/order
"""

import os
import argparse
import numpy as np
import figstyle  # noqa: F401  (shared fonts/sizes)
import matplotlib.pyplot as plt

from aggregate_results import load_results, summary_table

DEFAULT = [
    # (tag, perturbation, display label, group)
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
    ("pc_acc", "", "pitch class + accidental only", "ablation"),
    ("pc_only", "", "pitch class only", "ablation"),
    ("interval", "", "interval representation (+dur, meas)", "invariance"),
    ("interval_only", "", "interval representation only", "invariance"),
    ("transpose_aug", "", "abs. pitch + transposition augmentation", "invariance"),
    ("transpose_aug", "transpose+4", "  same, tested transposed +4 kommas", "invariance"),
    ("transpose_aug", "transpose+9", "  same, tested transposed +9 kommas", "invariance"),
    ("full", "transpose+4", "proposed, tested transposed +4 kommas", "control"),
    ("full", "transpose+9", "proposed, tested transposed +9 kommas", "control"),
    ("full", "transpose-9", "proposed, tested transposed −9 kommas", "control"),
    ("full", "shuffle", "proposed, tested on shuffled note order", "control"),
    ("train_shuffle", "shuffle", "trained and tested on shuffled order", "control"),
]
COLORS = {"model": "#3182bd", "baseline": "#636363", "ablation": "#e6550d",
          "invariance": "#31a354", "control": "#756bb1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--fig_dir", default="figs_rev")
    ap.add_argument("--chance", type=float, default=100 / 12)
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    rows = load_results(args.results_dir)
    _, summ = summary_table(rows)
    summ = summ.set_index(["tag", "perturbation"])

    items = []
    for tag, pert, label, group in DEFAULT:
        if (tag, pert) in summ.index:
            r = summ.loc[(tag, pert)]
            items.append((label, group, 100 * r["acc_mean"], 100 * (r["acc_ci"] if not np.isnan(r["acc_ci"]) else 0),
                          int(r["n"])))
    if not items:
        print("nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(items) + 1.2))
    y = np.arange(len(items))[::-1]
    for yi, (label, group, m, ci, n) in zip(y, items):
        ax.barh(yi, m, color=COLORS[group], alpha=0.85)
        ax.errorbar(m, yi, xerr=ci, fmt="none", ecolor="k", capsize=2, lw=0.8)
        ax.text(min(m + ci + 0.8, 99), yi, f"{m:.1f}" + (f" ± {ci:.1f}" if ci else "") + (f"  (n={n})" if n < 5 else ""),
                va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=8.5)
    ax.axvline(args.chance, color="k", ls=":", lw=0.8)
    ax.text(args.chance + 0.5, y.min() - 0.6, "chance", fontsize=7)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Test accuracy (%), mean ± 95% CI over 5 stratified splits")
    from matplotlib.patches import Patch
    present = [g for g in COLORS if any(it[1] == g for it in items)]
    ax.legend(handles=[Patch(color=COLORS[g], label=g) for g in present],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=len(present), frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "results_overview.pdf"))
    fig.savefig(os.path.join(args.fig_dir, "results_overview.png"), dpi=200)
    print(f"plotted {len(items)} rows -> {args.fig_dir}/results_overview.pdf")


if __name__ == "__main__":
    main()
