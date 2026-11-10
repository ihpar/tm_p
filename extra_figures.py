"""
Figures that complement attention_analysis.py / results_figure.py:
  figs_rev/confusion_pooled.pdf     pooled confusion matrix of the main model (5 test sets)
  figs_rev/example_piece.pdf        attention trajectory of one test piece with measure
                                    boundaries and the top peaks annotated (pitch, duration,
                                    position in measure)
  figs_rev/length_hist.pdf          distribution of sequence lengths in the dataset

python extra_figures.py [--piece <substring of file name>] [--seed 0]
"""

import os
import json
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import load_pieces, read_split_files, MakamDataset, collate_fn, MAKAM_VOCAB, OCT_VOCAB
OCT_ID_TO_VAL = {v: k for k, v in OCT_VOCAB.items()}
from faithfulness import load_model, to_device
from attention_analysis import pitch_label, symbtr_spelling_map, MEAS_LABELS

MAKAM_NAMES = [k for k, _ in sorted(MAKAM_VOCAB.items(), key=lambda kv: kv[1])]
PRETTY = {"hicaz": "Hicaz", "nihavent": "Nihavent", "ussak": "Uşşak", "rast": "Rast", "huzzam": "Hüzzam",
          "segah": "Segah", "huseyni": "Hüseyni", "mahur": "Mahur", "hicazkar": "Hicazkar",
          "kurdilihicazkar": "Kürdili Hicazkar", "muhayyer": "Muhayyer", "saba": "Saba"}


def confusion_figure(results_dir, fig_dir, tag="full"):
    cm = np.zeros((12, 12), dtype=int)
    n = 0
    for p in sorted(glob.glob(os.path.join(results_dir, tag, "seed*.json"))):
        r = json.load(open(p))
        if r.get("perturbation"):
            continue
        cm += np.array(r["test"]["confusion_matrix"])
        n += 1
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(12):
        for j in range(12):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    labels = [PRETTY[m] for m in MAKAM_NAMES]
    ax.set_xticks(range(12)); ax.set_yticks(range(12))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted makam"); ax.set_ylabel("True makam")
    ax.set_title(f"Pooled confusion matrix ({n} test sets, {cm.sum()} pieces)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "confusion_pooled.pdf"))
    fig.savefig(os.path.join(fig_dir, "confusion_pooled.png"), dpi=200)
    plt.close(fig)
    print("confusion matrix:", cm.sum(), "pieces")


def example_piece_figure(results_dir, split_dir, fig_dir, seed, piece_sub, top_k, device):
    pieces, labels, vocab = load_pieces()
    dur_inv = {v: k for k, v in vocab["dur"].items()}
    spelling = symbtr_spelling_map("data/txt")
    model, _ = load_model(os.path.join(results_dir, "full", f"seed{seed}.pt"), device)
    tr, va, te = read_split_files(pieces, seed, split_dir)
    cands = [i for i in te if piece_sub in pieces[i]["f_name"]]
    if not cands:
        raise SystemExit(f"no test piece of seed {seed} matches '{piece_sub}'")
    i = cands[0]
    piece, label = pieces[i], labels[i]
    ds = MakamDataset([piece], [label])
    batch = collate_fn([ds[0]])
    with torch.no_grad():
        logits, attn = model(to_device(batch, device))
    a = attn[0].cpu().numpy()
    T = len(a)
    pred = MAKAM_NAMES[int(logits.argmax())]
    notes = piece["notes"]
    norm = a * T
    starts = [t for t, n in enumerate(notes) if n[4] == 1]

    fig, ax = plt.subplots(figsize=(11, 4))
    for s in starts:
        ax.axvline(s - 0.5, color="#d62728", lw=0.6, alpha=0.5)
    ax.plot(range(T), norm, color="#1f77b4", lw=1)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    top = sorted(np.argsort(-a)[:top_k])
    ymax = norm.max()
    # group neighbouring peaks into one label each
    clusters, cur = [], [top[0]]
    for t in top[1:]:
        (cur.append(t) if t - cur[-1] <= max(3, T * 0.02) else (clusters.append(cur), cur := [t]))
    clusters.append(cur)
    prev_tc = -1e9
    level = 0
    for cl in clusters:
        lines = []
        for t in cl:
            pc, acc, octv, dur, meas = notes[t]
            oct_str = "" if pc == 1 else f" (oct. {OCT_ID_TO_VAL[octv]})"
            lines.append(f"{t}: {pitch_label(pc, acc, 'aeu', spelling)}{oct_str}, {dur_inv[dur]:.3g}, {MEAS_LABELS[meas]}")
        tc = cl[len(cl) // 2]
        ytop = max(norm[t] for t in cl)
        level = (level + 1) % 2 if tc - prev_tc < 0.15 * T else 0
        prev_tc = tc
        ha = "right" if tc > 0.85 * T else ("left" if tc < 0.15 * T else "center")
        ax.annotate("\n".join(lines), (tc, ytop), xytext=(tc, min(ytop + ymax * (0.1 + 0.22 * level), ymax * 1.2)),
                    textcoords="data",
                    ha=ha, va="bottom", fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color="gray"))
    ax.set_xlabel("Note index")
    ax.set_ylabel("Normalised attention  $T\\alpha_t$")
    title = piece["f_name"].replace(".txt", "").replace("--", " | ").replace("_", " ")
    ax.set_title(f"{title}\n true {PRETTY[MAKAM_NAMES[label]]}, predicted {PRETTY[pred]}, T = {T} notes; "
                 f"red lines = measure starts; labels = note index: pitch, relative duration, position in measure", fontsize=9)
    ax.set_ylim(0, norm.max() * 1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "example_piece.pdf"))
    fig.savefig(os.path.join(fig_dir, "example_piece.png"), dpi=200)
    plt.close(fig)
    print("example piece:", piece["f_name"], "T =", T, "pred", pred)
    # also dump the annotated peaks as a small table for the text
    rows = []
    for t in sorted(top):
        pc, acc, octv, dur, meas = notes[t]
        rows.append((int(t), pitch_label(pc, acc, "aeu", spelling), OCT_ID_TO_VAL.get(octv), dur_inv[dur], MEAS_LABELS[meas], round(float(norm[t]), 2)))
    print("top peaks (index, pitch, octave, rel. duration, measure position, T*alpha):")
    for r in rows:
        print("  ", r)
    return rows


def length_figure(fig_dir):
    pieces, labels, _ = load_pieces()
    L = np.array([len(p["notes"]) for p in pieces])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(L, bins=np.arange(0, L.max() + 100, 100), color="#9ecae1", edgecolor="white")
    for q, c in ((np.median(L), "k"), (np.percentile(L, 95), "#d62728")):
        ax.axvline(q, color=c, ls="--", lw=1)
    ax.set_xlabel("Notes per piece"); ax.set_ylabel("Pieces")
    ax.set_title(f"Sequence lengths (N = {len(L)}): median {int(np.median(L))}, mean {L.mean():.0f}, "
                 f"95th pct {int(np.percentile(L, 95))}, max {L.max()}", fontsize=9)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "length_hist.pdf"))
    fig.savefig(os.path.join(fig_dir, "length_hist.png"), dpi=200)
    plt.close(fig)
    print(f"lengths: min {L.min()} median {np.median(L):.0f} mean {L.mean():.1f} p95 {np.percentile(L, 95):.0f} max {L.max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--fig_dir", default="figs_rev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--piece", default="hicaz--sarki", help="substring selecting the example test piece")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    confusion_figure(args.results_dir, args.fig_dir)
    length_figure(args.fig_dir)
    example_piece_figure(args.results_dir, args.split_dir, args.fig_dir, args.seed, args.piece, args.top_k,
                         torch.device(args.device))
