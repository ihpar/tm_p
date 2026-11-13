"""
Attention analyses for the revision (replaces the note-level statistics of the
original submission).

Collects, for every note of every test piece of the chosen seeds, the attention
weight alpha, the length-normalised weight T*alpha (1 = uniform expectation),
the pitch (pitch class + accidental, octave folded; also the komma class),
relative duration, duration bin, measure position and whether the piece was
classified correctly.  Then:

  1. Pitch enrichment per makam  (R1 "attention vs frequency", R2 item 12)
       occurrence share, attention share, enrichment = attention/occurrence,
       mean T*alpha per occurrence, entropy of the attention distribution.
  2. Duration-bin statistics with the PIECE as the unit  (R1 independence)
       per piece & bin: median T*alpha -> Friedman test across the five bins
       (pieces with notes in all five bins), Kendall's W effect size, pairwise
       Wilcoxon signed-rank tests with Holm correction and paired Cliff's delta.
       Note-level Kruskal-Wallis with epsilon^2 is reported for reference only.
  3. High-attention analysis  (R1 chi-square, R2 item 11)
       high = T*alpha >= within-piece 95th percentile; 5x2 chi-square with
       Cramer's V (note level, reference) and piece-level Friedman on the
       per-piece share of high-attention notes per bin.
  4. Rests: attention on rests vs pitched notes.
  5. Measure position: mean T*alpha at start / middle / end positions.
  6. Figures (figs_rev/): enrichment per makam, duration-bin boxplots,
     measure-position bars.

python attention_analysis.py --tag full --seeds 0 1 2 3 4
python attention_analysis.py --tag no_meas --seeds 0 1 2 3 4   # for the measure-boundary check
python attention_analysis.py --tag full --from_csv              # recompute from the saved note table
"""

import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
from scipy import stats

from dataset import (load_pieces, read_split_files, MakamDataset, collate_fn, komma_index,
                     seed_everything, MAKAM_VOCAB)
from faithfulness import load_model, to_device

MAKAM_NAMES = [k for k, _ in sorted(MAKAM_VOCAB.items(), key=lambda kv: kv[1])]
BINS = [0.0, 0.0625, 0.1875, 0.375, 0.75, 1.01]
BIN_LABELS = ["Very short (≤1/16)", "Short (1/16–3/16]", "Medium (3/16–3/8]",
              "Long (3/8–3/4]", "Very long (>3/4)"]
MEAS_LABELS = {1: "start", 2: "middle", 3: "end"}
PRETTY = {"hicaz": "Hicaz", "nihavent": "Nihavent", "ussak": "Uşşak", "rast": "Rast", "huzzam": "Hüzzam",
          "segah": "Segah", "huseyni": "Hüseyni", "mahur": "Mahur", "hicazkar": "Hicazkar",
          "kurdilihicazkar": "Kürdili Hicazkar", "muhayyer": "Muhayyer", "saba": "Saba"}


def glyph(label):
    """'Bb4' -> 'B♭4', 'C#4' -> 'C♯4' for figure labels."""
    return label.replace("b", "♭", 1).replace("#", "♯", 1) if len(label) > 1 and label[1] in "b#" else label

# AEU-style display names for the (pc, acc) sharp-normalised vocabulary.
# acc id k>=2 means k-1 kommas above the natural pitch class.
PC_NAMES = {2: "G", 3: "A", 4: "B", 5: "C", 6: "D", 7: "E", 8: "F"}


def pitch_label(pc_id, acc_id, style="sharp", spelling=None):
    """Pitch label with octave folded. style='sharp' gives the model's
    sharp-normalised vocabulary token (e.g. 'A#5'); style='aeu' gives the most
    frequent spelling of that token in the SymbTr files (e.g. 'Bb4'), taken
    from the `spelling` map built by symbtr_spelling_map()."""
    if pc_id == 1:
        return "Rest"
    name = PC_NAMES[pc_id]
    k = 0 if acc_id <= 1 else acc_id - 1
    sharp = name if k == 0 else f"{name}#{k}"
    if style == "aeu" and spelling is not None:
        return spelling.get(sharp, sharp)
    return sharp


def symbtr_spelling_map(data_dir="data/txt"):
    """Most frequent original SymbTr spelling (octave removed) for every
    sharp-normalised (pc, acc) token, e.g. {'A#5': 'Bb4', 'C#4': 'C#4'}."""
    from collections import Counter, defaultdict
    from file_parser import FileParser
    if not os.path.isdir(data_dir):
        print(f"[spelling] {data_dir} not found; using sharp-normalised labels")
        return None
    fp = FileParser()
    counts = defaultdict(Counter)
    for fn in os.listdir(data_dir):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(data_dir, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                t = line.rstrip("\n").split("\t")
                if len(t) < 5 or t[1] not in fp.note_line_codes or t[3] in ("", "Es"):
                    continue
                pc, acc, _ = fp.parse_pitch(t[3])
                sharp = pc if acc is None else f"{pc}{acc}"
                counts[sharp][t[3][0] + t[3][2:]] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


def collect(model, pieces, labels, device, dur_inv, spelling=None, batch_size=32):
    ds = MakamDataset(pieces, labels)
    recs = []
    for i in range(0, len(ds), batch_size):
        items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        batch = collate_fn(items)
        with torch.no_grad():
            logits, attn = model(to_device(batch, device))
        preds = logits.argmax(1).cpu().numpy()
        attn = attn.cpu().numpy()
        for b, name in enumerate(batch["f_names"].tolist()):
            T = int(batch["lengths"][b])
            makam = int(batch["labels"][b])
            a = attn[b, :T]
            thr = np.percentile(a, 95)
            pcs, accs, octs = (batch[k][b, :T].numpy() for k in ("pcs", "accs", "octs"))
            durs, meas = batch["durs"][b, :T].numpy(), batch["meas"][b, :T].numpy()
            for t in range(T):
                recs.append({
                    "f_name": name, "makam": MAKAM_NAMES[makam], "correct": bool(preds[b] == makam),
                    "T": T, "t": t, "alpha": float(a[t]), "norm_attn": float(a[t] * T),
                    "high": bool(a[t] >= thr),
                    "pitch": pitch_label(int(pcs[t]), int(accs[t]), "sharp"),
                    "pitch_aeu": pitch_label(int(pcs[t]), int(accs[t]), "aeu", spelling),
                    "is_rest": int(pcs[t]) == 1,
                    "komma_class": (komma_index(int(pcs[t]), int(accs[t]), int(octs[t])) or 0) % 53,
                    "dur": dur_inv[int(durs[t])],
                    "meas_pos": MEAS_LABELS.get(int(meas[t]), "?"),
                })
    df = pd.DataFrame(recs)
    df["dur_bin"] = pd.cut(df["dur"], bins=BINS, labels=BIN_LABELS, include_lowest=True)
    return df


# ------------------------------------------------------------------ analyses
def pitch_enrichment(df, min_occ=1):
    rows = []
    for makam, g in df[~df["is_rest"]].groupby("makam"):
        occ = g.groupby("pitch").size()
        att = g.groupby("pitch")["alpha"].sum()
        mean_norm = g.groupby("pitch")["norm_attn"].mean()
        occ_share, att_share = occ / occ.sum(), att / att.sum()
        p = att_share.values
        entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        for pitch in occ.index:
            if occ[pitch] < min_occ:
                continue
            rows.append({"makam": makam, "pitch": pitch,
                         "pitch_aeu": g.loc[g["pitch"] == pitch, "pitch_aeu"].iloc[0],
                         "komma_class": int(g.loc[g["pitch"] == pitch, "komma_class"].iloc[0]),
                         "n_occ": int(occ[pitch]), "occ_share": occ_share[pitch],
                         "attn_share": att_share[pitch], "enrichment": att_share[pitch] / occ_share[pitch],
                         "mean_norm_attn": mean_norm[pitch], "attn_entropy_bits": entropy,
                         "n_pitches": len(occ)})
    return pd.DataFrame(rows).sort_values(["makam", "occ_share"], ascending=[True, False])


PIECE_KEY = ["seed", "f_name"]   # unit of analysis: one test classification (piece within a split)


def piece_level_bins(df, value="norm_attn", agg="median"):
    """One value per (test classification, bin): median T*alpha, or for
    value='high_share' the fraction of the bin's notes that belong to the
    piece's top-5% attention notes (expected 0.05 under no association)."""
    if value == "high_share":
        pb = df.groupby(PIECE_KEY + ["dur_bin"], observed=False)["high"].mean()
    else:
        pb = df.groupby(PIECE_KEY + ["dur_bin"], observed=False)[value].agg(agg)
    return pb.unstack("dur_bin")


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def paired_cliffs_delta(x, y):
    d = np.asarray(x) - np.asarray(y)
    return (np.sum(d > 0) - np.sum(d < 0)) / len(d)


def holm(pvals):
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    m = len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def duration_stats(df, label):
    out = {"label": label}
    # --- piece level: Friedman on pieces with all five bins
    pb = piece_level_bins(df, "norm_attn", "median").dropna()
    out["n_pieces_all_bins"] = int(len(pb))
    if len(pb) >= 5:
        stat, p = stats.friedmanchisquare(*[pb[c].values for c in pb.columns])
        out["friedman_chi2"], out["friedman_p"] = float(stat), float(p)
        out["kendall_W"] = float(stat / (len(pb) * (len(pb.columns) - 1)))
    out["piece_median_by_bin_complete_cases"] = {c: float(np.median(pb[c])) for c in pb.columns} if len(pb) else {}
    pb_all_tmp = piece_level_bins(df, "norm_attn", "median")
    out["piece_median_by_bin"] = {c: float(pb_all_tmp[c].dropna().median()) for c in pb_all_tmp.columns}
    out["n_pieces_by_bin"] = {c: int(pb_all_tmp[c].notna().sum()) for c in pb_all_tmp.columns}
    # pairwise Wilcoxon on pieces having both bins
    pb_all = piece_level_bins(df, "norm_attn", "median")
    pairs, pvals = [], []
    for a, b in itertools.combinations(BIN_LABELS, 2):
        sub = pb_all[[a, b]].dropna()
        if len(sub) < 10:
            continue
        try:
            w = stats.wilcoxon(sub[a], sub[b])
            pv = float(w.pvalue)
        except ValueError:
            pv = 1.0
        pairs.append({"bin_a": a, "bin_b": b, "n_pieces": int(len(sub)),
                      "median_a": float(sub[a].median()), "median_b": float(sub[b].median()),
                      "paired_cliffs_delta": float(paired_cliffs_delta(sub[a], sub[b])), "p_raw": pv})
        pvals.append(pv)
    if pairs:
        for r, pa in zip(pairs, holm(pvals)):
            r["p_holm"] = float(pa)
    out["pairwise"] = pairs
    # --- high-attention share, piece level
    hb = piece_level_bins(df, "high_share").dropna()
    if len(hb) >= 5:
        stat, p = stats.friedmanchisquare(*[hb[c].values for c in hb.columns])
        out["high_share_friedman_chi2"], out["high_share_friedman_p"] = float(stat), float(p)
        out["high_share_kendall_W"] = float(stat / (len(hb) * (len(hb.columns) - 1)))
        out["high_share_mean_by_bin"] = {c: float(hb[c].mean()) for c in hb.columns}
    # --- note level (reference only; observations are not independent)
    groups = [g["norm_attn"].values for _, g in df.groupby("dur_bin", observed=False) if len(g)]
    H, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    out["note_level"] = {"kruskal_H": float(H), "p": float(p), "epsilon2": float(H / (n - 1)), "n_notes": int(n),
                         "n_by_bin": {str(k): int(v) for k, v in df["dur_bin"].value_counts().items()},
                         "median_norm_attn_by_bin": {str(k): float(v) for k, v in
                                                     df.groupby("dur_bin", observed=False)["norm_attn"].median().items()},
                         "cliffs_delta_veryshort_vs_verylong": float(cliffs_delta(
                             df.loc[df["dur_bin"] == BIN_LABELS[0], "norm_attn"],
                             df.loc[df["dur_bin"] == BIN_LABELS[4], "norm_attn"]))}
    ct = pd.crosstab(df["dur_bin"], df["high"])
    chi2, pc, dof, _ = stats.chi2_contingency(ct)
    out["note_level"]["chi2_5x2"] = {"chi2": float(chi2), "dof": int(dof), "p": float(pc),
                                     "cramers_V": float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))),
                                     "high_rate_by_bin": {str(k): float(v) for k, v in
                                                          df.groupby("dur_bin", observed=False)["high"].mean().items()}}
    return out


def rest_and_measure_stats(df):
    out = {}
    out["rest_share_of_notes"] = float(df["is_rest"].mean())
    out["rest_share_of_attention"] = float(df.loc[df["is_rest"], "alpha"].sum() / df["alpha"].sum())
    out["mean_norm_attn_rests"] = float(df.loc[df["is_rest"], "norm_attn"].mean())
    out["mean_norm_attn_pitched"] = float(df.loc[~df["is_rest"], "norm_attn"].mean())
    pm = df.groupby(PIECE_KEY + ["meas_pos"])["norm_attn"].mean().unstack()
    out["measure_position_piece_mean"] = {c: float(pm[c].mean()) for c in pm.columns}
    out["measure_position_piece_median"] = {c: float(pm[c].median()) for c in pm.columns}
    hm = df.groupby(PIECE_KEY + ["meas_pos"])["high"].mean().unstack()
    out["measure_position_high_share_mean"] = {c: float(hm[c].mean()) for c in hm.columns}
    subh = hm.dropna()
    if len(subh) >= 5 and subh.shape[1] == 3:
        stat, p = stats.friedmanchisquare(*[subh[c].values for c in subh.columns])
        out["measure_position_high_share_friedman"] = {"chi2": float(stat), "p": float(p),
                                                       "kendall_W": float(stat / (len(subh) * 2)), "n_pieces": int(len(subh))}
    sub = pm.dropna()
    if len(sub) >= 5 and sub.shape[1] == 3:
        stat, p = stats.friedmanchisquare(*[sub[c].values for c in sub.columns])
        out["measure_position_friedman"] = {"chi2": float(stat), "p": float(p),
                                            "kendall_W": float(stat / (len(sub) * 2)), "n_pieces": int(len(sub))}
    return out


# ------------------------------------------------------------------ figures
def figures(df, enr, fig_dir, tag, pitch_order=False, df_dur=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. enrichment: occurrence share vs attention share, top pitches per makam
    makams = sorted(enr["makam"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(12, 13), sharey=False)
    for ax, makam in zip(axes.ravel(), makams):
        e = enr[enr["makam"] == makam].sort_values("occ_share", ascending=False).head(10)
        if pitch_order:
            e = e.sort_values("komma_class")
        x = np.arange(len(e))
        ax.bar(x - 0.2, e["occ_share"] * 100, 0.4, label="share of notes", color="#9ecae1")
        ax.bar(x + 0.2, e["attn_share"] * 100, 0.4, label="share of attention", color="#3182bd")
        ax2 = ax.twinx()
        ax2.plot(x, e["enrichment"], "k.-", lw=1, label="enrichment")
        ax2.axhline(1, color="k", ls=":", lw=0.8)
        ax2.set_ylim(0, max(2.5, e["enrichment"].max() * 1.1))
        ax.set_xticks(x)
        ax.set_xticklabels([glyph(x) for x in e["pitch_aeu"]], rotation=60, fontsize=8)
        ax.set_title(f"{PRETTY.get(makam, makam)}  (H = {e['attn_entropy_bits'].iloc[0]:.2f} bits)", fontsize=10)
        ax.set_ylabel("%", fontsize=8)
        ax2.set_ylabel("enrichment", fontsize=8)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles + [plt.Line2D([], [], color="k", marker=".", lw=1)], labels_ + ["enrichment (attention/occurrence)"],
               loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    suffix = "_pitchorder" if pitch_order else ""
    fig.savefig(os.path.join(fig_dir, f"enrichment_{tag}{suffix}.pdf"))
    fig.savefig(os.path.join(fig_dir, f"enrichment_{tag}{suffix}.png"), dpi=200)
    plt.close(fig)
    if not pitch_order:
        figures(df, enr, fig_dir, tag, pitch_order=True, df_dur=df_dur)
        return

    # 2. duration bins, piece-level medians (same subset as the statistics: rests excluded)
    pb = piece_level_bins(df_dur if df_dur is not None else df, "norm_attn", "median")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    data = [pb[c].dropna().values for c in BIN_LABELS]
    ax.boxplot(data, showfliers=False)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"{b}\n(n={len(d)})" for b, d in zip(BIN_LABELS, data)], fontsize=8)
    ax.set_ylabel("Per-unit median of T·α")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"duration_bins_{tag}.pdf"))
    fig.savefig(os.path.join(fig_dir, f"duration_bins_{tag}.png"), dpi=200)
    plt.close(fig)

    # 3. measure position
    pm = df.groupby(PIECE_KEY + ["meas_pos"])["norm_attn"].mean().unstack()
    fig, ax = plt.subplots(figsize=(4, 3.5))
    cols = [c for c in ("start", "middle", "end") if c in pm.columns]
    ax.boxplot([pm[c].dropna().values for c in cols], showfliers=False)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels([f"{c}\n(n={pm[c].notna().sum()})" for c in cols])
    ax.set_ylabel("Per-unit mean of T·α")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"measure_position_{tag}.pdf"))
    fig.savefig(os.path.join(fig_dir, f"measure_position_{tag}.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--fig_dir", default="figs_rev")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data_dir", default="data/txt", help="SymbTr txt files, for original pitch spellings")
    ap.add_argument("--include_rests_in_duration", action="store_true")
    ap.add_argument("--from_csv", action="store_true",
                    help="recompute statistics and figures from results/attention/<tag>_notes.csv.gz (no models needed)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else torch.device(args.device)

    pieces_all, labels_all, vocab = load_pieces()
    dur_inv = {v: k for k, v in vocab["dur"].items()}
    dur_inv[0] = 0.0
    out_dir = os.path.join(args.results_dir, "attention")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    spelling = symbtr_spelling_map(args.data_dir)
    frames = []
    if args.from_csv:
        df = pd.read_csv(os.path.join(out_dir, f"{args.tag}_notes.csv.gz"))
        df = df[df["seed"].isin(args.seeds)]
        df["dur_bin"] = pd.Categorical(df["dur_bin"], categories=BIN_LABELS)
        args.seeds = sorted(df["seed"].unique().tolist())
    for seed in ([] if args.from_csv else args.seeds):
        seed_everything(seed)
        model, _ = load_model(os.path.join(args.results_dir, args.tag, f"seed{seed}.pt"), device)
        tr, va, te = read_split_files(pieces_all, seed, args.split_dir)
        df = collect(model, [pieces_all[i] for i in te], [labels_all[i] for i in te], device, dur_inv, spelling)
        df["seed"] = seed
        frames.append(df)
        print(f"[{args.tag} s{seed}] collected {len(df)} notes from {df['f_name'].nunique()} pieces")
    if not args.from_csv:
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(os.path.join(out_dir, f"{args.tag}_notes.csv.gz"), index=False)

    enr = pitch_enrichment(df)
    enr.to_csv(os.path.join(out_dir, f"{args.tag}_pitch_enrichment.csv"), index=False)

    dfd = df if args.include_rests_in_duration else df[~df["is_rest"]]
    report = {
        "tag": args.tag, "seeds": args.seeds, "n_notes": int(len(df)),
        "n_test_classifications": int(df.groupby(PIECE_KEY).ngroups), "n_distinct_pieces": int(df["f_name"].nunique()),
        "duration_all": duration_stats(dfd, "all test pieces"),
        "duration_correct_only": duration_stats(dfd[dfd["correct"]], "correctly classified pieces"),
        "rest_and_measure": rest_and_measure_stats(df),
        "attention_entropy_bits_by_makam": enr.groupby("makam")["attn_entropy_bits"].first().to_dict(),
        "n_pitches_by_makam": enr.groupby("makam")["n_pitches"].first().to_dict(),
    }
    with open(os.path.join(out_dir, f"{args.tag}_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)

    figures(df, enr, args.fig_dir, args.tag, df_dur=dfd)

    # console summary
    d = report["duration_all"]
    print("\n=== Duration bins (piece level, rests excluded) ===")
    print("n by bin:", d["n_pieces_by_bin"])
    print(f"pieces with all 5 bins: {d['n_pieces_all_bins']}  Friedman chi2={d.get('friedman_chi2', float('nan')):.2f} "
          f"p={d.get('friedman_p', float('nan')):.2g}  Kendall W={d.get('kendall_W', float('nan')):.3f}")
    print("per-piece median T*alpha by bin:", {k: round(v, 3) for k, v in d["piece_median_by_bin"].items()})
    for r in d["pairwise"]:
        print(f"  {r['bin_a']:>20s} vs {r['bin_b']:<20s} n={r['n_pieces']:3d} delta={r['paired_cliffs_delta']:+.3f} "
              f"p_holm={r['p_holm']:.3g}")
    nl = d["note_level"]
    print(f"note level (reference): KW H={nl['kruskal_H']:.1f} eps2={nl['epsilon2']:.4f}; "
          f"chi2(5x2)={nl['chi2_5x2']['chi2']:.1f} dof={nl['chi2_5x2']['dof']} p={nl['chi2_5x2']['p']:.2g} "
          f"V={nl['chi2_5x2']['cramers_V']:.3f}; Cliff's d(very short vs very long)={nl['cliffs_delta_veryshort_vs_verylong']:+.3f}")
    print("high-attention rate by bin:", {k: round(v, 3) for k, v in nl["chi2_5x2"]["high_rate_by_bin"].items()})
    rm = report["rest_and_measure"]
    print("\n=== Rests / measure position ===")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rm.items()})
    print("\n=== Attention entropy (bits) by makam ===")
    print({k: round(v, 2) for k, v in report["attention_entropy_bits_by_makam"].items()})
    print("\n=== Top-3 enriched pitches per makam (min 1% occurrence) ===")
    for makam, g in enr[enr["occ_share"] >= 0.01].groupby("makam"):
        top = g.sort_values("enrichment", ascending=False).head(3)
        print(f"  {makam:>16s}: " + ", ".join(f"{r.pitch_aeu} ({r.enrichment:.2f}x, {r.occ_share*100:.1f}% of notes)"
                                              for r in top.itertuples()))
    print(f"\nfiles written to {out_dir} and {args.fig_dir}")


if __name__ == "__main__":
    main()
