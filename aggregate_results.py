"""
Aggregate results/<tag>/seed*.json files into the tables needed for the paper.

python aggregate_results.py                 # all tags, markdown summary table
python aggregate_results.py --tag full      # + pooled per-class report and confusion matrix
python aggregate_results.py --latex         # LaTeX rows instead of markdown
python aggregate_results.py --paired        # paired differences and t-tests against the proposed model,
                                            # and the error concentration in the A-tonic group per model
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix

from dataset import MAKAM_VOCAB

MAKAM_NAMES = [k for k, _ in sorted(MAKAM_VOCAB.items(), key=lambda kv: kv[1])]


def load_results(results_dir="results", tag=None):
    pattern = os.path.join(results_dir, tag or "*", "seed*.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)
        rows.append(r)
    return rows


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def summary_table(rows):
    recs = []
    for r in rows:
        recs.append({
            "tag": r["tag"], "perturbation": r.get("perturbation") or "",
            "seed": r["seed"], "accuracy": r["test"]["accuracy"],
            "macro_f1": r["test"]["macro_f1"], "val_accuracy": r.get("val_accuracy", np.nan),
            "best_epoch": r.get("best_epoch", np.nan),
        })
    df = pd.DataFrame(recs)
    g = df.groupby(["tag", "perturbation"])
    out = g.agg(n=("seed", "count"),
                acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"), acc_ci=("accuracy", ci95),
                acc_min=("accuracy", "min"), acc_max=("accuracy", "max"),
                f1_mean=("macro_f1", "mean"), f1_sd=("macro_f1", "std"), f1_ci=("macro_f1", ci95),
                epochs=("best_epoch", "mean")).reset_index()
    return df, out


def pooled_report(rows):
    """Per-class metrics computed on the union of the test predictions of all seeds."""
    y, p = [], []
    for r in rows:
        y.extend(r["test_labels"])
        p.extend(r["test_preds"])
    rep = classification_report(y, p, labels=list(range(len(MAKAM_NAMES))),
                                target_names=MAKAM_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(len(MAKAM_NAMES))))
    return pd.DataFrame(rep).T, pd.DataFrame(cm, index=MAKAM_NAMES, columns=MAKAM_NAMES)


def paired_tests(df, reference="full"):
    """Paired comparison of every (tag, perturbation) with the unperturbed reference model
    over the splits they share: mean difference in percentage points and paired t-test."""
    ref = df[(df["tag"] == reference) & (df["perturbation"] == "")].set_index("seed")["accuracy"]
    recs = []
    for (tag, pert), g in df.groupby(["tag", "perturbation"]):
        if tag == reference and pert == "":
            continue
        a = g.set_index("seed")["accuracy"]
        common = sorted(set(a.index) & set(ref.index))
        if len(common) < 3:
            continue
        d = 100 * (a.loc[common].values - ref.loc[common].values)
        recs.append({"tag": tag, "perturbation": pert, "n_splits": len(common),
                     "acc_mean": 100 * a.loc[common].mean(), "delta_points": d.mean(),
                     "p_paired_t": stats.ttest_rel(a.loc[common].values, ref.loc[common].values).pvalue,
                     "all_splits_same_sign": bool((d > 0).all() or (d < 0).all())})
    return pd.DataFrame(recs)


def pair_test(df, tag_a, tag_b):
    """Paired difference tag_a - tag_b (unperturbed runs) over the shared splits."""
    a = df[(df["tag"] == tag_a) & (df["perturbation"] == "")].set_index("seed")["accuracy"]
    b = df[(df["tag"] == tag_b) & (df["perturbation"] == "")].set_index("seed")["accuracy"]
    common = sorted(set(a.index) & set(b.index))
    d = 100 * (a.loc[common].values - b.loc[common].values)
    return d.mean(), stats.ttest_rel(a.loc[common].values, b.loc[common].values).pvalue


def group_errors(rows, group=("ussak", "huseyni", "muhayyer"), pair=("segah", "huzzam")):
    """Per tag (unperturbed runs, pooled over splits): number of errors, errors inside the
    A-tonic group, errors inside the Segah-Huzzam pair, recall of Huseyni and Muhayyer."""
    gi = [MAKAM_NAMES.index(m) for m in group]
    pi = [MAKAM_NAMES.index(m) for m in pair]
    by_tag = {}
    for r in rows:
        if not r.get("perturbation"):
            by_tag.setdefault(r["tag"], []).append(r)
    recs = []
    for tag, rs in sorted(by_tag.items()):
        if len(rs) < 5:
            continue
        _, cm = pooled_report(rs)
        c = cm.values
        recs.append({"tag": tag, "errors": int(c.sum() - np.trace(c)),
                     "within_group": int(sum(c[i, j] for i in gi for j in gi if i != j)),
                     "segah_huzzam": int(sum(c[i, j] for i in pi for j in pi if i != j)),
                     "recall_huseyni": c[gi[1], gi[1]] / c[gi[1]].sum(),
                     "recall_muhayyer": c[gi[2], gi[2]] / c[gi[2]].sum()})
    return pd.DataFrame(recs)


def fmt_pct(m, ci):
    return f"{100 * m:.2f} $\\pm$ {100 * ci:.2f}" if not np.isnan(ci) else f"{100 * m:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--paired", action="store_true")
    args = ap.parse_args()

    rows = load_results(args.results_dir, args.tag)
    if not rows:
        print("no results found")
        return
    df, out = summary_table(rows)
    pd.set_option("display.width", 200)
    print("\nPer-seed results\n", df.to_string(index=False))
    print("\nSummary (mean, SD, 95% CI half-width over seeds)\n", out.to_string(index=False))

    if args.latex:
        print("\nLaTeX rows (accuracy % mean ± 95% CI, macro-F1):")
        for _, r in out.iterrows():
            name = r["tag"] + (f" ({r['perturbation']})" if r["perturbation"] else "")
            f1 = (f"{r['f1_mean']:.3f} $\\pm$ {r['f1_ci']:.3f}" if not np.isnan(r['f1_ci'])
                  else f"{r['f1_mean']:.3f}")
            print(f"{name} & {fmt_pct(r['acc_mean'], r['acc_ci'])} & {f1} \\\\")

    if args.paired:
        pt = paired_tests(df)
        print("\nPaired comparison with the proposed model ('full', unperturbed) over the shared splits:\n"
              "delta_points = mean accuracy difference in percentage points; p = paired t-test\n",
              pt.round(4).to_string(index=False))
        if {"pc_acc", "pc_only"} <= set(df["tag"]):
            d, pv = pair_test(df, "pc_acc", "pc_only")
            print(f"\npc_acc - pc_only: {d:+.2f} points, paired t-test p = {pv:.2g}")
        print("\nErrors pooled over the five test subsets, per model "
              "(group = Ussak-Huseyni-Muhayyer):\n", group_errors(rows).round(3).to_string(index=False))

    if args.tag:
        rows_main = [r for r in rows if not r.get("perturbation")]
        rep, cm = pooled_report(rows_main)
        print(f"\nPooled per-class report for '{args.tag}' ({len(rows_main)} seeds)\n",
              rep.round(3).to_string())
        print("\nPooled confusion matrix\n", cm.to_string())
        # most frequent confusions
        conf = [(MAKAM_NAMES[i], MAKAM_NAMES[j], int(cm.values[i, j]))
                for i in range(len(MAKAM_NAMES)) for j in range(len(MAKAM_NAMES))
                if i != j and cm.values[i, j] > 0]
        conf.sort(key=lambda t: -t[2])
        print("\nOff-diagonal entries (true -> predicted, count):")
        for t in conf:
            print(f"  {t[0]:>16s} -> {t[1]:<16s} {t[2]}")


if __name__ == "__main__":
    main()
