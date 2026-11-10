"""
Aggregate results/<tag>/seed*.json files into the tables needed for the paper.

python aggregate_results.py                 # all tags, markdown summary table
python aggregate_results.py --tag full      # + pooled per-class report and confusion matrix
python aggregate_results.py --latex         # LaTeX rows instead of markdown
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


def fmt_pct(m, ci):
    return f"{100 * m:.2f} $\\pm$ {100 * ci:.2f}" if not np.isnan(ci) else f"{100 * m:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--latex", action="store_true")
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
