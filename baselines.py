"""
Non-neural baselines on exactly the same stratified splits as the BiLSTM runs.

Feature sets
  hist53_dur   duration-weighted pitch-class histogram at komma resolution
               (53 bins; octave folded; rests excluded)          <- requested by R2
  hist53_cnt   occurrence-count pitch-class histogram, 53 bins
  hist_abs_dur duration-weighted histogram of absolute pitches (octave kept,
               265 bins) -- closest to what the neural model actually sees
  bigram       TF-IDF of absolute-pitch bigrams (a minimal sequence baseline)

Classifiers (hyper-parameters chosen on the validation set, then the model
fitted on the training set is evaluated once on the test set)
  logreg, svm_rbf, rf, knn

python baselines.py --seeds 0 1 2 3 4
python aggregate_results.py            # the json files are in results/<feature>__<clf>/
"""

import os
import json
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from dataset import (load_pieces, make_split_indices, read_split_files, write_split_files,
                     komma_index, PC_VOCAB, MAKAM_VOCAB)
from run_experiments import metrics

MAKAM_NAMES = [k for k, _ in sorted(MAKAM_VOCAB.items(), key=lambda kv: kv[1])]
N_OCT = 5  # octaves 3..7 in the vocabulary


def piece_features(piece, dur_inv, kind):
    notes = piece["notes"]
    if kind == "bigram":
        toks = []
        for pc, acc, octv, dur, meas in notes:
            k = komma_index(pc, acc, octv)
            toks.append("R" if k is None else str(k))
        return " ".join(f"{a}_{b}" for a, b in zip(toks[:-1], toks[1:]))

    if kind == "hist_abs_dur":
        h = np.zeros(53 * N_OCT)
    else:
        h = np.zeros(53)
    for pc, acc, octv, dur, meas in notes:
        k = komma_index(pc, acc, octv)
        if k is None:
            continue  # rest
        w = dur_inv[dur] if kind.endswith("_dur") else 1.0
        if kind == "hist_abs_dur":
            idx = k - 3 * 53
            if 0 <= idx < len(h):
                h[idx] += w
        else:
            h[k % 53] += w
    s = h.sum()
    return h / s if s > 0 else h


def build_matrix(pieces, idx, dur_inv, kind):
    feats = [piece_features(pieces[i], dur_inv, kind) for i in idx]
    return feats if kind == "bigram" else np.vstack(feats)


def classifier_grid(name):
    if name == "logreg":
        return [("C", c, make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)))
                for c in (0.01, 0.1, 1, 10, 100)]
    if name == "svm_rbf":
        return [("C,gamma", f"{c},{g}", make_pipeline(StandardScaler(), SVC(C=c, gamma=g)))
                for c in (0.1, 1, 10, 100) for g in ("scale", 0.01, 0.001)]
    if name == "rf":
        return [("min_samples_leaf", m, RandomForestClassifier(
            n_estimators=500, min_samples_leaf=m, random_state=0, n_jobs=-1)) for m in (1, 2, 5)]
    if name == "knn":
        return [("k", k, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)))
                for k in (1, 3, 5, 9, 15)]
    raise ValueError(name)


def run(kind, clf_name, pieces, labels, dur_inv, split, seed, results_dir):
    tr, va, te = split
    y = np.asarray(labels)
    Xtr, Xva, Xte = (build_matrix(pieces, s, dur_inv, kind) for s in (tr, va, te))
    if kind == "bigram":
        vec = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(Xtr)
        Xva, Xte = vec.transform(Xva), vec.transform(Xte)
        grid = [(n, v, (clf.steps[-1][1] if hasattr(clf, "steps") else clf))
                for n, v, clf in classifier_grid(clf_name)]  # sparse: no scaler
    else:
        grid = classifier_grid(clf_name)

    best = None
    for pname, pval, clf in grid:
        clf.fit(Xtr, y[tr])
        va_acc = accuracy_score(y[va], clf.predict(Xva))
        if best is None or va_acc > best[0]:
            best = (va_acc, pname, pval, clf)
    va_acc, pname, pval, clf = best
    preds = clf.predict(Xte).tolist()
    m = metrics(y[te].tolist(), preds)
    tag = f"{kind}__{clf_name}"
    out_dir = os.path.join(results_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    result = {"tag": tag, "seed": seed, "perturbation": None,
              "args": {"features": kind, "classifier": clf_name, "selected": {pname: str(pval)}},
              "best_epoch": None, "history": [], "train_seconds": 0, "n_params": None,
              "val_accuracy": va_acc, "test": m,
              "test_files": [pieces[i]["f_name"] for i in te],
              "test_labels": y[te].tolist(), "test_preds": preds,
              "test_lengths": [len(pieces[i]["notes"]) for i in te]}
    with open(os.path.join(out_dir, f"seed{seed}.json"), "w") as f:
        json.dump(result, f)
    print(f"[{tag} s{seed}] val {va_acc:.4f} ({pname}={pval}) | TEST acc {m['accuracy']:.4f} "
          f"macro-F1 {m['macro_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--features", nargs="+",
                    default=["hist53_dur", "hist53_cnt", "hist_abs_dur", "bigram"])
    ap.add_argument("--classifiers", nargs="+", default=["logreg", "svm_rbf", "rf", "knn"])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--split_dir", default="splits")
    args = ap.parse_args()

    pieces, labels, vocab = load_pieces()
    dur_inv = {v: k for k, v in vocab["dur"].items()}
    dur_inv[0] = 0.0
    for seed in args.seeds:
        split = read_split_files(pieces, seed, args.split_dir)
        if split is None:
            split = make_split_indices(labels, seed)
            write_split_files(pieces, split, seed, args.split_dir)
        for kind in args.features:
            for clf in args.classifiers:
                if kind == "bigram" and clf in ("rf", "knn"):
                    continue  # slow / uninformative on sparse tf-idf; keep logreg & svm
                run(kind, clf, pieces, labels, dur_inv, split, seed, args.results_dir)


if __name__ == "__main__":
    main()
