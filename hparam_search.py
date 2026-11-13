"""
Random hyperparameter search for the proposed model (validation set of seed 0 only).

python hparam_search.py --trials 20 --seed 0

Each trial calls run_experiments.py with a sampled configuration and the tag
hp_<trial>; the selection criterion is validation loss of the best epoch
(the same criterion used for early stopping), with validation accuracy
reported for reference.  Results are collected in results/hparam_search.csv.
Nothing from the test set is used for the selection.
"""

import os
import json
import glob
import random
import argparse
import sys
import subprocess
import pandas as pd

SPACE = {
    "hidden": [64, 128, 256],
    "emb_dim": [16, 32, 64],
    "lr": [1e-3, 5e-4],
    "dropout": [0.2, 0.3, 0.5],
    "batch_size": [8, 16],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search_seed", type=int, default=123)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--extra", default="", help="extra args passed to run_experiments.py")
    args = ap.parse_args()

    rng = random.Random(args.search_seed)
    configs, seen = [], set()
    # always include the configuration of the original submission as trial 0
    base = {"hidden": 128, "emb_dim": 32, "lr": 1e-3, "dropout": 0.3, "batch_size": 8}
    configs.append(base)
    seen.add(tuple(base.values()))
    while len(configs) < args.trials:
        c = {k: rng.choice(v) for k, v in SPACE.items()}
        if tuple(c.values()) not in seen:
            seen.add(tuple(c.values()))
            configs.append(c)

    rows = []
    for i, c in enumerate(configs):
        tag = f"hp_{i:02d}"
        json_path = os.path.join(args.results_dir, tag, f"seed{args.seed}.json")
        if not os.path.exists(json_path):
            cmd = [sys.executable, "run_experiments.py", "--tag", tag, "--seed", str(args.seed),
                   "--hidden", str(c["hidden"]), "--emb_dim", str(c["emb_dim"]),
                   "--lr", str(c["lr"]), "--dropout", str(c["dropout"]),
                   "--batch_size", str(c["batch_size"])] + args.extra.split()
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
        with open(json_path) as f:
            r = json.load(f)
        best = min(r["history"], key=lambda h: h["val_loss"]) if r["history"] else {}
        rows.append({**c, "trial": i, "best_epoch": r["best_epoch"],
                     "val_loss": best.get("val_loss"), "val_acc": best.get("val_acc"),
                     "n_params": r["n_params"], "seconds": round(r["train_seconds"])})
        pd.DataFrame(rows).to_csv(os.path.join(args.results_dir, "hparam_search.csv"), index=False)

    df = pd.DataFrame(rows).sort_values("val_loss")
    pd.set_option("display.width", 200)
    print("\nSearch results sorted by validation loss:\n", df.to_string(index=False))
    print("\nSelected configuration:\n", df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
