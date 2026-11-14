"""
Single Python driver for the complete experiment pipeline of the revision.
Runs every stage of the pipeline in order (formerly split over shell scripts).

    python run_all.py                 # everything, in order, resume-safe
    python run_all.py --fresh         # move an existing results/ to results_old_<timestamp>/ first
    python run_all.py --stage train   # only one stage: hparam | train | evals | baselines |
                                      #   stage2 | analysis | figures
    python run_all.py --seeds 0 1     # restrict training seeds (for parallel runs on one GPU)
    python run_all.py --smoke         # 1-epoch, 3-batch dry run of every step (CPU ok)

Every training writes results/<tag>/seed<k>.json and .pt; finished runs are
skipped, so the script can be stopped and restarted at any time.  Run it with
    nohup python run_all.py > log_run_all.txt 2>&1 &
on a remote machine so that a dropped connection does not kill it.
"""

import os
import sys
import time
import shutil
import argparse
import subprocess

PY = sys.executable
SEEDS_ALL = [0, 1, 2, 3, 4]
TRANSPOSITIONS = [4, 5, 9, 13, -4, -5, -9, -13]

# tag -> extra arguments of run_experiments.py (E1, E3, E4, reference variants)
TRAIN_GRID = {
    "full": [],
    "mean_pool": ["--pooling", "mean"],
    "max_pool": ["--pooling", "max"],
    "last_pool": ["--pooling", "last"],
    "no_dur": ["--features", "pc,acc,oct,meas"],
    "no_meas": ["--features", "pc,acc,oct,dur"],
    "no_rhythm": ["--features", "pc,acc,oct"],
    "no_oct": ["--features", "pc,acc,dur,meas"],
    "pc_acc": ["--features", "pc,acc"],
    "pc_only": ["--features", "pc"],
    "full_unpacked": ["--no_pack"],
    # stage 2 (E5a, E5c, E6)
    "interval": ["--features", "int,dur,meas"],
    "interval_only": ["--features", "int"],
    "transpose_aug": ["--train_transpose_aug", "13"],
    "train_shuffle": ["--train_shuffle"],
}
STAGE2_TAGS = ["interval", "interval_only", "transpose_aug", "train_shuffle"]


def sh(cmd, log):
    """Run one command, echo it, stop the pipeline on failure."""
    line = " ".join(cmd)
    print(f"\n$ {line}", flush=True)
    log.write(line + "\n")
    log.flush()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed (exit {r.returncode}): {line}")


def done(tag, seed, pert=None):
    name = f"seed{seed}" + (f"__{pert}" if pert else "") + ".json"
    return os.path.exists(os.path.join("results", tag, name))


def train(tag, seed, extra, args, log):
    if done(tag, seed):
        print(f"  [skip] {tag} seed {seed} already done")
        return
    cmd = [PY, "run_experiments.py", "--tag", tag, "--seed", str(seed)] + extra + args.hp
    if args.smoke:
        cmd += ["--epochs", "1", "--max_train_batches", "3"]
    if args.device:
        cmd += ["--device", args.device]
    sh(cmd, log)


def evaluate(tag, seed, pert_args, pert_name, args, log):
    if done(tag, seed, pert_name):
        print(f"  [skip] {tag} seed {seed} {pert_name} already done")
        return
    cmd = [PY, "run_experiments.py", "--tag", tag, "--seed", str(seed), "--eval_only"] + pert_args + args.hp
    if args.device:
        cmd += ["--device", args.device]
    sh(cmd, log)


def stage_hparam(args, log):
    csv = os.path.join("results", "hparam_search.csv")
    if os.path.exists(csv) and not args.smoke:
        with open(csv) as f:
            n_rows = sum(1 for _ in f) - 1
        if n_rows >= 20:
            print("  [skip] hyperparameter search already done (results/hparam_search.csv)")
            return
        print(f"  hyperparameter search incomplete ({n_rows}/20 trials); resuming")
    cmd = [PY, "hparam_search.py", "--trials", "2" if args.smoke else "20", "--seed", "0"]
    if args.smoke:
        cmd += ["--extra", "--epochs 1 --max_train_batches 3" + (f" --device {args.device}" if args.device else "")]
    sh(cmd, log)


def stage_train(args, log):
    for tag, extra in TRAIN_GRID.items():
        if tag in STAGE2_TAGS:
            continue
        for seed in args.seeds:
            train(tag, seed, extra, args, log)


def stage_evals(args, log):
    for seed in args.seeds:
        evaluate("full", seed, ["--test_shuffle"], "shuffle", args, log)
        for k in TRANSPOSITIONS:
            evaluate("full", seed, ["--test_transpose", str(k)], f"transpose{k:+d}", args, log)


def stage_baselines(args, log):
    sh([PY, "baselines.py", "--seeds"] + [str(s) for s in args.seeds], log)


def stage_stage2(args, log):
    for tag in STAGE2_TAGS:
        for seed in args.seeds:
            train(tag, seed, TRAIN_GRID[tag], args, log)
    for seed in args.seeds:
        for k in (4, 9):
            evaluate("transpose_aug", seed, ["--test_transpose", str(k)], f"transpose{k:+d}", args, log)
        evaluate("train_shuffle", seed, ["--test_shuffle"], "shuffle", args, log)


def stage_analysis(args, log):
    seeds = [str(s) for s in args.seeds]
    fast = ["--fracs", "0.1", "--n_random", "1"] if args.smoke else []
    sh([PY, "faithfulness.py", "--tag", "full", "--seeds"] + seeds + fast + (["--device", args.device] if args.device else []), log)
    for tag in ("full", "no_meas"):
        sh([PY, "attention_analysis.py", "--tag", tag, "--seeds"] + seeds + (["--device", args.device] if args.device else []), log)


def stage_figures(args, log):
    sh([PY, "results_figure.py"], log)
    sh([PY, "extra_figures.py", "--seed", "0"], log)
    with open(os.path.join("results", "summary.txt"), "w") as f:
        subprocess.run([PY, "aggregate_results.py", "--latex"], stdout=f, check=True)
    with open(os.path.join("results", "full_pooled.txt"), "w") as f:
        subprocess.run([PY, "aggregate_results.py", "--tag", "full"], stdout=f, check=True)
    with open(os.path.join("results", "paired_tests.txt"), "w") as f:
        subprocess.run([PY, "aggregate_results.py", "--paired"], stdout=f, check=True)
    print("\nwrote results/summary.txt, results/full_pooled.txt and results/paired_tests.txt")


STAGES = {
    "hparam": stage_hparam,
    "train": stage_train,
    "evals": stage_evals,
    "baselines": stage_baselines,
    "stage2": stage_stage2,
    "analysis": stage_analysis,
    "figures": stage_figures,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), default=None, help="run only this stage")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_ALL)
    ap.add_argument("--hp", default="", help='hyperparameter overrides as one quoted string, e.g. --hp "--hidden 256 --dropout 0.2"')
    ap.add_argument("--device", default=None, help="auto (default) | cuda | mps | cpu")
    ap.add_argument("--fresh", action="store_true", help="move existing results/ and figs_rev/ aside before starting (splits/ is kept)")
    ap.add_argument("--smoke", action="store_true", help="tiny dry run of every step")
    args = ap.parse_args()
    args.hp = args.hp.split()

    for f in ("run_experiments.py", "baselines.py", "faithfulness.py", "attention_analysis.py", "dataset.pkl", "vocab.pkl"):
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found: run this script from the repository root")

    if args.fresh:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for d in ("results", "figs_rev"):
            if os.path.exists(d):
                shutil.move(d, f"{d}_old_{stamp}")
                print(f"moved {d}/ to {d}_old_{stamp}/")
        # splits/ is kept on purpose: the same stratified splits are reused

    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    with open("results/run_all_commands.log", "a") as log:
        log.write(f"\n# run_all.py started {time.ctime()} args={vars(args)}\n")
        stages = [args.stage] if args.stage else list(STAGES)
        for name in stages:
            print(f"\n==================== stage: {name} ====================", flush=True)
            STAGES[name](args, log)
    print(f"\nall done in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
