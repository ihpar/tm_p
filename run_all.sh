#!/usr/bin/env bash
# Full experiment grid for the JNMR revision (E1-E4 + E5b/E6 evaluations).
# Run from the repository root on a GPU machine:  bash run_all.sh
# Re-running is safe: finished runs are skipped.
set -e
SEEDS="0 1 2 3 4"

run () {  # run <tag> <extra args...>
  tag=$1; shift
  for s in $SEEDS; do
    if [ ! -f results/$tag/seed$s.json ]; then
      python run_experiments.py --tag $tag --seed $s "$@"
    fi
  done
}

# --- 0.5 hyper-parameter search (validation set of seed 0 only) --------------
# python hparam_search.py --trials 20 --seed 0
# -> if the selected configuration differs from the defaults, add the chosen
#    --hidden/--emb_dim/--lr/--dropout/--batch_size to EVERY call below.
HP=""

# --- E1 proposed model -------------------------------------------------------
run full $HP
# original (unpacked) LSTM, for the response letter only
run full_unpacked --no_pack $HP

# --- E3 pooling alternatives -------------------------------------------------
run mean_pool --pooling mean $HP
run max_pool  --pooling max  $HP
run last_pool --pooling last $HP

# --- E4 feature ablations ----------------------------------------------------
run no_dur   --features pc,acc,oct,meas $HP
run no_meas  --features pc,acc,oct,dur  $HP
run no_oct   --features pc,acc,dur,meas $HP
run no_rhythm --features pc,acc,oct     $HP     # neither duration nor measure position
run pc_acc   --features pc,acc          $HP
run pc_only  --features pc              $HP

# --- E5b / E6: perturbed test sets on the trained E1 models ------------------
for s in $SEEDS; do
  python run_experiments.py --tag full --seed $s --eval_only --test_shuffle
  for k in 4 5 9 13 -4 -5 -9 -13; do
    python run_experiments.py --tag full --seed $s --eval_only --test_transpose $k
  done
done
# E6 upper bound: model trained on shuffled sequences is a later step (needs a
# training-time flag) -- not included here.

# --- E2 non-neural baselines -------------------------------------------------
python baselines.py --seeds $SEEDS

# --- tables ------------------------------------------------------------------
python aggregate_results.py --latex | tee results/summary.txt
python aggregate_results.py --tag full | tee results/full_pooled.txt
