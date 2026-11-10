#!/usr/bin/env bash
# Same experiments as run_all.sh, ordered so that the most important results
# come first. Safe to stop (Ctrl-C) and restart at any time.
#   bash run_priority.sh            # auto device (cuda > mps > cpu)
#   DEV="--device cpu" bash run_priority.sh
set -e
SEEDS="0 1 2 3 4"; DEV=${DEV:-}; HP=${HP:-}
run () { tag=$1; shift; for s in $SEEDS; do
  [ -f results/$tag/seed$s.json ] || python run_experiments.py --tag $tag --seed $s $DEV "$@"; done; }

# 1. main result + baselines (what the whole paper rests on)
run full $HP
python baselines.py --seeds $SEEDS
# 2. cheap evaluations on the trained main models
for s in $SEEDS; do
  python run_experiments.py --tag full --seed $s --eval_only --test_shuffle $DEV
  for k in 4 5 9 13 -4 -5 -9 -13; do
    python run_experiments.py --tag full --seed $s --eval_only --test_transpose $k $DEV; done
done
# 3. ablations that the central claims depend on
run no_dur    --features pc,acc,oct,meas $HP
run no_meas   --features pc,acc,oct,dur  $HP
run no_rhythm --features pc,acc,oct      $HP
# 4. pooling baselines
run mean_pool --pooling mean $HP
run max_pool  --pooling max  $HP
# 5. remaining ablations / variants
run pc_acc    --features pc,acc $HP
run pc_only   --features pc     $HP
run no_oct    --features pc,acc,dur,meas $HP
run last_pool --pooling last $HP
run full_unpacked --no_pack $HP
python aggregate_results.py --latex | tee results/summary.txt
python aggregate_results.py --tag full | tee results/full_pooled.txt
