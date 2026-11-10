#!/usr/bin/env bash
# Stage 2: transposition-invariant models, shuffled-training upper bound,
# faithfulness tests, attention analyses and figures.  Run after run_priority.sh.
#   bash run_stage2.sh            (SEEDS / DEV / HP can be overridden as before)
set -e
SEEDS=${SEEDS:-"0 1 2 3 4"}; DEV=${DEV:-}; HP=${HP:-}
run () { tag=$1; shift; for s in $SEEDS; do
  [ -f results/$tag/seed$s.json ] || python run_experiments.py --tag $tag --seed $s $DEV $HP "$@"; done; }

# E5a interval representation (no absolute pitch at all)
run interval      --features int,dur,meas
run interval_only --features int
# E5c absolute pitch + random transposition augmentation (+-13 kommas) in training
run transpose_aug --train_transpose_aug 13
for s in $SEEDS; do for k in 4 9; do
  python run_experiments.py --tag transpose_aug --seed $s --eval_only --test_transpose $k $DEV $HP; done; done
# E6 upper bound: bag-of-notes training (trained and tested on shuffled order)
for s in $SEEDS; do
  [ -f results/train_shuffle/seed$s.json ] || \
  python run_experiments.py --tag train_shuffle --seed $s --train_shuffle $DEV $HP
  python run_experiments.py --tag train_shuffle --seed $s --eval_only --test_shuffle $DEV $HP
done

# E7 faithfulness + attention analyses (fast; CPU is fine)
python faithfulness.py --tag full --seeds $SEEDS
python attention_analysis.py --tag full --seeds $SEEDS
python attention_analysis.py --tag no_meas --seeds $SEEDS     # measure-boundary check without the feature
python results_figure.py
python aggregate_results.py --latex | tee results/summary.txt
python aggregate_results.py --tag full | tee results/full_pooled.txt
