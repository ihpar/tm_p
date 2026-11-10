# Revision experiments – how to run

## Files (drop into the repository root, replacing the old `dataset.py` / `bilstm.py`)

| file | purpose |
|---|---|
| `dataset.py` | stratified seeded splits (written to `splits/`), class weights, test-time perturbations (shuffle / transpose), komma-index helpers |
| `bilstm.py` | packed BiLSTM, pooling = attn / mean / max / last, feature subsets |
| `run_experiments.py` | trains or evaluates one configuration, writes `results/<tag>/seed<k>.json` + `.pt` |
| `hparam_search.py` | random search on the validation set of seed 0 |
| `baselines.py` | pitch-histogram / bigram baselines on the same splits |
| `aggregate_results.py` | mean ± SD ± 95 % CI tables, pooled per-class report, confusion matrix |
| `run_all.sh` | the whole grid |

The old `train.py`, `train.ipynb` and `evaluation.ipynb` keep working: the model's
`forward(batch) -> (logits, attn_weights)` interface and the
`build_dataloaders(...) -> (train, class_weights, val, test)` return order are unchanged.
`evaluation.ipynb` should now be pointed at `results/full/seed0.pt` and
`build_dataloaders(..., seed=0)`.

## Setup on lightning.ai

```bash
git clone https://github.com/ihpar/tmm_rep_learn && cd tmm_rep_learn
pip install torch scikit-learn scipy pandas scikit-posthocs matplotlib openpyxl
pip freeze > requirements.txt          # commit this
bash run_all.sh                        # ~50 trainings; a few minutes each on a T4
```

## Order

1. `python hparam_search.py --trials 20` – look at `results/hparam_search.csv`.
   If the best configuration (by validation loss) differs from the defaults
   (hidden 128, emb 32, lr 1e-3, dropout 0.3, batch 8), put those values in the
   `HP=` line of `run_all.sh`. Either way the search table goes into the paper
   (Reviewer 2, item 6).
2. `bash run_all.sh`.
3. `python aggregate_results.py --latex` → rows for the results table;
   `python aggregate_results.py --tag full` → per-class table (pooled over the 5 test
   sets, 875 pieces) and confusion matrix for Fig. 3 / Table 2.
4. Commit `splits/`, `results/**/*.json`, `results/*.txt`, `requirements.txt`
   (the `.pt` checkpoints are ~1 MB each; commit at least `results/full/seed0.pt`,
   which is the model used for all attention figures).

## What each result answers

| reviewer item | result files |
|---|---|
| single split / CI (R1, R2-4) | `full/seed0-4.json` → mean ± CI |
| padding inside LSTM (R1) | `full` vs `full_unpacked` |
| baselines (R1, R2-3, R2-5) | `hist53_dur__*`, `hist53_cnt__*`, `hist_abs_dur__*`, `bigram__*`, `mean_pool`, `max_pool`, `last_pool` |
| ablations / duration role / measure feature (R1) | `no_dur`, `no_meas`, `no_oct`, `no_rhythm`, `pc_acc`, `pc_only` |
| absolute pitch / transposition (R1) | `full/seed*__transpose±k.json` |
| note order / seyir (R1) | `full/seed*__shuffle.json` |
| hyper-parameters (R1, R2-6) | `results/hparam_search.csv` + `args` block in every json |

Not in this package yet (next step): interval representation (E5a),
training with transposition augmentation (E5c), train-on-shuffled upper bound (E6),
faithfulness tests (E7), attention normalisation / enrichment analyses and
piece-level statistics (Step 2), figure regeneration (Step 3).

## Sanity checks already done (CPU)

* `python bilstm.py` prints, for a randomly initialised model, the maximum
  logit difference of the same piece evaluated alone vs. inside a padded batch:
  ≈1e-8 with packing, ≈1e-4 without packing (grows after training). This is
  the evidence for the padding comment in the response letter.
* A 2-epoch smoke run, the `--eval_only --test_shuffle/--test_transpose` paths,
  the ablation flags and all baselines run end-to-end; splits are stratified
  (per-makam counts via `python -c "from dataset import split_summary; print(split_summary())"`).
