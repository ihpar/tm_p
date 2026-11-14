# Code and results for "Attention-Based Inspection of Symbolic Makam Recognition with a BiLSTM on SymbTr"

Tag `jnmr-r1` marks the state used in the revised manuscript (deterministic run);
`jnmr-r1-prelim` marks the earlier, non-deterministic run that the manuscript refers to
for the duration analysis.

## Files

| file | purpose |
|---|---|
| `run_all.py` | single driver for the whole pipeline (hyperparameter search, all trainings, test-time perturbations, baselines, deletion tests, attention statistics, figures, summary tables); resume-safe |
| `dataset.py` | stratified seeded splits (written to `splits/`), class weights, perturbations (shuffle / transpose / transposition augmentation), interval tokens, komma-index helpers |
| `bilstm.py` | packed BiLSTM, pooling = attn / mean / max / last, feature subsets |
| `run_experiments.py` | trains or evaluates one configuration, writes `results/<tag>/seed<k>.json` + `.pt` |
| `hparam_search.py` | random search (20 configurations) on the validation subset of split 0 |
| `baselines.py` | pitch-histogram / bigram baselines on the same splits |
| `faithfulness.py` | deletion tests and attention-saliency rank correlation |
| `attention_analysis.py` | note-level attention table, pitch enrichment, piece-level duration and measure-position statistics, figures |
| `results_figure.py`, `extra_figures.py`, `figstyle.py` | overview figure; confusion matrix, example piece, length histogram, makam distribution; shared figure style |
| `aggregate_results.py` | mean ± SD ± 95 % CI tables, pooled per-class report, confusion matrix; `--paired` gives the paired differences and paired t-tests against the proposed model (Δ and p of Tables 4 and 7) and the error concentration in the Uşşak–Hüseyni–Muhayyer group per model |
| `file_parser.py`, `txt_parser.ipynb` | SymbTr `.txt` parser that produced `dataset.pkl` and `vocab.pkl` from `data/txt/` (SymbTr v2.4.3) |
| `requirements.txt` | `pip freeze` of the environment in which the archived results were produced |

`train.py`, `train.ipynb`, `evaluation.ipynb`, `attention_inspector.py`, `plot_manager.py`,
`wv_utils.py` and `dataset_checks.ipynb` are legacy files of the preliminary version and are
not used by the pipeline.

## Reproducing the results

```bash
git clone https://github.com/ihpar/tm_p && cd tm_p
pip install torch scikit-learn scipy pandas matplotlib      # or: pip install -r requirements.txt
python run_all.py --fresh        # moves the archived results/ and figs_rev/ aside and re-runs everything
```

`python run_all.py` without `--fresh` skips every run whose result file already exists, so on a
fresh clone it only regenerates what is missing. Other options: `--stage hparam | train | evals |
baselines | stage2 | analysis | figures` runs one stage, `--seeds 0 1` restricts the splits,
`--hp "--hidden 256 --dropout 0.2"` overrides hyperparameters, `--device cuda|mps|cpu`, and
`--smoke` does a one-epoch dry run of every step. The full pipeline (about 100 trainings of roughly
two minutes each plus the analyses) takes a few hours on one consumer GPU. Training is seeded and
uses deterministic kernels; on the same hardware and library versions it reproduces the archived
models exactly (`hp_00` and `full/seed0` are two independent trainings of the same configuration
and are identical).

The stages write `results/hparam_search.csv`, `results/<tag>/seed<k>[__<perturbation>].json`,
`results/faithfulness/`, `results/attention/`, `figs_rev/`, `results/summary.txt` (all accuracies,
mean ± CI, LaTeX rows), `results/paired_tests.txt` (paired comparisons with the proposed model) and `results/full_pooled.txt` (per-class report and confusion matrix pooled
over the five test subsets, 875 test classifications of 656 distinct pieces).

Only the checkpoints of `full` and `no_meas` (seeds 0-4), which the attention analyses need, are
archived (`.gitignore` drops the others). The attention statistics and figures also regenerate
without checkpoints from `results/attention/*_notes.csv.gz`
(`python attention_analysis.py --tag full --from_csv`), and the deletion-test figure from
`results/faithfulness/` (`python faithfulness.py --figure_only`). The unit of analysis in
`attention_analysis.py` is the test classification of a piece (seed, file name).

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

## Stage 2, analysis and figure scripts (stages `stage2`, `analysis`, `figures` of `run_all.py`)

| file | purpose | reviewer item |
|---|---|---|
| `run_experiments.py --features int,...` | interval (transposition-invariant) representation | R1 absolute pitch |
| `run_experiments.py --train_transpose_aug 13` | transposition augmentation during training | R1 absolute pitch |
| `run_experiments.py --train_shuffle` | bag-of-notes upper bound | R1 note order |
| `faithfulness.py` | deletion tests (attention-top / saliency-top / random / attention-bottom), Spearman(attention, grad x input) | R1 faithfulness, R2-10 |
| `attention_analysis.py` | T*alpha normalisation, pitch enrichment per makam, piece-level duration statistics (Friedman, Kendall W, Wilcoxon+Holm, paired Cliff's delta), 5x2 chi-square + Cramer's V, within-piece 95th-percentile threshold, rests, measure position, attention entropy per makam, figures | R1 normalisation/frequency/statistics, R2-7, 9, 11, 12 |
| `results_figure.py` | overview bar chart of all accuracies with CIs | new results figure |
| `extra_figures.py` | pooled confusion matrix, annotated example piece (`--piece bogazici_sen` is the piece in the paper), sequence-length histogram, makam distribution | Figures 1, 2, 5, 8 |
| `attention_analysis.py --from_csv` | recompute statistics and figures from the saved note table without the checkpoints | |

Outputs: `results/faithfulness/`, `results/attention/` (note-level table, enrichment CSV,
report JSON), `figs_rev/` (enrichment, duration bins, measure position, faithfulness,
results overview).  The notation key for the paper is the spelling map printed by
`python -c "from attention_analysis import symbtr_spelling_map; print(symbtr_spelling_map())"`.

## Sanity checks already done (CPU)

* `python bilstm.py` prints, for a randomly initialised model, the maximum
  logit difference of the same piece evaluated alone vs. inside a padded batch:
  ≈1e-8 with packing, ≈1e-4 without packing (grows after training). This is
  the evidence for the padding comment in the response letter.
* A 2-epoch smoke run, the `--eval_only --test_shuffle/--test_transpose` paths,
  the ablation flags and all baselines run end-to-end; splits are stratified
  (per-makam counts via `python -c "from dataset import split_summary; print(split_summary())"`).
