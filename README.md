# Attention-Based Inspection of Symbolic Makam Recognition with a BiLSTM on SymbTr

Code, data and results of the paper. A BiLSTM with attention is trained to recognise
the 12 most frequent makams of SymbTr v2.4.3 from note sequences, and controlled
experiments show what the model uses and whether its attention weights can be trusted.

Tag `jnmr-r1` marks the state of the first revision. Tag `jnmr-r1-prelim` marks the
earlier, non-deterministic run that the paper refers to for the duration analysis.

## How to run

```bash
git clone https://github.com/ihpar/tm_p && cd tm_p
pip install -r requirements.txt
```

Run the notebooks in this order, every notebook from top to bottom:

| notebook | what it does | needs |
|---|---|---|
| `1_train.ipynb` | hyperparameter search, the proposed model on the five splits, pooling variants | GPU |
| `2_ablations_and_controls.ipynb` | feature ablations, transposed and shuffled test pieces, interval representation | GPU, notebook 1 |
| `3_baselines.ipynb` | pitch-histogram and pitch-bigram classifiers | CPU |
| `4_attention_analysis.ipynb` | deletion tests, attention statistics | GPU, notebooks 1 and 2 |
| `5_tables_and_figures.ipynb` | summary tables, all figures, comparison with the archived results | notebooks 1 to 4 |

All of it is about 95 trainings of roughly two minutes each plus the analyses: a few hours
on one consumer GPU.

* A run whose result file already exists is skipped, so a notebook can be stopped and
  started again. To repeat a run, delete its result file.
* `QUICK_TEST = True` in the second cell of a notebook makes a tiny dry run of every step
  (1 epoch, 3 batches) and writes to `results_quick_test/` and `figs_quick_test/`. Use it to
  check the setup before the long run.
* A notebook can also run without the browser staying open (the outputs are saved into the
  notebook):
  `nohup jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 1_train.ipynb &`
* Nothing is passed on the command line. Every experiment is a function call with named
  parameters, e.g.

```python
run_experiment(tag="no_dur", seed=0, device=device, features=["pc", "acc", "oct", "meas"])
evaluate_trained_model(tag="full", seed=0, device=device, test_transpose=9)
```

## Files

| file | purpose |
|---|---|
| `constants.py` | vocabularies, makam names, file locations, hyperparameters, evaluation protocol |
| `dataset.py` | loading, stratified splits (written to `splits/`), `MakamDataset` with the shuffle and transposition perturbations, interval tokens, komma arithmetic, batches of similar length |
| `bilstm.py` | `BiLSTMAttentionClassifier`: packed BiLSTM, pooling = attention / mean / max / last, selectable note attributes |
| `train.py` | `train_one_epoch`, `evaluate`, `predict`, `train_model` |
| `early_stopper.py` | `EarlyStopper` |
| `experiments.py` | `run_experiment` (train one model on one split and test it), `evaluate_trained_model` (test a trained model on perturbed test pieces) |
| `hparam_search.py` | `run_hyperparameter_search`: random search over 20 configurations on split 0 |
| `baselines.py` | `run_baselines`: histogram and bigram features with logistic regression, SVM, random forest, k-NN |
| `faithfulness.py` | `run_deletion_tests`: deletion tests and attention-saliency rank correlation |
| `attention_inspector.py` | `AttentionInspector`: note table, pitch enrichment, duration, measure-position and rest statistics |
| `stat_utils.py` | confidence interval, Holm correction, Cliff's delta, Friedman test with Kendall's W |
| `result_tables.py` | mean ± CI tables, paired t-tests against the proposed model, pooled per-makam report and confusions |
| `plot_manager.py` | `PlotManager`: every figure of the paper |
| `pitch_names.py` | pitch names as stored by the model and as written in SymbTr |
| `compare_with_archive.py` | compares a new run with the archived results |
| `disk_utils.py`, `utils.py` | result files and checkpoints; seeding, device, classification metrics |
| `file_parser.py`, `txt_parser.ipynb` | SymbTr `.txt` parser that produced `dataset.pkl` and `vocab.pkl` from `data/txt/` (SymbTr v2.4.3) |
| `requirements.txt` | `pip freeze` of the environment in which the archived results were produced |

`wv_utils.py` and `dataset_checks.ipynb` are files of the preliminary version and are not used.

## Results

| location | content |
|---|---|
| `results/<tag>/seed<k>.json` | one model on the test subset of split `k`: accuracy, macro-F1, per-makam report, confusion matrix, the prediction for every test piece, the training history |
| `results/<tag>/seed<k>__<perturbation>.json` | the same model on perturbed test pieces (`shuffle`, `transpose+9`, ...) |
| `results/<tag>/seed<k>.pt` | trained model (kept for `full` and `no_meas`, which the attention analysis needs) |
| `results/hparam_search.csv` | the 20 configurations of the hyperparameter search |
| `results/faithfulness/` | deletion tests |
| `results/attention/` | note table (one row per note of every test piece), pitch enrichment, report with all attention statistics |
| `results/summary.txt`, `full_pooled.txt`, `paired_tests.txt` | the tables of the paper as text |
| `figs/` | figures |
| `results_archive/`, `figs_archive/` | results and figures of the previous run, for the comparison in notebook 5 |

Tags of the models:

| tag | model |
|---|---|
| `full` | proposed model: attention, all five note attributes |
| `full_unpacked` | the same without packing (LSTM runs over the padding) |
| `mean_pool`, `max_pool`, `last_pool` | other pooling functions in place of attention |
| `no_dur`, `no_meas`, `no_rhythm`, `no_oct`, `pc_acc`, `pc_only` | feature ablations |
| `interval`, `interval_only` | interval representation, with and without duration and measure position |
| `transpose_aug` | trained with random transposition of the training pieces |
| `train_shuffle` | trained on pieces with shuffled note order |
| `hp_00` ... `hp_19` | trials of the hyperparameter search |
| `hist53_dur__*`, `hist53_cnt__*`, `hist_abs_dur__*`, `bigram__*` | baselines: feature set and classifier |

## Reproducibility

Every training is seeded and uses deterministic kernels. On the same GPU type and the
library versions of `requirements.txt` a run reproduces the archived models exactly
(`hp_00` and `full/seed0` are two independent trainings of the same configuration and are
identical). On other hardware the accuracies can differ slightly. The last section of
`5_tables_and_figures.ipynb` compares a new run with `results_archive/` and lists every
run whose predictions differ.

The attention statistics and the figures can be computed again without a GPU from the note
tables `results/attention/*_notes.csv.gz` (`AttentionInspector.load_note_table`).

The unit of analysis of the attention statistics is the test classification of a piece
(seed, file name): 875 units, 656 distinct pieces.
