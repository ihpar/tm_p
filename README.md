# Attention-Based Inspection of Symbolic Makam Recognition with a BiLSTM on SymbTr

Code, data and results of the paper. A BiLSTM with attention is trained to recognise
the 12 most frequent makams of SymbTr v2.4.3 from note sequences, and controlled
experiments show what the model uses and whether its attention weights can be trusted.

Tag `jnmr-r1` marks the state of the first revision. Two earlier trainings of the same
pipeline, which the paper refers to for the duration analysis, are kept in the history of
the repository: tag `jnmr-r1-prelim` (non-deterministic run) and commit `0dbc397`.

## Data

`data/txt/` holds the score files of [SymbTr](https://github.com/MTG/SymbTr) v2.4.3
(release tag `v2.4.3`), unchanged. `dataset.pkl` and `vocab.pkl` are derived from them.
SymbTr is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License (CC BY-NC-SA 4.0); the score files and the files derived from them
are redistributed here under the same license. If you use them, please cite:

> M. K. Karaosmanoğlu (2012). A Turkish makam music symbolic database for music
> information retrieval: SymbTr. In Proceedings of the 13th International Society for
> Music Information Retrieval Conference (ISMIR), pp. 223-228.

## License

The code is released under the MIT License (see `LICENSE`). The SymbTr score files and the
files derived from them keep the SymbTr license described above.

## How to run

```bash
git clone https://github.com/ihpar/tm_p && cd tm_p
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

PyTorch is installed first because its build has to match the GPU driver of the machine
(the line above is an example, see https://pytorch.org/get-started). `requirements.txt`
lists the other five libraries. The figures use the font Liberation Sans if it is installed
(`apt-get update && apt-get install -y fonts-liberation`), otherwise DejaVu Sans.

Run the notebooks in this order, every notebook from top to bottom:

| notebook | what it does | needs |
|---|---|---|
| `1_train.ipynb` | hyperparameter search, the proposed model on the five splits, pooling variants, the best configuration of the search on the five splits | GPU |
| `2_ablations_and_controls.ipynb` | feature ablations, transposed and shuffled test pieces, interval representation | GPU, notebook 1 |
| `3_baselines.ipynb` | pitch-histogram and pitch-bigram classifiers | CPU |
| `4_attention_analysis.ipynb` | deletion tests, attention statistics | GPU, notebooks 1 and 2 |
| `5_tables_and_figures.ipynb` | summary tables, all figures | notebooks 1 to 4 |

All of it is about 105 trainings of roughly two minutes each plus the analyses: a few hours
on one consumer GPU.

* A training or evaluation whose result file already exists is skipped, so notebooks 1-3
  can be stopped and started again. To repeat a run, delete its result file. The deletion
  tests and the note tables of notebook 4 are always computed again and overwrite their files.
* `QUICK_TEST = True` in the second code cell of a notebook makes a tiny dry run of every step
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
| `faithfulness.py` | `run_deletion_tests`: deletion tests and attention-saliency rank correlation; `run_pitch_matched_deletion_tests`: the control that deletes the same pitches at random positions |
| `attention_inspector.py` | `AttentionInspector`: note table, pitch enrichment, duration, measure-position and rest statistics |
| `stat_utils.py` | confidence interval, Holm correction, Cliff's delta, Friedman test with Kendall's W |
| `result_tables.py` | mean ± CI tables, paired t-tests against the proposed model, pooled per-makam report and confusions |
| `plot_manager.py` | `PlotManager`: every figure of the paper |
| `pitch_names.py` | pitch names as stored by the model and as written in SymbTr |
| `disk_utils.py`, `utils.py` | result files and checkpoints; seeding, device, classification metrics |
| `file_parser.py`, `txt_parser.ipynb` | SymbTr `.txt` parser that produced `dataset.pkl` and `vocab.pkl` from `data/txt/` (SymbTr v2.4.3) |
| `requirements.txt` | the libraries the code uses, with the versions that produced `results/` |
| `environment_freeze.txt` | `pip freeze` of the machine that produced `results/` (NVIDIA GeForce RTX 4090) |

## Results

| location | content |
|---|---|
| `results/<tag>/seed<k>.json` | one model on the test subset of split `k`: accuracy, macro-F1, per-makam report, confusion matrix, the prediction for every test piece, the training history |
| `results/<tag>/seed<k>__<perturbation>.json` | the same model on perturbed test pieces (`shuffle`, `transpose+9`, ...) |
| `results/<tag>/seed<k>.pt` | trained model (kept for `full` and `no_meas`, which the attention analysis needs) |
| `results/hparam_search.csv` | the 20 configurations of the hyperparameter search |
| `results/faithfulness/` | deletion tests (`full_*`) and the pitch-matched control (`full_pitch_matched_*`) |
| `results/attention/` | note table (one row per note of every test piece), pitch enrichment, report with all attention statistics |
| `results/summary.txt`, `full_pooled.txt`, `paired_tests.txt` | the tables of the paper as text |
| `figs/` | figures |

Tags of the models:

| tag | model |
|---|---|
| `full` | proposed model: attention, all five note attributes |
| `full_unpacked` | the same without packing (LSTM runs over the padding) |
| `mean_pool`, `max_pool`, `last_pool` | other pooling functions in place of attention |
| `no_dur`, `no_meas`, `no_rhythm`, `no_oct`, `pc_acc`, `pc_only` | feature ablations |
| `interval`, `interval_only` | interval representation, with and without duration and measure position |
| `transpose_aug` | trained with random transposition of the training pieces |
| `transpose_aug_long` | the same with a larger training budget: up to 300 epochs, early stopping after 30 epochs, learning rate halved after 10 epochs without improvement (reported in the paper) |
| `train_shuffle` | trained on pieces with shuffled note order |
| `hp_00` ... `hp_19` | trials of the hyperparameter search |
| `search_best` | the configuration with the lowest validation loss in the search (`hp_16`: 64 hidden units), trained on the five splits |
| `hist53_dur__*`, `hist53_cnt__*`, `hist_abs_dur__*`, `bigram__*` | baselines: feature set and classifier |

## Reproducibility

Every training is seeded and uses deterministic kernels: on one machine, two trainings of
the same configuration give the same model (`hp_00` and `full/seed0` are two independent
trainings of the same configuration and are identical). Across machines this does not
hold. With another GPU type, CUDA version or PyTorch build the trained models differ, and
the accuracies of single splits move by a few points in either direction. The files in
`results/` were produced on an NVIDIA GeForce RTX 4090 with the packages of
`environment_freeze.txt`. A new run on another machine should therefore be compared with
the paper through the means and confidence intervals over the five splits, not run by run.

The attention statistics and the figures can be computed again without a GPU from the note
tables `results/attention/*_notes.csv.gz` (`AttentionInspector.load_note_table`).

The unit of analysis of the attention statistics is the test classification of a piece
(seed, file name): 875 units, 656 distinct pieces.
