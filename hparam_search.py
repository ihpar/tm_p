import os
import random
import pandas as pd

from constants import *
from experiments import run_experiment
from disk_utils import get_result_path, load_json


def sample_search_configs(
    num_trials=NUM_SEARCH_TRIALS,
    search_seed=SEARCH_SEED
):
    """
    Draws the hyperparameter configurations of the random search.
    Trial 0 is always the configuration that was fixed beforehand.

    returns List[dict], one configuration per trial
    """
    rng = random.Random(search_seed)

    base_config = {
        "lstm_hidden": LSTM_HIDDEN,
        "emb_dim": EMB_DIM,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE
    }

    configs = [base_config]
    seen = set()
    seen.add(tuple(base_config.values()))

    while len(configs) < num_trials:
        config = {}
        for name, choices in SEARCH_SPACE.items():
            config[name] = rng.choice(choices)

        # skip a configuration that was already drawn
        if tuple(config.values()) in seen:
            continue

        seen.add(tuple(config.values()))
        configs.append(config)

    return configs


def run_hyperparameter_search(
    device,
    num_trials=NUM_SEARCH_TRIALS,
    seed=0,
    search_seed=SEARCH_SEED,
    num_epochs=NUM_EPOCHS,
    max_train_batches=None,
    results_dir=RESULTS_DIR,
    split_dir=SPLIT_DIR
):
    """
    Random search over the hyperparameters of the proposed model, on the
    split of `seed` only.

    Trial i is trained with the tag hp_<i>. The configurations are compared
    by the validation loss of their best epoch, the same criterion that is
    used for early stopping. The test subset is not used for the selection.

    The table of all trials is written to <results_dir>/hparam_search.csv

    returns the table, sorted by validation loss
    """
    configs = sample_search_configs(num_trials, search_seed)

    rows = []
    for trial, config in enumerate(configs):
        tag = f"hp_{trial:02d}"

        run_experiment(
            tag=tag,
            seed=seed,
            device=device,
            lstm_hidden=config["lstm_hidden"],
            emb_dim=config["emb_dim"],
            learning_rate=config["learning_rate"],
            dropout=config["dropout"],
            batch_size=config["batch_size"],
            num_epochs=num_epochs,
            max_train_batches=max_train_batches,
            results_dir=results_dir,
            split_dir=split_dir,
            skip_if_done=True
        )

        result = load_json(get_result_path(tag, seed, None, results_dir))

        # the epoch with the lowest validation loss
        best = None
        for epoch_stats in result["history"]:
            if best is None or epoch_stats["val_loss"] < best["val_loss"]:
                best = epoch_stats

        row = dict(config)
        row["trial"] = trial
        row["best_epoch"] = result["best_epoch"]
        row["val_loss"] = best["val_loss"]
        row["val_acc"] = best["val_acc"]
        row["n_params"] = result["n_params"]
        row["seconds"] = round(result["train_seconds"])
        rows.append(row)

        # write the table after every trial, so that nothing is lost if
        # the search is interrupted
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(results_dir, "hparam_search.csv"), index=False)

    df = pd.DataFrame(rows).sort_values("val_loss")

    return df
