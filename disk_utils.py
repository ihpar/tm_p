import os
import json
import glob
import torch

from constants import *


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------
def get_perturbation_name(test_shuffle=False, test_transpose=0):
    """
    Name of a perturbation of the test pieces, used in the result file names.
    "shuffle", "transpose+9", "transpose-13" or None if nothing is perturbed
    """
    name_parts = []
    if test_shuffle:
        name_parts.append("shuffle")
    if test_transpose:
        name_parts.append(f"transpose{test_transpose:+d}")

    if len(name_parts) == 0:
        return None

    return "__".join(name_parts)


def get_checkpoint_path(tag, seed, results_dir=RESULTS_DIR):
    """
    <results_dir>/<tag>/seed<seed>.pt
    """
    return os.path.join(results_dir, tag, f"seed{seed}.pt")


def get_result_path(tag, seed, perturbation=None, results_dir=RESULTS_DIR):
    """
    <results_dir>/<tag>/seed<seed>.json
    <results_dir>/<tag>/seed<seed>__<perturbation>.json
    """
    file_name = f"seed{seed}"
    if perturbation:
        file_name += f"__{perturbation}"
    file_name += ".json"

    return os.path.join(results_dir, tag, file_name)


def is_done(tag, seed, perturbation=None, results_dir=RESULTS_DIR):
    """
    True if the result file of this run already exists.
    """
    return os.path.exists(get_result_path(tag, seed, perturbation, results_dir))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(model, save_path, epoch, val_loss, model_config):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
            "model_config": model_config
        },
        save_path
    )


def load_checkpoint(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device)


def get_model_config(checkpoint):
    """
    Returns the settings needed to rebuild the model of a checkpoint.
    """
    return checkpoint["model_config"]


# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as out_file:
        json.dump(data, out_file)


def load_json(path):
    with open(path) as in_file:
        return json.load(in_file)


def load_results(results_dir=RESULTS_DIR, tag=None):
    """
    Loads the result files <results_dir>/<tag>/seed*.json
    tag: None loads the results of every tag.
    returns List[dict]
    """
    if tag is None:
        pattern = os.path.join(results_dir, "*", "seed*.json")
    else:
        pattern = os.path.join(results_dir, tag, "seed*.json")

    results = []
    for path in sorted(glob.glob(pattern)):
        results.append(load_json(path))

    return results
