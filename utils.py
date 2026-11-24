import os

# Makes the CUDA matrix kernels deterministic.
# Must be set before CUDA is initialised, so it is set before torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import random
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from constants import *


def get_device(use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    """
    Seeds every random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def use_deterministic_kernels():
    """
    Switches PyTorch to deterministic kernels, so that a training run with
    the same seed gives the same model on the same machine.
    """
    # warn_only: an operation without a deterministic implementation
    # gives a warning instead of stopping the run
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_batch_to_device(batch, device):
    """
    Moves every tensor of a batch to the device.
    The file names (a numpy array of strings) are left out.
    """
    batch_on_device = {}
    for key, value in batch.items():
        if key == "f_names":
            continue
        batch_on_device[key] = value.to(device)

    return batch_on_device


def classification_metrics(labels, preds):
    """
    labels: List[int]  true makam labels
    preds:  List[int]  predicted makam labels
    """
    class_ids = list(range(NUM_CLASSES))

    results = {}
    results["accuracy"] = accuracy_score(labels, preds)
    results["macro_f1"] = f1_score(labels, preds, average="macro")
    results["weighted_f1"] = f1_score(labels, preds, average="weighted")

    results["report"] = classification_report(
        labels,
        preds,
        labels=class_ids,
        target_names=MAKAM_NAMES,
        output_dict=True,
        zero_division=0
    )

    results["confusion_matrix"] = confusion_matrix(
        labels,
        preds,
        labels=class_ids
    ).tolist()

    return results
