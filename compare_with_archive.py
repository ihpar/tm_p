import os
import numpy as np
import pandas as pd

from constants import *
from disk_utils import load_results, load_json, get_result_path


def compare_classification_results(
    results_dir=RESULTS_DIR,
    archive_dir=ARCHIVE_RESULTS_DIR
):
    """
    Compares every result file of the archive with the result file of the
    same run in results_dir.

    returns a DataFrame with one row per archived result file:
    status:           "identical"  same predictions for every test piece
                      "different"  at least one prediction differs
                      "missing"    the run is not in results_dir
    acc_archive, acc_new, acc_difference: test accuracy in %
    same_history:     the validation loss of every epoch is the same
    """
    rows = []

    for archived in load_results(archive_dir):
        tag = archived["tag"]
        seed = archived["seed"]
        perturbation = archived.get("perturbation")

        row = {
            "tag": tag,
            "perturbation": perturbation or "",
            "seed": seed,
            "acc_archive": 100 * archived["test"]["accuracy"]
        }

        new_path = get_result_path(tag, seed, perturbation, results_dir)
        if not os.path.exists(new_path):
            row["status"] = "missing"
            rows.append(row)
            continue

        new = load_json(new_path)

        same_predictions = (
            archived["test_files"] == new["test_files"] and
            archived["test_preds"] == new["test_preds"]
        )

        archived_val_losses = [e["val_loss"] for e in archived["history"]]
        new_val_losses = [e["val_loss"] for e in new["history"]]

        if same_predictions:
            row["status"] = "identical"
        else:
            row["status"] = "different"

        row["acc_new"] = 100 * new["test"]["accuracy"]
        row["acc_difference"] = row["acc_new"] - row["acc_archive"]
        row["same_history"] = archived_val_losses == new_val_losses

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_comparison(comparison):
    """
    Prints how many runs are identical, different or missing, and lists
    the runs that are not identical.
    """
    counts = comparison["status"].value_counts()

    print(f"Runs in the archive: {len(comparison)}")
    for status in ["identical", "different", "missing"]:
        print(f"  {status}: {counts.get(status, 0)}")

    not_identical = comparison[comparison["status"] != "identical"]

    if len(not_identical) == 0:
        print("Every run reproduces the archived predictions.")
        return

    different = comparison[comparison["status"] == "different"]
    if len(different) > 0:
        largest = different["acc_difference"].abs().max()
        print(f"Largest accuracy difference: {largest:.2f} points")

    print()
    print(not_identical.to_string(index=False))


def flatten_numbers(data, prefix=""):
    """
    Flattens nested dicts and lists into {"a.b.0.c": number}.
    Only numbers are kept.
    """
    numbers = {}

    if isinstance(data, dict):
        for key, value in data.items():
            numbers.update(flatten_numbers(value, f"{prefix}{key}."))

    elif isinstance(data, list):
        for i, value in enumerate(data):
            numbers.update(flatten_numbers(value, f"{prefix}{i}."))

    elif isinstance(data, bool):
        pass

    elif isinstance(data, (int, float)):
        numbers[prefix.rstrip(".")] = data

    return numbers


def compare_json_numbers(file_name, results_dir=RESULTS_DIR,
                         archive_dir=ARCHIVE_RESULTS_DIR):
    """
    Compares every number of a JSON file in results_dir with the same
    number in the archive, e.g. "attention/full_report.json".

    returns a DataFrame with the numbers that differ, largest relative
    difference first (empty if every number is the same)
    """
    archived = flatten_numbers(load_json(os.path.join(archive_dir, file_name)))
    new = flatten_numbers(load_json(os.path.join(results_dir, file_name)))

    rows = []
    for name, archived_value in archived.items():
        if name not in new:
            rows.append({"name": name, "archive": archived_value,
                         "new": np.nan, "relative_difference": np.nan})
            continue

        new_value = new[name]
        if new_value == archived_value:
            continue

        scale = max(abs(archived_value), abs(new_value))
        rows.append({
            "name": name,
            "archive": archived_value,
            "new": new_value,
            "relative_difference": abs(new_value - archived_value) / scale
        })

    differences = pd.DataFrame(
        rows, columns=["name", "archive", "new", "relative_difference"])
    differences = differences.sort_values(
        "relative_difference", ascending=False)

    print(f"{file_name}: {len(archived)} numbers, {len(differences)} differ")

    return differences


def compare_csv_numbers(file_name, results_dir=RESULTS_DIR,
                        archive_dir=ARCHIVE_RESULTS_DIR):
    """
    Compares the numeric columns of a CSV file in results_dir with the
    archive, e.g. "faithfulness/full_summary.csv". The rows must be in the
    same order.

    returns the largest absolute difference of every numeric column
    """
    archived = pd.read_csv(os.path.join(archive_dir, file_name))
    new = pd.read_csv(os.path.join(results_dir, file_name))

    if len(archived) != len(new):
        print(f"{file_name}: {len(archived)} rows in the archive, "
              f"{len(new)} rows in the new file")
        return None

    largest_differences = {}
    for column in archived.columns:
        if column not in new.columns:
            continue
        if not pd.api.types.is_numeric_dtype(archived[column]):
            continue

        difference = (new[column] - archived[column]).abs().max()
        largest_differences[column] = difference

    largest_differences = pd.Series(largest_differences)

    num_different = int((largest_differences > 0).sum())
    print(f"{file_name}: {len(largest_differences)} numeric columns, "
          f"{num_different} differ")

    return largest_differences
