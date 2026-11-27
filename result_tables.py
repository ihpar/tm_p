import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from constants import *
from stat_utils import ci95
from disk_utils import load_results

# The three makams that share the tonic A and differ mainly in melodic
# progression, and the pair that differs mainly in one degree.
A_TONIC_GROUP = ["ussak", "huseyni", "muhayyer"]
SEGAH_HUZZAM_PAIR = ["segah", "huzzam"]


def per_seed_table(results):
    """
    One row per result file: accuracy and macro-F1 of one model on the
    test subset of one split.

    results: List[dict], loaded with disk_utils.load_results
    """
    rows = []
    for result in results:
        rows.append({
            "tag": result["tag"],
            "perturbation": result.get("perturbation") or "",
            "seed": result["seed"],
            "accuracy": result["test"]["accuracy"],
            "macro_f1": result["test"]["macro_f1"],
            "val_accuracy": result.get("val_accuracy", np.nan),
            "best_epoch": result.get("best_epoch", np.nan)
        })

    return pd.DataFrame(rows)


def summary_table(per_seed):
    """
    Mean, SD and half-width of the 95% confidence interval over the seeds,
    for every tag and perturbation.
    """
    grouped = per_seed.groupby(["tag", "perturbation"])

    summary = grouped.agg(
        n=("seed", "count"),
        acc_mean=("accuracy", "mean"),
        acc_sd=("accuracy", "std"),
        acc_ci=("accuracy", ci95),
        acc_min=("accuracy", "min"),
        acc_max=("accuracy", "max"),
        f1_mean=("macro_f1", "mean"),
        f1_sd=("macro_f1", "std"),
        f1_ci=("macro_f1", ci95),
        epochs=("best_epoch", "mean")
    ).reset_index()

    return summary


def latex_rows(summary):
    """
    One LaTeX table row per tag: accuracy % (mean ± 95% CI) and macro-F1.
    returns List[str]
    """
    rows = []
    for _, row in summary.iterrows():
        name = row["tag"]
        if row["perturbation"]:
            name += f" ({row['perturbation']})"

        if np.isnan(row["acc_ci"]):
            accuracy = f"{100 * row['acc_mean']:.2f}"
        else:
            accuracy = (
                f"{100 * row['acc_mean']:.2f} $\\pm$ "
                f"{100 * row['acc_ci']:.2f}"
            )

        if np.isnan(row["f1_ci"]):
            macro_f1 = f"{row['f1_mean']:.3f}"
        else:
            macro_f1 = f"{row['f1_mean']:.3f} $\\pm$ {row['f1_ci']:.3f}"

        rows.append(f"{name} & {accuracy} & {macro_f1} \\\\")

    return rows


def pooled_report(results):
    """
    Per-class metrics and confusion matrix on the union of the test
    predictions of all given results (the five test subsets of one model).

    returns report, cm (DataFrames)
    """
    labels = []
    preds = []
    for result in results:
        labels.extend(result["test_labels"])
        preds.extend(result["test_preds"])

    class_ids = list(range(NUM_CLASSES))

    report = classification_report(
        labels,
        preds,
        labels=class_ids,
        target_names=MAKAM_NAMES,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=class_ids)

    report = pd.DataFrame(report).T
    cm = pd.DataFrame(cm, index=MAKAM_NAMES, columns=MAKAM_NAMES)

    return report, cm


def list_confusions(cm):
    """
    The off-diagonal entries of a confusion matrix, most frequent first.
    returns List[(true makam, predicted makam, count)]
    """
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm.values[i, j] > 0:
                confusions.append(
                    (MAKAM_NAMES[i], MAKAM_NAMES[j], int(cm.values[i, j])))

    confusions.sort(key=lambda confusion: -confusion[2])

    return confusions


def get_accuracy_by_seed(per_seed, tag, perturbation=""):
    """
    returns the accuracies of one tag as a Series indexed by seed
    """
    rows = per_seed[
        (per_seed["tag"] == tag) &
        (per_seed["perturbation"] == perturbation)
    ]
    return rows.set_index("seed")["accuracy"]


def paired_difference(per_seed, tag_a, tag_b):
    """
    Paired comparison of two models over the splits they share.
    returns the mean accuracy difference tag_a - tag_b in percentage
    points and the p-value of the paired t-test
    """
    accuracy_a = get_accuracy_by_seed(per_seed, tag_a)
    accuracy_b = get_accuracy_by_seed(per_seed, tag_b)

    common_seeds = sorted(set(accuracy_a.index) & set(accuracy_b.index))
    values_a = accuracy_a.loc[common_seeds].values
    values_b = accuracy_b.loc[common_seeds].values

    difference = 100 * (values_a - values_b)
    p_value = stats.ttest_rel(values_a, values_b).pvalue

    return difference.mean(), p_value


def corrected_resampled_t_test(difference, num_train, num_test):
    """
    Corrected resampled t-test of Nadeau & Bengio (2003).

    The splits share training and test pieces, so they are not independent
    and the usual paired t-test is too optimistic. The correction replaces
    the factor 1/J of the variance of the mean difference by
    1/J + num_test/num_train (J = number of splits).

    difference: differences of two models over the splits (array)
    returns the two-sided p-value
    """
    difference = np.asarray(difference, dtype=float)
    num_splits = len(difference)

    variance = difference.var(ddof=1)
    if variance == 0:
        return float("nan")

    factor = 1 / num_splits + num_test / num_train
    t_value = difference.mean() / np.sqrt(factor * variance)

    return float(2 * stats.t.sf(abs(t_value), num_splits - 1))


def paired_tests(per_seed, reference_tag="full", num_train=813, num_test=175):
    """
    Paired comparison of every tag and perturbation with the unperturbed
    reference model, over the splits they share.

    delta_points: mean accuracy difference in percentage points
    ci_low, ci_high: 95% confidence interval of delta_points (t distribution)
    p_paired_t:   p-value of the paired t-test
    p_corrected:  p-value of the corrected resampled t-test, which takes
                  the overlap of the splits into account
                  (num_train, num_test: pieces per training and test subset)
    all_splits_same_sign: the difference has the same sign on every split
    """
    reference = get_accuracy_by_seed(per_seed, reference_tag)

    rows = []
    for (tag, perturbation), group in per_seed.groupby(["tag", "perturbation"]):
        if tag == reference_tag and perturbation == "":
            continue

        accuracy = group.set_index("seed")["accuracy"]
        common_seeds = sorted(set(accuracy.index) & set(reference.index))
        if len(common_seeds) < 3:
            continue

        values = accuracy.loc[common_seeds].values
        reference_values = reference.loc[common_seeds].values
        difference = 100 * (values - reference_values)

        all_positive = bool((difference > 0).all())
        all_negative = bool((difference < 0).all())

        # half-width of the 95% confidence interval of the mean difference
        num_splits = len(common_seeds)
        standard_error = difference.std(ddof=1) / np.sqrt(num_splits)
        half_width = stats.t.ppf(0.975, num_splits - 1) * standard_error

        rows.append({
            "tag": tag,
            "perturbation": perturbation,
            "n_splits": num_splits,
            "acc_mean": 100 * accuracy.loc[common_seeds].mean(),
            "delta_points": difference.mean(),
            "ci_low": difference.mean() - half_width,
            "ci_high": difference.mean() + half_width,
            "p_paired_t": stats.ttest_rel(values, reference_values).pvalue,
            "p_corrected": corrected_resampled_t_test(
                difference, num_train, num_test),
            "all_splits_same_sign": all_positive or all_negative
        })

    return pd.DataFrame(rows)


def group_errors(results):
    """
    For every model (unperturbed runs, pooled over the splits):

    errors:          number of misclassified test pieces
    within_group:    errors inside the group Uşşak-Hüseyni-Muhayyer
    segah_huzzam:    errors between Segah and Hüzzam
    recall_huseyni, recall_muhayyer
    """
    group_ids = [MAKAM_NAMES.index(makam) for makam in A_TONIC_GROUP]
    pair_ids = [MAKAM_NAMES.index(makam) for makam in SEGAH_HUZZAM_PAIR]
    huseyni = MAKAM_NAMES.index("huseyni")
    muhayyer = MAKAM_NAMES.index("muhayyer")

    results_by_tag = {}
    for result in results:
        if result.get("perturbation"):
            continue
        if result["tag"] not in results_by_tag:
            results_by_tag[result["tag"]] = []
        results_by_tag[result["tag"]].append(result)

    rows = []
    for tag, tag_results in sorted(results_by_tag.items()):
        if len(tag_results) < 5:
            continue

        _, cm = pooled_report(tag_results)
        cm = cm.values

        within_group = 0
        for i in group_ids:
            for j in group_ids:
                if i != j:
                    within_group += cm[i, j]

        within_pair = 0
        for i in pair_ids:
            for j in pair_ids:
                if i != j:
                    within_pair += cm[i, j]

        rows.append({
            "tag": tag,
            "errors": int(cm.sum() - np.trace(cm)),
            "within_group": int(within_group),
            "segah_huzzam": int(within_pair),
            "recall_huseyni": cm[huseyni, huseyni] / cm[huseyni].sum(),
            "recall_muhayyer": cm[muhayyer, muhayyer] / cm[muhayyer].sum()
        })

    return pd.DataFrame(rows)


def write_summary_files(results_dir=RESULTS_DIR, main_tag="full"):
    """
    Writes the tables of all results as text files:

    <results_dir>/summary.txt       every accuracy: per seed, mean ± CI,
                                    LaTeX rows
    <results_dir>/full_pooled.txt   per-class report and confusion matrix
                                    of the main model, pooled over the splits
    <results_dir>/paired_tests.txt  paired comparisons with the main model,
                                    errors per makam group
    """
    pd.set_option("display.width", 200)

    results = load_results(results_dir)
    per_seed = per_seed_table(results)
    summary = summary_table(per_seed)

    def write_per_seed_and_summary(out_file, per_seed, summary):
        print("\nPer-seed results\n",
              per_seed.to_string(index=False), file=out_file)
        print("\nSummary (mean, SD, 95% CI half-width over seeds)\n",
              summary.to_string(index=False), file=out_file)

    # --- summary.txt
    path = os.path.join(results_dir, "summary.txt")
    with open(path, "w") as out_file:
        write_per_seed_and_summary(out_file, per_seed, summary)

        print("\nLaTeX rows (accuracy % mean ± 95% CI, macro-F1):",
              file=out_file)
        for row in latex_rows(summary):
            print(row, file=out_file)

    # --- full_pooled.txt
    main_results = load_results(results_dir, main_tag)
    main_per_seed = per_seed_table(main_results)
    main_summary = summary_table(main_per_seed)

    unperturbed_results = []
    for result in main_results:
        if not result.get("perturbation"):
            unperturbed_results.append(result)

    report, cm = pooled_report(unperturbed_results)

    path = os.path.join(results_dir, f"{main_tag}_pooled.txt")
    with open(path, "w") as out_file:
        write_per_seed_and_summary(out_file, main_per_seed, main_summary)

        print(f"\nPooled per-class report for '{main_tag}' "
              f"({len(unperturbed_results)} seeds)\n",
              report.round(3).to_string(), file=out_file)
        print("\nPooled confusion matrix\n", cm.to_string(), file=out_file)

        print("\nOff-diagonal entries (true -> predicted, count):",
              file=out_file)
        for true_makam, pred_makam, count in list_confusions(cm):
            print(f"  {true_makam:>16s} -> {pred_makam:<16s} {count}",
                  file=out_file)

    # --- paired_tests.txt
    path = os.path.join(results_dir, "paired_tests.txt")
    with open(path, "w") as out_file:
        write_per_seed_and_summary(out_file, per_seed, summary)

        print(f"\nPaired comparison with the proposed model ('{main_tag}', "
              "unperturbed) over the shared splits:\n"
              "delta_points = mean accuracy difference in percentage points; "
              "p = paired t-test\n",
              paired_tests(per_seed, main_tag).round(4).to_string(index=False),
              file=out_file)

        tags = set(per_seed["tag"])
        if "pc_acc" in tags and "pc_only" in tags:
            difference, p_value = paired_difference(
                per_seed, "pc_acc", "pc_only")
            print(f"\npc_acc - pc_only: {difference:+.2f} points, "
                  f"paired t-test p = {p_value:.2g}", file=out_file)

        print("\nErrors pooled over the five test subsets, per model "
              "(group = Ussak-Huseyni-Muhayyer):\n",
              group_errors(results).round(3).to_string(index=False),
              file=out_file)

    print("Wrote summary.txt, "
          f"{main_tag}_pooled.txt and paired_tests.txt to {results_dir}")
