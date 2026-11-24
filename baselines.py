import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import *
from utils import classification_metrics
from disk_utils import get_result_path, is_done, save_json
from dataset import (
    load_pieces,
    get_split,
    get_duration_values,
    komma_index
)

# Feature sets
#   hist53_dur    duration-weighted pitch-class histogram at komma resolution
#                 (53 bins, octave folded, rests excluded)
#   hist53_cnt    occurrence-count pitch-class histogram, 53 bins
#   hist_abs_dur  duration-weighted histogram of absolute pitches
#                 (octave kept, 5 octaves x 53 = 265 bins)
#   bigram        TF-IDF of absolute-pitch bigrams
FEATURE_KINDS = ["hist53_dur", "hist53_cnt", "hist_abs_dur", "bigram"]

# Classifiers
CLASSIFIER_NAMES = ["logreg", "svm_rbf", "rf", "knn"]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def pitch_histogram(piece, dur_values, weight_by_duration, keep_octave):
    """
    Normalised pitch histogram of one piece. Rests are left out.

    weight_by_duration: True  -> every note counts with its relative duration
                        False -> every note counts as 1
    keep_octave:        True  -> absolute pitches (265 bins)
                        False -> pitch classes, octave folded (53 bins)
    """
    if keep_octave:
        histogram = np.zeros(KOMMAS_PER_OCTAVE * NUM_OCTAVES)
    else:
        histogram = np.zeros(KOMMAS_PER_OCTAVE)

    for pc, acc, octave, dur, meas in piece["notes"]:
        komma = komma_index(pc, acc, octave)
        if komma is None:
            # rest
            continue

        if weight_by_duration:
            weight = dur_values[dur]
        else:
            weight = 1.0

        if keep_octave:
            bin_idx = komma - LOWEST_OCTAVE * KOMMAS_PER_OCTAVE
            if 0 <= bin_idx < len(histogram):
                histogram[bin_idx] += weight
        else:
            histogram[komma % KOMMAS_PER_OCTAVE] += weight

    total = histogram.sum()
    if total > 0:
        histogram = histogram / total

    return histogram


def pitch_bigram_text(piece):
    """
    The piece as a text of pitch bigrams, e.g. "252_261 261_252 252_R ..."
    Every pitch is its absolute komma index, a rest is "R".
    """
    tokens = []
    for pc, acc, octave, dur, meas in piece["notes"]:
        komma = komma_index(pc, acc, octave)
        if komma is None:
            tokens.append("R")
        else:
            tokens.append(str(komma))

    bigrams = []
    for first, second in zip(tokens[:-1], tokens[1:]):
        bigrams.append(f"{first}_{second}")

    return " ".join(bigrams)


def build_features(pieces, indices, dur_values, feature_kind):
    """
    Features of the pieces with the given indices.
    returns a matrix (histograms) or a list of texts (bigrams)
    """
    if feature_kind == "bigram":
        return [pitch_bigram_text(pieces[i]) for i in indices]

    if feature_kind == "hist53_dur":
        weight_by_duration, keep_octave = True, False
    elif feature_kind == "hist53_cnt":
        weight_by_duration, keep_octave = False, False
    elif feature_kind == "hist_abs_dur":
        weight_by_duration, keep_octave = True, True
    else:
        raise ValueError(f"Unknown feature kind: {feature_kind}")

    histograms = []
    for i in indices:
        histograms.append(pitch_histogram(
            pieces[i], dur_values, weight_by_duration, keep_octave))

    return np.vstack(histograms)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def get_classifier_grid(classifier_name, scale_features=True):
    """
    The candidate settings of one classifier.
    returns List[(parameter name, parameter value, classifier)]

    scale_features: False leaves out the StandardScaler (for sparse TF-IDF)
    """
    def with_scaler(classifier):
        if scale_features:
            return make_pipeline(StandardScaler(), classifier)
        return classifier

    grid = []

    if classifier_name == "logreg":
        for C in [0.01, 0.1, 1, 10, 100]:
            classifier = LogisticRegression(C=C, max_iter=5000)
            grid.append(("C", C, with_scaler(classifier)))

    elif classifier_name == "svm_rbf":
        for C in [0.1, 1, 10, 100]:
            for gamma in ["scale", 0.01, 0.001]:
                classifier = SVC(C=C, gamma=gamma)
                grid.append(("C,gamma", f"{C},{gamma}", with_scaler(classifier)))

    elif classifier_name == "rf":
        for min_samples_leaf in [1, 2, 5]:
            classifier = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=min_samples_leaf,
                random_state=0,
                n_jobs=-1
            )
            # a random forest does not need scaled features
            grid.append(("min_samples_leaf", min_samples_leaf, classifier))

    elif classifier_name == "knn":
        for k in [1, 3, 5, 9, 15]:
            classifier = KNeighborsClassifier(n_neighbors=k)
            grid.append(("k", k, with_scaler(classifier)))

    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return grid


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_baseline(
    feature_kind,
    classifier_name,
    seed,
    pieces,
    labels,
    dur_values,
    results_dir=RESULTS_DIR,
    split_dir=SPLIT_DIR,
    skip_if_done=True
):
    """
    Trains one baseline on the split of `seed` and tests it once.

    Every candidate setting of the classifier is fitted on the training
    subset. The one with the highest validation accuracy is tested.

    The results are written to
    <results_dir>/<feature_kind>__<classifier_name>/seed<seed>.json
    """
    tag = f"{feature_kind}__{classifier_name}"

    if skip_if_done and is_done(tag, seed, None, results_dir):
        print(f"[{tag} s{seed}] already done, skipping")
        return None

    train_idx, val_idx, test_idx = get_split(pieces, labels, seed, split_dir)
    y = np.asarray(labels)

    X_train = build_features(pieces, train_idx, dur_values, feature_kind)
    X_val = build_features(pieces, val_idx, dur_values, feature_kind)
    X_test = build_features(pieces, test_idx, dur_values, feature_kind)

    if feature_kind == "bigram":
        vectorizer = TfidfVectorizer(
            token_pattern=r"\S+", min_df=2, sublinear_tf=True)
        X_train = vectorizer.fit_transform(X_train)
        X_val = vectorizer.transform(X_val)
        X_test = vectorizer.transform(X_test)
        # TF-IDF features are sparse: no scaler
        grid = get_classifier_grid(classifier_name, scale_features=False)
    else:
        grid = get_classifier_grid(classifier_name, scale_features=True)

    # select the setting with the highest validation accuracy
    best_val_acc = None
    best_param_name = None
    best_param_value = None
    best_classifier = None

    for param_name, param_value, classifier in grid:
        classifier.fit(X_train, y[train_idx])
        val_acc = accuracy_score(y[val_idx], classifier.predict(X_val))

        if best_val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_param_name = param_name
            best_param_value = param_value
            best_classifier = classifier

    test_labels = y[test_idx].tolist()
    test_preds = best_classifier.predict(X_test).tolist()
    test_metrics = classification_metrics(test_labels, test_preds)

    result = {
        "tag": tag,
        "seed": seed,
        "perturbation": None,
        "config": {
            "features": feature_kind,
            "classifier": classifier_name,
            "selected": {best_param_name: str(best_param_value)}
        },
        "best_epoch": None,
        "history": [],
        "train_seconds": 0,
        "n_params": None,
        "val_accuracy": best_val_acc,
        "test": test_metrics,
        "test_files": [pieces[i]["f_name"] for i in test_idx],
        "test_labels": test_labels,
        "test_preds": test_preds,
        "test_lengths": [len(pieces[i]["notes"]) for i in test_idx]
    }
    save_json(result, get_result_path(tag, seed, None, results_dir))

    print(
        f"[{tag} s{seed}] "
        f"Val Acc: {best_val_acc:.4f} ({best_param_name}={best_param_value}) | "
        f"Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f}"
    )

    return result


def run_baselines(
    seeds=SEEDS,
    feature_kinds=FEATURE_KINDS,
    classifier_names=CLASSIFIER_NAMES,
    results_dir=RESULTS_DIR,
    split_dir=SPLIT_DIR,
    skip_if_done=True
):
    """
    Runs every combination of feature set and classifier on every split.
    """
    pieces, labels, vocab = load_pieces()
    dur_values = get_duration_values(vocab)

    for seed in seeds:
        for feature_kind in feature_kinds:
            for classifier_name in classifier_names:
                # random forest and k-NN are slow and uninformative on
                # sparse TF-IDF features: bigrams use logreg and SVM only
                if feature_kind == "bigram" and classifier_name in ["rf", "knn"]:
                    continue

                run_baseline(
                    feature_kind,
                    classifier_name,
                    seed,
                    pieces,
                    labels,
                    dur_values,
                    results_dir=results_dir,
                    split_dir=split_dir,
                    skip_if_done=skip_if_done
                )
