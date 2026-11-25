import time
import torch
from sklearn.metrics import accuracy_score

from constants import *
from train import train_model, predict
from dataset import build_dataloaders
from bilstm import BiLSTMAttentionClassifier, count_parameters
from utils import (
    seed_everything,
    use_deterministic_kernels,
    classification_metrics
)
from disk_utils import (
    get_perturbation_name,
    get_checkpoint_path,
    get_result_path,
    is_done,
    load_checkpoint,
    get_model_config,
    save_json
)


def build_model(model_config, device):
    """
    model_config: dict with the keys emb_dim, lstm_hidden, num_layers,
                  dropout, pooling, features, packed
    """
    model = BiLSTMAttentionClassifier(
        pc_vocab=PC_VOCAB_SIZE,
        acc_vocab=ACC_VOCAB_SIZE,
        oct_vocab=OCT_VOCAB_SIZE,
        dur_vocab=DUR_VOCAB_SIZE,
        meas_vocab=MEAS_VOCAB_SIZE,
        int_vocab=INTERVAL_VOCAB_SIZE,
        emb_dim=model_config["emb_dim"],
        lstm_hidden=model_config["lstm_hidden"],
        num_layers=model_config["num_layers"],
        num_classes=NUM_CLASSES,
        dropout=model_config["dropout"],
        pooling=model_config["pooling"],
        features=model_config["features"],
        packed=model_config["packed"]
    )
    model.to(device)

    return model


def load_trained_model(tag, seed, device, results_dir=RESULTS_DIR):
    """
    Loads the trained model <results_dir>/<tag>/seed<seed>.pt
    returns model, checkpoint
    """
    checkpoint_path = get_checkpoint_path(tag, seed, results_dir)
    checkpoint = load_checkpoint(checkpoint_path, device)

    model_config = get_model_config(checkpoint)
    model = build_model(model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def run_experiment(
    tag,
    seed,
    device,
    features=ALL_FEATURES,
    pooling="attn",
    packed=True,
    train_shuffle=False,
    train_transpose_aug=0,
    emb_dim=EMB_DIM,
    lstm_hidden=LSTM_HIDDEN,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    scheduler_patience=SCHEDULER_PATIENCE,
    max_train_batches=None,
    results_dir=RESULTS_DIR,
    split_dir=SPLIT_DIR,
    skip_if_done=True
):
    """
    Trains one model on the split of `seed` and tests it once.

    tag:  name of the experiment. The results are written to
          <results_dir>/<tag>/seed<seed>.json (accuracies, predictions,
          training history) and seed<seed>.pt (trained model).
    seed: selects the split and seeds the training

    Model variants:
    features:            note attributes given to the model
    pooling:             "attn", "mean", "max" or "last"
    packed:              False runs the LSTM over the padding as well

    Variants of the training data:
    train_shuffle:       shuffle the note order of every training piece
    train_transpose_aug: random transposition in [-k, +k] kommas

    Training budget:
    num_epochs:          maximum number of epochs
    patience:            early stopping after this many epochs without a
                         lower validation loss
    scheduler_patience:  the learning rate is halved after this many epochs
                         without a lower validation loss

    max_train_batches:   batches per epoch, for quick tests (None = all)
    skip_if_done:        do nothing if the result file already exists

    returns the result dict
    """
    result_path = get_result_path(tag, seed, None, results_dir)
    checkpoint_path = get_checkpoint_path(tag, seed, results_dir)

    if skip_if_done and is_done(tag, seed, None, results_dir):
        print(f"[{tag} s{seed}] already done, skipping")
        return None

    seed_everything(seed)
    use_deterministic_kernels()

    train_loader, class_weights, val_loader, test_loader = build_dataloaders(
        batch_size=batch_size,
        seed=seed,
        split_dir=split_dir,
        train_shuffle=train_shuffle,
        train_transpose_aug=train_transpose_aug
    )

    model_config = {
        "emb_dim": emb_dim,
        "lstm_hidden": lstm_hidden,
        "num_layers": NUM_LAYERS,
        "dropout": dropout,
        "pooling": pooling,
        "features": list(features),
        "packed": packed
    }
    model = build_model(model_config, device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=scheduler_patience
    )

    start_time = time.time()

    history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        model_config,
        num_epochs=num_epochs,
        patience=patience,
        save_path=checkpoint_path,
        max_train_batches=max_train_batches,
        name=f"[{tag} s{seed}]"
    )

    # go back to the weights of the best epoch
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_labels, test_preds, test_f_names, test_lengths = predict(
        model, test_loader, device)
    val_labels, val_preds, _, _ = predict(model, val_loader, device)

    test_metrics = classification_metrics(test_labels, test_preds)

    result = {
        "tag": tag,
        "seed": seed,
        "perturbation": None,
        "config": {
            "model": model_config,
            "train_shuffle": train_shuffle,
            "train_transpose_aug": train_transpose_aug,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "patience": patience,
            "scheduler_patience": scheduler_patience,
            "max_train_batches": max_train_batches
        },
        "features": list(features),
        "n_params": count_parameters(model),
        "best_epoch": best_epoch,
        "device": str(device),
        "train_seconds": time.time() - start_time,
        "history": history,
        "val_accuracy": accuracy_score(val_labels, val_preds),
        "test": test_metrics,
        "test_files": test_f_names,
        "test_labels": test_labels,
        "test_preds": test_preds,
        "test_lengths": test_lengths
    }
    save_json(result, result_path)

    print(
        f"[{tag} s{seed}] "
        f"Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Best epoch: {best_epoch} | "
        f"Saved {result_path}"
    )

    return result


def evaluate_trained_model(
    tag,
    seed,
    device,
    test_shuffle=False,
    test_transpose=0,
    results_dir=RESULTS_DIR,
    split_dir=SPLIT_DIR,
    skip_if_done=True
):
    """
    Tests an already trained model on perturbed test pieces.

    tag, seed:      the trained model <results_dir>/<tag>/seed<seed>.pt
    test_shuffle:   shuffle the note order of every test piece
    test_transpose: transpose every test piece by this many kommas

    The results are written to
    <results_dir>/<tag>/seed<seed>__<perturbation>.json

    returns the result dict
    """
    perturbation = get_perturbation_name(test_shuffle, test_transpose)
    result_path = get_result_path(tag, seed, perturbation, results_dir)

    if skip_if_done and is_done(tag, seed, perturbation, results_dir):
        print(f"[{tag} s{seed} {perturbation}] already done, skipping")
        return None

    seed_everything(seed)
    use_deterministic_kernels()

    _, _, val_loader, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        seed=seed,
        split_dir=split_dir,
        test_shuffle=test_shuffle,
        test_transpose=test_transpose
    )

    start_time = time.time()

    model, checkpoint = load_trained_model(tag, seed, device, results_dir)
    best_epoch = checkpoint["epoch"]

    test_labels, test_preds, test_f_names, test_lengths = predict(
        model, test_loader, device)
    val_labels, val_preds, _, _ = predict(model, val_loader, device)

    test_metrics = classification_metrics(test_labels, test_preds)

    result = {
        "tag": tag,
        "seed": seed,
        "perturbation": perturbation,
        "config": {
            "model": get_model_config(checkpoint),
            "test_shuffle": test_shuffle,
            "test_transpose": test_transpose
        },
        "features": list(model.features),
        "n_params": count_parameters(model),
        "best_epoch": best_epoch,
        "device": str(device),
        "train_seconds": time.time() - start_time,
        "history": [],
        "val_accuracy": accuracy_score(val_labels, val_preds),
        "test": test_metrics,
        "test_files": test_f_names,
        "test_labels": test_labels,
        "test_preds": test_preds,
        "test_lengths": test_lengths
    }
    save_json(result, result_path)

    print(
        f"[{tag} s{seed} {perturbation}] "
        f"Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Saved {result_path}"
    )

    return result
