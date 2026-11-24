import torch

from constants import *
from early_stopper import EarlyStopper
from disk_utils import save_checkpoint
from utils import move_batch_to_device


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    max_batches=None
):
    """
    max_batches: stop the epoch after this many batches (for quick tests).
                 None trains on every batch.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        logits, _ = model(batch)

        loss = criterion(logits, batch["labels"])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRAD_CLIP_NORM)

        optimizer.step()

        total_loss += loss.item() * batch["labels"].size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        logits, _ = model(batch)

        loss = criterion(logits, batch["labels"])

        total_loss += loss.item() * batch["labels"].size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def predict(model, dataloader, device):
    """
    returns all_labels, all_preds, all_f_names, all_lengths (python lists)
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_f_names = []
    all_lengths = []

    for batch in dataloader:
        f_names = batch["f_names"]
        batch = move_batch_to_device(batch, device)

        logits, _ = model(batch)
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(batch["labels"].cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_f_names.extend(f_names.tolist())
        all_lengths.extend(batch["lengths"].cpu().tolist())

    return all_labels, all_preds, all_f_names, all_lengths


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    model_config,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    save_path="best_model.pt",
    max_train_batches=None,
    name=""
):
    """
    Trains the model and saves the weights of the epoch with the lowest
    validation loss to save_path. Stops early after `patience` epochs
    without improvement.

    model_config: settings of the model, saved with the weights
    name:         printed in front of every line

    returns history, best_epoch
    history: List[dict], one entry per epoch
    """
    early_stopper = EarlyStopper(patience=patience)
    history = []
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            max_batches=max_train_batches
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr
        })

        print(
            f"{name} Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {lr:.1e}"
        )

        if early_stopper.is_improvement(val_loss):
            best_epoch = epoch
            save_checkpoint(model, save_path, epoch, val_loss, model_config)

        elif early_stopper.early_stop():
            print(f"{name} Early stopping triggered!")
            break

    return history, best_epoch
