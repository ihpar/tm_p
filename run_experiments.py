"""
Single entry point for all neural experiments of the revision.

Examples
--------
# E1: proposed model, seed 0
python run_experiments.py --tag full --seed 0

# E1 for all seeds
for s in 0 1 2 3 4; do python run_experiments.py --tag full --seed $s; done

# E3: pooling baselines
python run_experiments.py --tag mean_pool --pooling mean --seed 0

# E4: feature ablations
python run_experiments.py --tag no_dur  --features pc,acc,oct,meas --seed 0
python run_experiments.py --tag no_meas --features pc,acc,oct,dur  --seed 0
python run_experiments.py --tag pc_acc  --features pc,acc            --seed 0
python run_experiments.py --tag pc_only --features pc                --seed 0

# E5b / E6: evaluate an already trained model on a perturbed test set
python run_experiments.py --tag full --seed 0 --eval_only --test_transpose 9
python run_experiments.py --tag full --seed 0 --eval_only --test_shuffle

Every run writes results/<tag>/seed<k>[__<perturbation>].json and (when training)
results/<tag>/seed<k>.pt.  Aggregate with aggregate_results.py.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

from dataset import build_dataloaders, seed_everything, MAKAM_VOCAB
from bilstm import BiLSTMAttentionClassifier, ALL_FEATURES, DEFAULT_VOCAB_SIZES

MAKAM_NAMES = [k for k, _ in sorted(MAKAM_VOCAB.items(), key=lambda kv: kv[1])]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="experiment name (sub-folder of results/)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--split_dir", default="splits")
    p.add_argument("--vocab", default="vocab.pkl")
    p.add_argument("--dataset", default="dataset.pkl")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--num_workers", type=int, default=0)
    # model
    p.add_argument("--pooling", default="attn", choices=["attn", "mean", "max", "last"])
    p.add_argument("--features", default=",".join(ALL_FEATURES),
                   help="comma-separated subset of pc,acc,oct,dur,meas")
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no_pack", action="store_true", help="reproduce the original (unpacked) LSTM")
    # training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--sched_patience", type=int, default=3)
    p.add_argument("--sched_factor", type=float, default=0.5)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--no_class_weights", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=0, help="debug: limit batches/epoch")
    # evaluation-only / perturbations of the test set
    p.add_argument("--eval_only", action="store_true", help="load results/<tag>/seed<k>.pt and evaluate")
    p.add_argument("--test_shuffle", action="store_true", help="shuffle note order in test pieces")
    p.add_argument("--test_transpose", type=int, default=0, help="transpose test pieces by k kommas")
    p.add_argument("--test_drop_rests", action="store_true")
    p.add_argument("--save_attention", action="store_true",
                   help="store per-piece attention weights of the test set in the json")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, clip=5.0, max_batches=0):
    train = optimizer is not None
    model.train(train)
    tot_loss, tot_correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if k != "f_names"}
            logits, _ = model(batch)
            loss = criterion(logits, batch["labels"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            bs = batch["labels"].size(0)
            tot_loss += loss.item() * bs
            tot_correct += (logits.argmax(1) == batch["labels"]).sum().item()
            n += bs
    return tot_loss / n, tot_correct / n


@torch.no_grad()
def predict(model, loader, device, save_attention=False):
    model.eval()
    labels, preds, names, lengths, attn = [], [], [], [], []
    for batch in loader:
        f_names = batch["f_names"]
        batch = {k: v.to(device) for k, v in batch.items() if k != "f_names"}
        logits, w = model(batch)
        labels.extend(batch["labels"].cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
        names.extend(f_names.tolist())
        lengths.extend(batch["lengths"].cpu().tolist())
        if save_attention:
            for row, L in zip(w.cpu().numpy(), batch["lengths"].cpu().tolist()):
                attn.append(row[:L].tolist())
    return labels, preds, names, lengths, attn


def metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "report": classification_report(labels, preds, labels=list(range(len(MAKAM_NAMES))),
                                        target_names=MAKAM_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(MAKAM_NAMES)))).tolist(),
    }


def main():
    args = get_args()
    seed_everything(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)
    features = tuple(f.strip() for f in args.features.split(",") if f.strip())

    perturbation = {}
    if args.test_shuffle:
        perturbation["shuffle_order"] = True
    if args.test_transpose:
        perturbation["transpose"] = args.test_transpose
    if args.test_drop_rests:
        perturbation["drop_rests"] = True
    pert_name = "__".join(
        ([f"shuffle"] if args.test_shuffle else []) +
        ([f"transpose{args.test_transpose:+d}"] if args.test_transpose else []) +
        ([f"norests"] if args.test_drop_rests else []))

    train_loader, class_weights, val_loader, test_loader = build_dataloaders(
        args.vocab, args.dataset, batch_size=args.batch_size, seed=args.seed,
        split_dir=args.split_dir, test_perturbation=perturbation or None,
        num_workers=args.num_workers)

    model = BiLSTMAttentionClassifier(
        **{f"{k}_vocab": v for k, v in DEFAULT_VOCAB_SIZES.items()},
        emb_dim=args.emb_dim, lstm_hidden=args.hidden, num_layers=args.num_layers,
        dropout=args.dropout, pooling=args.pooling, features=features,
        packed=not args.no_pack).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    out_dir = os.path.join(args.results_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"seed{args.seed}.pt")
    json_path = os.path.join(out_dir, f"seed{args.seed}" + (f"__{pert_name}" if pert_name else "") + ".json")

    weights = None if args.no_class_weights else class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    history = []
    t0 = time.time()

    if args.eval_only:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = ckpt["epoch"]
        print(f"loaded {ckpt_path} (epoch {best_epoch})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.sched_factor, patience=args.sched_patience)
        best_val, best_epoch, bad = float("inf"), 0, 0
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                        args.clip, args.max_train_batches)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step(va_loss)
            lr = optimizer.param_groups[0]["lr"]
            history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": va_loss, "val_acc": va_acc, "lr": lr})
            print(f"[{args.tag} s{args.seed}] ep {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} "
                  f"| val {va_loss:.4f}/{va_acc:.4f} | lr {lr:.1e}")
            if va_loss < best_val:
                best_val, best_epoch, bad = va_loss, epoch, 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_loss": va_loss, "args": vars(args)}, ckpt_path)
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stopping")
                    break
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])

    labels, preds, names, lengths, attn = predict(model, test_loader, device, args.save_attention)
    val_labels, val_preds, *_ = predict(model, val_loader, device)
    m = metrics(labels, preds)
    result = {
        "tag": args.tag, "seed": args.seed, "perturbation": pert_name or None,
        "args": vars(args), "features": features, "n_params": n_params,
        "best_epoch": best_epoch, "device": str(device),
        "train_seconds": time.time() - t0, "history": history,
        "val_accuracy": accuracy_score(val_labels, val_preds),
        "test": m,
        "test_files": names, "test_labels": labels, "test_preds": preds, "test_lengths": lengths,
    }
    if args.save_attention:
        result["test_attention"] = attn
    with open(json_path, "w") as f:
        json.dump(result, f)
    print(f"[{args.tag} s{args.seed}{' ' + pert_name if pert_name else ''}] "
          f"TEST acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"best epoch {best_epoch} | saved {json_path}")


if __name__ == "__main__":
    main()
