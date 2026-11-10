"""
Faithfulness tests for the attention weights (Reviewer 1, Reviewer 2 item 10).

For every test piece of a trained model we delete a fraction k of its notes and
re-classify the shortened piece. Four deletion orders are compared:
    attn_top     notes with the highest attention first
    attn_bottom  notes with the lowest attention first
    random       uniformly random notes (averaged over `--n_random` draws)
    grad_top     notes with the highest gradient x input saliency first
If attention identifies the notes the prediction depends on, deleting attn_top
notes should hurt accuracy much more than deleting random or attn_bottom notes,
and about as much as grad_top.  We also report the per-piece Spearman rank
correlation between attention and saliency.

python faithfulness.py --tag full --seeds 0 1 2 3 4
-> results/faithfulness/<tag>_seed<k>.json, results/faithfulness/<tag>_summary.csv,
   figs_rev/faithfulness_<tag>.pdf
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from dataset import (load_pieces, read_split_files, MakamDataset, collate_fn, seed_everything)
from bilstm import BiLSTMAttentionClassifier, DEFAULT_VOCAB_SIZES


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    features = tuple(f for f in a["features"].split(",") if f)
    model = BiLSTMAttentionClassifier(
        **{f"{k}_vocab": v for k, v in DEFAULT_VOCAB_SIZES.items()},
        emb_dim=a["emb_dim"], lstm_hidden=a["hidden"], num_layers=a["num_layers"],
        dropout=a["dropout"], pooling=a["pooling"], features=features,
        packed=not a.get("no_pack", False)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, a


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != "f_names"}


def predict_pieces(model, pieces, labels, device, batch_size=32):
    """Returns predictions for a list of pieces (no perturbation)."""
    ds = MakamDataset(pieces, labels)
    preds = {}
    for i in range(0, len(ds), batch_size):
        items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        batch = collate_fn(items)
        with torch.no_grad():
            logits, _ = model(to_device(batch, device))
        for name, p in zip(batch["f_names"].tolist(), logits.argmax(1).cpu().tolist()):
            preds[name] = p
    return preds


def attention_and_saliency(model, pieces, labels, device):
    """Per piece: attention weights and gradient x input saliency (both length T)."""
    ds = MakamDataset(pieces, labels)
    out = {}
    model.keep_input_grad = True
    # cuDNN only supports RNN backward in training mode; the LSTM has no dropout
    # (single layer), so switching only this module changes nothing numerically.
    model.bilstm.train()
    for j in range(len(ds)):
        batch = collate_fn([ds[j]])
        b = to_device(batch, device)
        model.zero_grad()
        logits, attn = model(b)
        pred = logits.argmax(1)
        logits[0, pred].sum().backward()
        x = model.last_input
        sal = (x.grad * x).sum(-1).abs()[0].detach().cpu().numpy()
        out[batch["f_names"][0]] = {
            "pred": int(pred.item()),
            "attn": attn[0].detach().cpu().numpy(),
            "sal": sal,
        }
    model.keep_input_grad = False
    model.bilstm.eval()
    model.zero_grad()
    return out


def delete_notes(piece, order, k_frac, rng=None):
    """Return a copy of the piece with the first k_frac*T notes of `order` removed."""
    T = len(piece["notes"])
    n_del = int(round(k_frac * T))
    n_del = min(n_del, T - 1)                     # keep at least one note
    if order is None:                              # random
        drop = set(rng.choice(T, size=n_del, replace=False).tolist())
    else:
        drop = set(order[:n_del].tolist())
    return {"f_name": piece["f_name"], "notes": [n for i, n in enumerate(piece["notes"]) if i not in drop]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40])
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--split_dir", default="splits")
    ap.add_argument("--fig_dir", default="figs_rev")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)

    pieces_all, labels_all, _ = load_pieces()
    out_dir = os.path.join(args.results_dir, "faithfulness")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    rows = []

    for seed in args.seeds:
        seed_everything(seed)
        ckpt = os.path.join(args.results_dir, args.tag, f"seed{seed}.pt")
        model, margs = load_model(ckpt, device)
        tr, va, te = read_split_files(pieces_all, seed, args.split_dir)
        pieces = [pieces_all[i] for i in te]
        labels = [labels_all[i] for i in te]
        y = np.array(labels)

        info = attention_and_saliency(model, pieces, labels, device)
        base_pred = np.array([info[p["f_name"]]["pred"] for p in pieces])
        base_acc = (base_pred == y).mean()
        rhos = [spearmanr(info[p["f_name"]]["attn"], info[p["f_name"]]["sal"]).correlation
                for p in pieces if len(p["notes"]) > 2]
        rho_mean = float(np.nanmean(rhos))
        print(f"[{args.tag} s{seed}] base acc {base_acc:.4f} | Spearman(attn, grad x input) "
              f"mean {rho_mean:.3f} (median {np.nanmedian(rhos):.3f})")

        rng = np.random.RandomState(seed)
        orders = {
            "attn_top": {p["f_name"]: np.argsort(-info[p["f_name"]]["attn"]) for p in pieces},
            "attn_bottom": {p["f_name"]: np.argsort(info[p["f_name"]]["attn"]) for p in pieces},
            "grad_top": {p["f_name"]: np.argsort(-info[p["f_name"]]["sal"]) for p in pieces},
            "random": None,
        }
        result = {"tag": args.tag, "seed": seed, "base_accuracy": base_acc,
                  "spearman_attn_saliency_mean": rho_mean,
                  "spearman_attn_saliency_median": float(np.nanmedian(rhos)),
                  "spearman_per_piece": [float(r) for r in rhos], "curves": {}}
        for frac in args.fracs:
            for name, order in orders.items():
                reps = args.n_random if name == "random" else 1
                accs, flips = [], []
                for _ in range(reps):
                    pert = [delete_notes(p, None if order is None else order[p["f_name"]], frac, rng)
                            for p in pieces]
                    preds = predict_pieces(model, pert, labels, device)
                    pv = np.array([preds[p["f_name"]] for p in pieces])
                    accs.append((pv == y).mean())
                    flips.append((pv != base_pred).mean())
                acc, flip = float(np.mean(accs)), float(np.mean(flips))
                result["curves"][f"{name}@{frac}"] = {"accuracy": acc, "flip_rate": flip}
                rows.append({"tag": args.tag, "seed": seed, "order": name, "frac": frac,
                             "accuracy": acc, "flip_rate": flip, "base_accuracy": base_acc})
                print(f"    delete {int(frac * 100):2d}% {name:12s} acc {acc:.4f} flip {flip:.3f}")
        with open(os.path.join(out_dir, f"{args.tag}_seed{seed}.json"), "w") as f:
            json.dump(result, f)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{args.tag}_curves.csv"), index=False)
    summ = df.groupby(["order", "frac"]).agg(acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                                              flip_mean=("flip_rate", "mean"), flip_sd=("flip_rate", "std"),
                                              n=("seed", "count")).reset_index()
    summ.to_csv(os.path.join(out_dir, f"{args.tag}_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n", summ.to_string(index=False))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    base = df["base_accuracy"].mean()
    labels_map = {"attn_top": "highest attention first", "grad_top": "highest saliency first",
                  "random": "random", "attn_bottom": "lowest attention first"}
    for name in ("attn_top", "grad_top", "random", "attn_bottom"):
        s = summ[summ["order"] == name].sort_values("frac")
        x = [0] + (s["frac"] * 100).tolist()
        yv = [base] + s["acc_mean"].tolist()
        err = [0] + s["acc_sd"].fillna(0).tolist()
        ax.errorbar(x, np.array(yv) * 100, yerr=np.array(err) * 100, marker="o", capsize=3, label=labels_map[name])
    ax.set_xlabel("Notes deleted per piece (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Deletion test ({args.tag}, {len(args.seeds)} seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, f"faithfulness_{args.tag}.pdf"))
    fig.savefig(os.path.join(args.fig_dir, f"faithfulness_{args.tag}.png"), dpi=200)
    print("figure saved to", args.fig_dir)


if __name__ == "__main__":
    main()
