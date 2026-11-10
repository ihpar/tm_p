"""
BiLSTM sequence classifier with selectable pooling and feature subsets.

Changes w.r.t. the original submission:
  * padded positions are excluded from the recurrent computation with
    pack_padded_sequence / pad_packed_sequence (previously only the attention
    softmax was masked, so the backward LSTM direction saw padding tokens);
  * `pooling` selects how the sequence is summarised:
        "attn"  masked additive attention (the proposed model)
        "mean"  masked mean pooling
        "max"   masked max pooling
        "last"  concatenation of the final forward and first backward state
  * `features` selects which symbolic attributes are embedded; this is used
    for the feature-ablation experiments (e.g. no duration, no measure
    position, pitch-only).

The forward interface is unchanged: model(batch) -> (logits, attn_weights).
For non-attention pooling, attn_weights is the (masked) uniform distribution
so that downstream code keeps working.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ALL_FEATURES = ("pc", "acc", "oct", "dur", "meas")
FEATURE_TO_BATCH_KEY = {"pc": "pcs", "acc": "accs", "oct": "octs", "dur": "durs", "meas": "meas"}
DEFAULT_VOCAB_SIZES = {"pc": 9, "acc": 10, "oct": 7, "dur": 33, "meas": 4}


class MaskedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, mask):
        scores = self.attn(lstm_out).squeeze(-1)            # (B, T)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)                    # (B, T)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, pc_vocab=9, acc_vocab=10, oct_vocab=7, dur_vocab=33, meas_vocab=4,
                 emb_dim=32, lstm_hidden=128, num_layers=1, num_classes=12, dropout=0.3,
                 pooling="attn", features=ALL_FEATURES, packed=True):
        super().__init__()
        assert pooling in ("attn", "mean", "max", "last")
        self.pooling = pooling
        self.features = tuple(features)
        self.packed = packed
        vocab_sizes = {"pc": pc_vocab, "acc": acc_vocab, "oct": oct_vocab,
                       "dur": dur_vocab, "meas": meas_vocab}

        self.embeddings = nn.ModuleDict({
            f: nn.Embedding(vocab_sizes[f], emb_dim, padding_idx=0) for f in self.features
        })
        total_emb_dim = emb_dim * len(self.features)

        self.bilstm = nn.LSTM(
            input_size=total_emb_dim, hidden_size=lstm_hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = MaskedAttention(2 * lstm_hidden) if pooling == "attn" else None
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )
        self.lstm_hidden = lstm_hidden

    def encode(self, batch):
        mask = batch["mask"]
        x = torch.cat([self.embeddings[f](batch[FEATURE_TO_BATCH_KEY[f]])
                       for f in self.features], dim=-1)
        if self.packed:
            lengths = batch["lengths"].cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.bilstm(packed)
            lstm_out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        else:
            lstm_out, _ = self.bilstm(x)
        return lstm_out, mask

    def pool(self, lstm_out, mask, lengths):
        maskf = mask.unsqueeze(-1).float()
        if self.pooling == "attn":
            return self.attention(lstm_out, mask)
        uniform = maskf.squeeze(-1) / maskf.squeeze(-1).sum(1, keepdim=True)
        if self.pooling == "mean":
            context = (lstm_out * maskf).sum(1) / maskf.sum(1)
        elif self.pooling == "max":
            context = lstm_out.masked_fill(maskf == 0, float("-inf")).max(1).values
        else:  # last: forward state at t=len-1, backward state at t=0
            H = self.lstm_hidden
            idx = (lengths - 1).to(lstm_out.device).view(-1, 1, 1).expand(-1, 1, H)
            fwd_last = lstm_out[:, :, :H].gather(1, idx).squeeze(1)
            bwd_first = lstm_out[:, 0, H:]
            context = torch.cat([fwd_last, bwd_first], dim=-1)
        return context, uniform

    def forward(self, batch):
        lstm_out, mask = self.encode(batch)
        context, attn_weights = self.pool(lstm_out, mask, batch["lengths"])
        logits = self.classifier(context)
        return logits, attn_weights


if __name__ == "__main__":
    from dataset import build_dataloaders
    train_loader, _, _, _ = build_dataloaders(batch_size=4, seed=0)
    batch = next(iter(train_loader))
    for pooling in ("attn", "mean", "max", "last"):
        model = BiLSTMAttentionClassifier(pooling=pooling)
        logits, w = model(batch)
        print(pooling, logits.shape, w.shape, w.sum(1), (w * (1 - batch["mask"])).max().item())
    # padding independence check: same piece, different padding, same logits
    model = BiLSTMAttentionClassifier().eval()
    with torch.no_grad():
        full, _ = model(batch)
        single = {k: (v[:1, :batch["lengths"][0]] if v.dim() == 2 else v[:1])
                  for k, v in batch.items() if k != "f_names"}
        alone, _ = model(single)
    print("max |logit diff| for longest piece with/without batch padding:",
          (full[0] - alone[0]).abs().max().item())
    model_unpacked = BiLSTMAttentionClassifier(packed=False).eval()
    model_unpacked.load_state_dict(model.state_dict())
    with torch.no_grad():
        full_u, _ = model_unpacked(batch)
        short_i = batch["lengths"].argmin().item()
        single = {k: (v[short_i:short_i + 1, :batch["lengths"][short_i]] if v.dim() == 2
                      else v[short_i:short_i + 1]) for k, v in batch.items() if k != "f_names"}
        alone_u, _ = model_unpacked(single)
        alone_p, _ = model(single)
        full_p, _ = model(batch)
    print("shortest piece, UNPACKED: |logit diff| padded vs alone =",
          (full_u[short_i] - alone_u[0]).abs().max().item())
    print("shortest piece, PACKED  : |logit diff| padded vs alone =",
          (full_p[short_i] - alone_p[0]).abs().max().item())
