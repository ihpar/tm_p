import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from constants import *


class MaskedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, mask):
        """
        lstm_out: (B: batch size, T: time steps, H: hidden dim)
        mask:     (B, T)   1 = valid (real), 0 = padding (fake)
        """
        # (B, T, 1)
        scores = self.attn(lstm_out)

        # Mask padding tokens
        scores = scores.squeeze(-1)          # (B, T)
        scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=1)   # (B, T)

        # Weighted sum
        context = torch.sum(
            lstm_out * weights.unsqueeze(-1),
            dim=1
        )  # (B, H)

        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        pc_vocab=PC_VOCAB_SIZE,
        acc_vocab=ACC_VOCAB_SIZE,
        oct_vocab=OCT_VOCAB_SIZE,
        dur_vocab=DUR_VOCAB_SIZE,
        meas_vocab=MEAS_VOCAB_SIZE,
        int_vocab=INTERVAL_VOCAB_SIZE,
        emb_dim=EMB_DIM,
        lstm_hidden=LSTM_HIDDEN,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        pooling="attn",
        features=ALL_FEATURES,
        packed=True
    ):
        """
        pooling: how the sequence of LSTM states is summarised
            "attn"  masked attention (the proposed model)
            "mean"  masked mean pooling
            "max"   masked max pooling
            "last"  last forward state + first backward state

        features: the note attributes that are embedded, a subset of
            ["pc", "acc", "oct", "dur", "meas", "int"]
            ("int" is the interval representation)

        packed: if True, the padding is left out of the recurrent
            computation (pack_padded_sequence). If False, the LSTM also
            runs over the padding, as in the preliminary version.
        """
        super().__init__()

        assert pooling in ["attn", "mean", "max", "last"]

        self.pooling = pooling
        self.features = list(features)
        self.packed = packed
        self.lstm_hidden = lstm_hidden

        vocab_sizes = {
            "pc": pc_vocab,
            "acc": acc_vocab,
            "oct": oct_vocab,
            "dur": dur_vocab,
            "meas": meas_vocab,
            "int": int_vocab
        }

        # one embedding table per selected feature
        self.embeddings = nn.ModuleDict()
        for feature in self.features:
            self.embeddings[feature] = nn.Embedding(
                vocab_sizes[feature], emb_dim, padding_idx=PAD_ID)

        total_emb_dim = emb_dim * len(self.features)

        if num_layers > 1:
            lstm_dropout = dropout
        else:
            lstm_dropout = 0.0

        self.bilstm = nn.LSTM(
            input_size=total_emb_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )

        if pooling == "attn":
            self.attention = MaskedAttention(hidden_dim=2 * lstm_hidden)
        else:
            self.attention = None

        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

        # Set keep_input_grad to True to keep the gradient w.r.t. the
        # embedded input, which the saliency of the deletion tests needs.
        self.keep_input_grad = False
        self.last_input = None

    def embed(self, batch):
        """
        returns the concatenated feature embeddings, (B, T, total_emb_dim)
        """
        feature_embeddings = []
        for feature in self.features:
            batch_key = FEATURE_TO_BATCH_KEY[feature]
            feature_embeddings.append(self.embeddings[feature](batch[batch_key]))

        x = torch.cat(feature_embeddings, dim=-1)

        if self.keep_input_grad:
            x.retain_grad()
            self.last_input = x

        return x

    def encode(self, x, lengths):
        """
        x:       (B, T, total_emb_dim)
        lengths: (B,)  true lengths of the pieces
        returns the LSTM states, (B, T, 2 * lstm_hidden)
        """
        if not self.packed:
            lstm_out, _ = self.bilstm(x)
            return lstm_out

        packed_x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed_x)
        lstm_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1)
        )

        return lstm_out

    def pool(self, lstm_out, mask, lengths):
        """
        Summarises the LSTM states of each piece in one vector.
        returns context (B, 2 * lstm_hidden), weights (B, T)

        For "mean", "max" and "last" pooling the weights are uniform over
        the real tokens, so that the output has the same form as attention.
        """
        if self.pooling == "attn":
            return self.attention(lstm_out, mask)

        mask_f = mask.unsqueeze(-1).float()   # (B, T, 1)

        uniform_weights = mask_f.squeeze(-1) / \
            mask_f.squeeze(-1).sum(1, keepdim=True)

        if self.pooling == "mean":
            context = (lstm_out * mask_f).sum(1) / mask_f.sum(1)

        elif self.pooling == "max":
            masked_out = lstm_out.masked_fill(mask_f == 0, float("-inf"))
            context = masked_out.max(1).values

        else:
            # "last": forward state at the last real token,
            # backward state at the first token
            H = self.lstm_hidden
            last_idx = (lengths - 1).to(lstm_out.device)
            last_idx = last_idx.view(-1, 1, 1).expand(-1, 1, H)

            forward_last = lstm_out[:, :, :H].gather(1, last_idx).squeeze(1)
            backward_first = lstm_out[:, 0, H:]

            context = torch.cat([forward_last, backward_first], dim=-1)

        return context, uniform_weights

    def forward(self, batch):
        mask = batch["mask"]
        lengths = batch["lengths"]

        x = self.embed(batch)

        lstm_out = self.encode(x, lengths)

        context, attn_weights = self.pool(lstm_out, mask, lengths)

        logits = self.classifier(context)

        return logits, attn_weights


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    from dataset import build_dataloaders

    train_loader, _, _, _ = build_dataloaders(batch_size=4)
    batch = next(iter(train_loader))
    print(batch["pcs"].shape)

    for pooling in ["attn", "mean", "max", "last"]:
        model = BiLSTMAttentionClassifier(pooling=pooling)
        logits, attn_weights = model(batch)
        print(pooling)
        print(logits.size())
        print(attn_weights.size())
        print(attn_weights.sum(dim=1))
        print((attn_weights * (1 - batch["mask"])).max())

    check_padding_independence(batch)


def check_padding_independence(batch):
    """
    Classifies the shortest piece of a batch twice: inside the padded batch
    and alone (without padding), and prints the largest difference between
    the two outputs. With packing the padding has no effect.
    """
    shortest = batch["lengths"].argmin().item()
    length = batch["lengths"][shortest]

    # the same piece as a batch of its own, cut to its true length
    single = {}
    for key, value in batch.items():
        if key == "f_names":
            continue
        if value.dim() == 2:
            single[key] = value[shortest:shortest + 1, :length]
        else:
            single[key] = value[shortest:shortest + 1]

    model_packed = BiLSTMAttentionClassifier(packed=True).eval()
    model_unpacked = BiLSTMAttentionClassifier(packed=False).eval()
    model_unpacked.load_state_dict(model_packed.state_dict())

    with torch.no_grad():
        for name, model in [("packed", model_packed), ("unpacked", model_unpacked)]:
            logits_in_batch, _ = model(batch)
            logits_alone, _ = model(single)
            difference = (logits_in_batch[shortest] - logits_alone[0]).abs().max()
            print(f"{name}: max logit difference, padded vs alone = {difference.item():.2e}")


if __name__ == "__main__":
    main()
