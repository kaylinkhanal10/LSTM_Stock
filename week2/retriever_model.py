import torch
import torch.nn as nn
import re

# ---------------- Tokenizer ----------------
class SimpleTokenizer:
    def __init__(self, word2idx, max_len=300):
        self.word2idx = word2idx
        self.max_len = max_len

    def encode(self, text):
        tokens = re.findall(r"\w+", text.lower())
        ids = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        ids = ids[: self.max_len]

        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)


# ---------------- BiLSTM Encoder ----------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, embeddings=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, mask=None):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            out = out * mask
            pooled = out.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = out.mean(dim=1)

        return pooled
