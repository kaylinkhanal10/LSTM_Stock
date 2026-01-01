import torch
import torch.nn as nn
import torch.nn.functional as F

class BiAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.scale = hidden ** 0.5

    def forward(self, Q, P, q_mask, p_mask):
        # Q: [B, q_len, 2H]
        # P: [B, p_len, 2H]
        scores = torch.bmm(P, Q.transpose(1, 2)) / self.scale

        scores = scores.masked_fill(q_mask.unsqueeze(1) == 0, -1e9)
        attn_q = F.softmax(scores, dim=-1)
        P2Q = torch.bmm(attn_q, Q)

        attn_p = F.softmax(scores.max(dim=-1)[0], dim=-1)
        Q2P = torch.bmm(attn_p.unsqueeze(1), P).repeat(1, P.size(1), 1)

        return torch.cat([P, P2Q, P * P2Q, P * Q2P], dim=-1)


class ReaderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.q_lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.p_lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)

        self.attn = BiAttention(hidden * 2)

        self.model_lstm = nn.LSTM(hidden * 8, hidden, batch_first=True, bidirectional=True)

        self.start_fc = nn.Linear(hidden * 2, 1)
        self.end_fc   = nn.Linear(hidden * 2, 1)

    def forward(self, q, p, q_mask, p_mask):
        q_emb = self.embedding(q)
        p_emb = self.embedding(p)

        q_enc, _ = self.q_lstm(q_emb)
        p_enc, _ = self.p_lstm(p_emb)

        fused = self.attn(q_enc, p_enc, q_mask, p_mask)
        modeled, _ = self.model_lstm(fused)

        start = self.start_fc(modeled).squeeze(-1)
        end   = self.end_fc(modeled).squeeze(-1)

        start = start.masked_fill(p_mask == 0, -1e9)
        end   = end.masked_fill(p_mask == 0, -1e9)

        return start, end
