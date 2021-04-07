import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def multi_head_attention(q, k, v, mask=None):
    # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
    # k,v shape = (B, n_heads, N, key_dim)
    # mask.shape = (B, group, N)

    B, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_heads, n, N)
    score = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(q.size(-1))

    if mask is not None:
        score += mask[:, None, :, :].expand_as(score)

    shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
    attn = torch.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
    return attn.reshape(*shp)


def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads=8,
    ):
        super(EncoderLayer, self).__init__()

        self.n_heads = n_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim))
        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.norm2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, mask=None):
        q = make_heads(self.Wq(x), self.n_heads)
        k = make_heads(self.Wk(x), self.n_heads)
        v = make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class MHAEncoder(nn.Module):
    def __init__(self,
                 n_layers,
                 n_heads,
                 embedding_dim,
                 input_dim,
                 add_init_projection=True,
                 max_len=128):
        super(MHAEncoder, self).__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = nn.Linear(input_dim, embedding_dim)
        self.attn_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=embedding_dim, n_heads=n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        if hasattr(self, 'init_projection_layer'):
            x = self.init_projection_layer(x)
        for layer in self.attn_layers:
            x = layer(x, mask)
        return x
