from typing import List

import torch
import torch.nn as nn


class OpalNetBlock(nn.Module):
    def __init__(self, in_chn, out_chn, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_chn, out_chn),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class OpalNetModule(nn.Module):
    def __init__(self, n_uid, n_mid,
                 emb_dim: int,
                 mf_repeats: int = 2,
                 mlp_range: List[int] = [512, 256, 128, 64, 32, 16]):
        super(OpalNetModule, self).__init__()

        self.u_mf_emb = nn.Embedding(n_uid, emb_dim)
        self.m_mf_emb = nn.Embedding(n_mid, emb_dim)
        self.mf_net = nn.Sequential(*[OpalNetBlock(emb_dim, emb_dim) for _ in range(mf_repeats)])
        self.u_mlp_emb = nn.Embedding(n_uid, emb_dim)
        self.m_mlp_emb = nn.Embedding(n_mid, emb_dim)
        self.mlp_net = nn.Sequential(
            OpalNetBlock(emb_dim * 2, mlp_range[0]),
            *[OpalNetBlock(i, j) for i, j in zip(mlp_range[:-1], mlp_range[1:])]
        )
        self.neu_mf_net = nn.Sequential(
            nn.Linear(mlp_range[-1] + emb_dim, 1),
            nn.Tanh(),
        )

    def forward(self, uid, mid):
        u_mf_emb = self.u_mf_emb(uid)
        m_mf_emb = self.m_mf_emb(mid)
        mf_out = self.mf_net(torch.mul(u_mf_emb, m_mf_emb))

        u_mlp_emb = self.u_mlp_emb(uid)
        m_mlp_emb = self.m_mlp_emb(mid)
        mlp_out = self.mlp_net(torch.concat([u_mlp_emb, m_mlp_emb], dim=-1))

        pred = self.neu_mf_net(torch.concat([mf_out, mlp_out], dim=-1)) * 2

        return torch.tanh(pred)[:, :, 0] * 5
