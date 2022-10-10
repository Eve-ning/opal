import torch
import torch.nn as nn


class NeuMFNet(nn.Module):
    def __init__(self, uid_no, mid_no, mf_emb_dim, mlp_emb_dim, mlp_chn_out):
        super(NeuMFNet, self).__init__()

        self.u_mf_emb = nn.Embedding(uid_no, mf_emb_dim)
        self.m_mf_emb = nn.Embedding(mid_no, mf_emb_dim)
        self.mf_net = nn.Sequential(
            nn.Linear(mlp_emb_dim, mlp_emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_emb_dim, mlp_emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.u_mlp_emb = nn.Embedding(uid_no, mlp_emb_dim)
        self.m_mlp_emb = nn.Embedding(mid_no, mlp_emb_dim)
        self.mlp_net = nn.Sequential(
            nn.Linear(mlp_emb_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, mlp_chn_out),
        )
        self.neu_mf_net = nn.Sequential(
            nn.Linear(mlp_chn_out + mf_emb_dim, 1),
            nn.Softplus(),
        )

    def forward(self, uid, mid):
        u_mf_emb = self.u_mf_emb(uid)
        m_mf_emb = self.m_mf_emb(mid)
        mf_out = self.mf_net(torch.mul(u_mf_emb, m_mf_emb))

        u_mlp_emb = self.u_mlp_emb(uid)
        m_mlp_emb = self.m_mlp_emb(mid)
        mlp_out = self.mlp_net(torch.concat([u_mlp_emb, m_mlp_emb], dim=-1))

        pred = self.neu_mf_net(torch.concat([mf_out, mlp_out], dim=-1))

        return pred[:, :, 0]
