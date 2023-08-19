import torch
import torch.nn as nn


class OpalNetBlock(nn.Module):
    def __init__(self, in_chn, out_chn, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_chn, out_chn),
            nn.ReLU(),
            nn.BatchNorm1d(out_chn),
        )

    def forward(self, x):
        return self.net(x)


class OpalNetModule(nn.Module):
    def __init__(self, n_uid, n_mid, emb_dim: int,
                 mlp_range: list[int] = (128, 64, 32, 8),
                 ):
        super(OpalNetModule, self).__init__()

        self.u_emb = nn.Embedding(n_uid, emb_dim)
        self.m_emb = nn.Embedding(n_mid, emb_dim)

        self.bn_u_emb = nn.BatchNorm1d(emb_dim)
        self.bn_m_emb = nn.BatchNorm1d(emb_dim)

        self.mlp_net = nn.Sequential(
            OpalNetBlock(emb_dim * 2, mlp_range[0]),
            *[OpalNetBlock(i, j) for i, j in zip(mlp_range[:-1], mlp_range[1:])]
        )

        self.linear = nn.Linear(mlp_range[-1] + 1, 1)

    def forward(self, uid, mid):
        """

        Notes:
            ┌───┐
            │UID├────┬───────────────────┐
            └───┘    │                   │
                 ┌───▼────┐ ┌────┐ ┌─────▼─────┐ ┌──────┐ ┌───────┐
                 │DOT PROD├─┤RELU├─►CONCATENATE├─►LINEAR├─►PREDICT│
                 └───▲────┘ └────┘ └─────▲─────┘ └──────┘ └───────┘
            ┌───┐    │                   │
            │MID├────┴───────────────────┘
            └───┘

        Args:
            uid:
            mid:

        Returns:

        """

        # Shape of Emb: [Batch Size, Embed Size]
        u_emb = self.u_emb(uid)[:, 0, :]
        m_emb = self.m_emb(mid)[:, 0, :]
        u_emb: torch.Tensor = self.bn_u_emb(u_emb)
        m_emb: torch.Tensor = self.bn_m_emb(m_emb)

        # [Batch Size, Embed Size * 2]
        x_mlp = torch.concat([u_emb, m_emb], dim=1)
        x_mlp = self.mlp_net(x_mlp)

        # [Batch Size, 1]
        x_mf = torch.bmm(torch.unsqueeze(u_emb, 1),
                         torch.unsqueeze(m_emb, 2))[:, 0, :]
        x_mf = torch.relu(x_mf)

        # [Batch Size, MLP Range[-1] + 1]
        x = torch.concat((x_mlp, x_mf), dim=1)
        x = self.linear(x)
        return x
