import torch
import torch.nn as nn


class OpalNetModule(nn.Module):
    def __init__(self, n_uid, n_mid, emb_dim: int):
        super(OpalNetModule, self).__init__()

        self.u_mf_emb = nn.Embedding(n_uid, emb_dim)
        self.m_mf_emb = nn.Embedding(n_mid, emb_dim)
        self.linear = nn.Linear(emb_dim * 2 + 1, 1)

    def forward(self, uid, mid):
        """

        Notes:
            ┌───┐ ┌───┐   ┌────────┐    ┌─────┐
            │UID├─► E ├─┬─►MULTIPLY├────►MFNet│
            └───┘ │ M │ │ └───▲────┘    └──┬──┘
                  │ B │ │     │            │
                  │ E │ └─────┼─┐       ┌──▼────────┐ ┌────────┐
                  │ D │       │ │       │CONCATENATE├─►NeuMFNet│
                  │ D │ ┌─────┘ │       └──▲────────┘ └────────┘
                  │ I │ │       │          │
            ┌───┐ │ N │ │ ┌─────▼─────┐ ┌──┴───┐
            │MID├─► G ├─┴─►CONCATENATE├─►MLPNet│
            └───┘ └───┘   └───────────┘ └──────┘

        Args:
            uid:
            mid:

        Returns:

        """

        u_mf_emb = self.u_mf_emb(uid)
        m_mf_emb = self.m_mf_emb(mid)
        x = torch.bmm(u_mf_emb, m_mf_emb.swapaxes(1, 2))[:, :, 0]
        x = torch.relu(x)
        x = self.linear(torch.concat((u_mf_emb[:, 0], m_mf_emb[:, 0], x), dim=1))
        return x
