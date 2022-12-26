# Models

There are 2 types of models I've done. 

Medium is not uploaded due to its size.

**Tiny**: `LitNeuMFNet(dm.n_uid, dm.n_mid, 64, 64, 8, dm.scaler_accuracy, lr=0.005)`.

**Medium**: `LitNeuMFNet(dm.n_uid, dm.n_mid, 256, 256, 32, dm.scaler_accuracy, lr=0.005)`.

## Loading

To load these models, make sure that the emb dims are right

```python
from opal.score.collaborative_filtering import NeuMF
from opal.score.datamodule import ScoreDataModule

dm = ScoreDataModule(
    ds_yyyy_mm="2022_12", batch_size=256,
    m_min_support=50, u_min_support=50,
    score_bounds=(7.5e5, 1e6)
)
net = NeuMF.load_from_checkpoint(
    "path/to/tiny/model.ckpt",
    uid_no=4007,
    mid_no=6189,
    mf_emb_dim=64,  # 256 for Medium
    mlp_emb_dim=64,  # 256 for Medium
    mlp_chn_out=8,  # 32 for Medium
    scaler=dm.scaler_accuracy,
    lr=0.005
)
```