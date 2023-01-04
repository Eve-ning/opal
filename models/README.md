# Models

Currently, there's only 1 model, which is `tiny`.
Note that this model is only trained on the year and month specified, thus is not compatible with different datasets.

**See below on how to load the model.**

**tiny**: `NeuMF(16, 16, 8, lr=0.005)`.

## Loading

```python
from opal.score.collaborative_filtering import NeuMF
from opal.score.datamodule import ScoreDataModule
import pytorch_lightning as pl

dm = ScoreDataModule(
    ds_yyyy_mm="2022_11",       # Must match checkpoint meta 
    score_bounds=(7.5e5, 1e6),  # Must match checkpoint meta
)
net = NeuMF.load_from_checkpoint("path/to/tiny/checkpoint.ckpt", dm=dm)
net.eval()
trainer = pl.Trainer()
```