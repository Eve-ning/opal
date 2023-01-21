# Models

*Note: We provided V1_2022_11_NoQT additionally to compare the effects between `QuantileTransform` and `StandardScaler`.
We found no significant differences.*

**See below on how to load the model.**

| Model      | MAE   | RMSE  | Error Distribution             |
|------------|-------|-------|--------------------------------|
| V1_2022_11 | 0.99% | 1.62% | ![Error](V1_2022_11/error.png) |

## Dependencies

Each model is strongly coupled with the `DataModule` trained on it.
To load the checkpoint, the correct `DataModule` must be provided, else will not function properly.

| Model      | Dataset | Score Filter         |
|------------|---------|----------------------|
| V1_2022_11 | 2022_11 | (500,000, 1,000,000) |

## Loading

```python
from opal.score.collaborative_filtering import NeuMF
from opal.score.datamodule import ScoreDataModule
import pytorch_lightning as pl

dm = ScoreDataModule(
    ds_yyyy_mm="2022_11",  # Must match checkpoint meta 
    score_bounds=(5e5, 1e6),  # Must match checkpoint meta
)
net = NeuMF.load_from_checkpoint("path/to/model/checkpoint.ckpt", dm=dm)
net.eval()  # Prevent gradient updates.
trainer = pl.Trainer()
```

