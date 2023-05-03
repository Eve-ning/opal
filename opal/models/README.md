# Models

**See below on how to load the model.**

| Model      | R2     | MAE   | RMSE  | Error Distribution             |
|------------|--------|-------|-------|--------------------------------|
| V2_2023_01 | 81.48% | 1.18% | 1.71% | ![Error](V2_2023_01/error.png) |
| V2_2023_04 | 71.88% | 1.14% | 1.68% | ![Error](V2_2023_04/error.png) |

**Note: V1 is deprecated due to incompatibility with new model signature.**

**Old notebooks may not be runnable**

## Limitations

The model cannot ...
- predict maps not played by at least 50 players within the top 10k
- predict players not in the top 10k
- predict players who have not played at least 50 unique ranked maps in that year.

The predictive power (i.e. accuracy) is dependent on the number of players associated with each map.
Thus, these will be less accurate
- Half-time/Double-time maps
- Unpopular maps
- Players who play little


## Loading

```python
from opal.module.collaborative_filtering import NeuMF
import pytorch_lightning as pl

net = NeuMF.load_from_checkpoint("path/to/model/checkpoint.ckpt")
net.eval()  # Prevent gradient updates.
USER_ID = 12345
YEAR = 2020
MAP_ID = 54321
SPEED = 0  # 0: Normal Time, -1: Half Time, 1: Double Time
pred_acc = net.predict(f"{USER_ID}/{YEAR}", f"{MAP_ID}/{SPEED}")
```

