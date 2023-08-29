# :arrow_forward: [**Try Out Opal on Streamlit**](https://opal-ai.streamlit.app/)
![modelsize](https://img.shields.io/github/size/Eve-ning/opal/src/opal/models/V4/2023_08_01_performance_mania_top_10000_20230819163602.csv/lightning_logs/version_1/evaluation/model.ckpt)
![version](https://img.shields.io/pypi/v/opal-net)
![pyversions](https://img.shields.io/pypi/pyversions/opal-net)
[![https://img.shields.io/pypi/dm/opal-net](https://img.shields.io/pypi/dm/opal-net)](https://pypi.org/project/opal-net/)

[![Test Model Pipeline Inference](https://github.com/Eve-ning/opal/actions/workflows/pipeline-test.yml/badge.svg?branch=master)](https://github.com/Eve-ning/opal/actions/workflows/pipeline-test.yml)
# :comet: opal-net
opal is an accuracy-prediction model.

It uses a Matrix Factorization branch & Multi-layered Perceptron branch to learn associations between user and maps,
then use those associations to predict new scores never before seen.

## :hourglass_flowing_sand: Project Status
Currently, it's in its early access, that means, it'll have many problems!
However, we're working on it to minimize these issues o wo)b

## :arrow_double_down: Dataset Used

I used the top 10K mania users data from https://data.ppy.sh.
After preprocessing, we use
- ~10m scores for training
- ~1m scores for validation and testing each

After preprocessing, we found ~30K valid users, ~10K valid maps
This models can thus help predict ~300m unplayed scores!

### Users
We deem a player on separate years as a different user. This is to reflect
the improvement of the player after time.

## :high_brightness: Usage

To use this, install `opal-net`

```bash
pip install opal-net
```

Then in a python script
> Tip: GPU doesn't speed this up significantly, you can use a CPU.
```py
from opal import OpalNet

# Load in the model
# You can explicitly specify map_location='cpu' or 'cuda' in map_location=...
opal = OpalNet.load()

# You can predict a single instance.
#
# The 1st arg: "<USER_ID>/<YEAR>",
# The 2nd arg: "<MAP_ID>/<SPEED>" 
#   <YEAR> is the year of the user to test.
#   <SPEED> can be {-1, 0, or 1} for {HT, NT, DT}
#
# For example: 
# Predict Evening on Year 2020, on the map Triumph & Regret [Regret] at Double Time
pred = opal.predict("2193881/2020", "767046/1")

# You can predict multiple entries at the same time. This is much faster that looping the above.
# Note that both lists must be of the same length!
# Note: If you're predicting millions, partition the predictions to reduce GPU memory usage!
preds = opal.predict(["2193881/2020", "2193881/2017"], ["767046/0", "767046/1"])

# Note that if the prediction doesn't exist, then it'll raise a ValueError
try:
    opal.predict("2193881/2018", "767046/0")
except ValueError:
    print("Prediction Failed!")
```

## :brain: AlphaOsu!
Currently, opal doesn't provide recommendations, however, you can try out [AlphaOsu!](https://alphaosu.keytoix.vip/).
- [AlphaOsu! GitHub](https://github.com/AlphaOSU)
- [Support AlphaOsu!](https://alphaosu.keytoix.vip/support)

## Annex

### Why not Score Metric?
Score is not straightforward to calculate, and may be difficult to debug. Furthermore, score isn't of interest when
calculating performance points anymore.

[osu!mania ScoreV1 Reference](https://osu.ppy.sh/wiki/en/Gameplay/Score/ScoreV1/osu%21mania)
