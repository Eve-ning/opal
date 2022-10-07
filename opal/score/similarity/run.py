import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from opal.score.dataset import Dataset
from opal.score.preprocessing_dynamic import PreprocessingDynamic
from opal.score.similarity.similarity_eval import similarity_model_eval

if True:
    # PyCharm linting keeps wrecking this import, so it's in a True cond
    pass
import tensorflow as tf

from keras.models import load_model

m = load_model("../../../model-7025351535907307520/upper_bound/001/saved_model.pb")


