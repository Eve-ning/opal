""" This preprocesses the data within a dataset.

The dataset dir must contain
- beatmaps.csv (osu_beatmaps)
- scores.csv (osu_scores_mania_high)
- files
  - 1.osu
  - 2.osu
  - ...
"""

from opal.conf import OSU_DS_2022_04
from opal.dataset import Dataset
from opal.preprocessing_static import classify_sv_maps, filter_scores, \
    remove_non_mania, filter_beatmaps, join_score_beatmaps


def test_preprocessing_static():
    ds = Dataset(OSU_DS_2022_04, score_set="scores_top10k")
    print("Removing Non Mania Maps")
    remove_non_mania(ds)
    print("Classifying SV Maps")
    classify_sv_maps(ds)
    print("Filtering Scores")
    filter_scores(ds)
    print("Filtering Beatmaps")
    filter_beatmaps(ds)
    print("Joining Filtered Datasets")
    join_score_beatmaps(ds)
    print("Completed")
