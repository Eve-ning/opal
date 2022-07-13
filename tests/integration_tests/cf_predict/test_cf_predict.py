from __future__ import annotations

from pathlib import Path

from opal.collaborative_filtering.predict import predict
from opal.conf import OSU_DS_2022_04
from opal.dataset import Dataset
from opal.preprocessing_dynamic import PreprocessingDynamic


def test_cf_predict():
    ds = Dataset(OSU_DS_2022_04, score_set="scores_top10k")
    unpopular_maps_thres = 0.2
    unpopular_plays_thres = 0.2
    score_year_filter = (2019, 2022)
    sr_min_thres = 4.5
    acc_filter = (0.80, 0.98)
    remove_mod = False
    sample_users = None

    df = PreprocessingDynamic(
        ds.joined_filtered_df,
        unpopular_maps_thres=unpopular_maps_thres,
        unpopular_plays_thres=unpopular_plays_thres,
        score_year_filter=score_year_filter,
        sr_min_thres=sr_min_thres,
        acc_filter=acc_filter,
        remove_mod=remove_mod,
        sample_users=sample_users,
    ).filter(calc_acc=True)
    pred = predict(df, Path("predictions") /
                   f"{unpopular_maps_thres},"
                   f"{unpopular_plays_thres},"
                   f"{score_year_filter},"
                   f"{sr_min_thres},"
                   f"{acc_filter},"
                   f"{remove_mod},"
                   f"{sample_users}.csv")
