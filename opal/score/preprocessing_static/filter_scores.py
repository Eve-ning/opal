from opal.score.dataset import Dataset
from opal.score.preprocessing_static.conf import SCORE_MOD_DOUBLE, SCORE_FILTER, \
    MOD_DT, MOD_HT, MOD_EZ
from opal.conf.mods import OsuMod


def filter_scores(ds: Dataset):
    """ Removes unwanted scores from the dataset """

    df = ds.scores_df
    # Correct half time, easy, no fail to 1M
    for mod in SCORE_MOD_DOUBLE:
        mask = (df.enabled_mods & mod) > 0
        print(f"Scores with mod {mod}: {sum(mask)}")
        df.loc[mask, 'score'] = df[mask].score * 2
    # Remove all scores outside of range
    df = df[(SCORE_FILTER[0] < df.score) & (df.score < SCORE_FILTER[1])]

    # This creates the mod column, which is easier to understand
    df['mod'] = 0
    df['mod'] += ((df.enabled_mods & OsuMod.DOUBLE_TIME) > 0) * MOD_DT
    df['mod'] += ((df.enabled_mods & OsuMod.NIGHTCORE) > 0) * MOD_DT
    df['mod'] += ((df.enabled_mods & OsuMod.HALF_TIME) > 0) * MOD_HT
    df['mod'] += ((df.enabled_mods & OsuMod.EASY) > 0) * MOD_EZ

    df = df[
        ['beatmap_id', 'user_id', 'score', 'count50', 'count100', 'count300',
         'countmiss', 'countgeki', 'countkatu', 'mod', 'year', 'month', 'pp',
         'replay']
    ]

    df.to_csv(ds.scores_filtered_csv_path, index=None)
