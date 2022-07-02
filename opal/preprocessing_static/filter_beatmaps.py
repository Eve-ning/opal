from opal.dataset import Dataset
from opal.preprocessing_static.conf import BEATMAP_KEYS


def filter_beatmaps(ds: Dataset):
    """ Removes unwanted beatmaps from the dataset """
    df = ds.beatmaps_df

    df = df[df.diff_size.isin(BEATMAP_KEYS)]
    df = df[df.beatmap_id.isin(ds.nsv_ids)]
    df = df[
        ['beatmap_id', 'beatmapset_id', 'user_id', 'filename', 'checksum',
         'version', 'diff_size', 'diff_overall', 'approved',
         'difficultyrating']
    ]
    df.to_csv(ds.beatmaps_filtered_csv_path, index=None)
