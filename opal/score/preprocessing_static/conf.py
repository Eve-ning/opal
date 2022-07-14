from opal.score.conf.mods import OsuMod

SCORE_FILTER = (600000, 1000000)
SCORE_MOD_DOUBLE = (OsuMod.EASY, OsuMod.NO_FAIL, OsuMod.HALF_TIME)
BEATMAP_KEYS = (4, 7)

MOD_DT = 2 ** 0
MOD_HT = 2 ** 1
MOD_EZ = 2 ** 2

RENAME_MAPPING = {
    'user_id_x': 'user_id',
    'user_id_y': 'user_id_creator',
    'countgeki': "count300g",
    'countkatu': "count200",
    'countmiss': "count0",
    'difficultyrating': "sr",
    'beatmap_id': "map_id",
    'beatmapset_id': "mapset_id",
    'version': "map_diff_name",
    'filename': "map_file_name",
    'diff_overall': "od",
    'diff_size': "keys",
}
