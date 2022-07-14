import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from opal.conf.conf import PUBLIC_OSU_REPLAYS_DIR
from opal.replay.preprocess_replay_error import preprocess_replay_error


def test_preprocess_replay_error():
    map_dirs = [d for d in PUBLIC_OSU_REPLAYS_DIR.iterdir() if d.is_dir()]
    df = pd.concat([preprocess_replay_error(d) for d in map_dirs])

    y = df['error']
    X = df.drop('error', axis=1)

    skf = KFold(n_splits=3, shuffle=True, random_state=0)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rfg = RandomForestRegressor(verbose=2, n_estimators=100)
        rfg.fit(X_train, y_train)
        y_pred = rfg.predict(X_test)
        print(rfg.score(X_test, y_test))
