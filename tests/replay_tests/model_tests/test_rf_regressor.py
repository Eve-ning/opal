import numpy as np
import pandas as pd
import pytest
from junit_xml import TestSuite, TestCase
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


@pytest.fixture(scope='session')
def test_cases():
    tests = []
    yield tests
    ts = TestSuite("RF Regressor Suite", tests)
    with open('rf_regressor_score.xml', 'w+') as f:
        TestSuite.to_file(f, [ts])


def roll_x(X: pd.DataFrame, window_size: int, agg_func: str):
    X_ohe = X.drop('diff', axis=1)
    X_ohe = X_ohe.multiply(X['diff'], axis='index')
    X_ohe: pd.DataFrame = 1 / X_ohe
    X_ohe = X_ohe.replace(np.inf, np.nan)
    X_ohe = X_ohe.rolling(window_size, min_periods=1).agg(agg_func)
    return X_ohe.fillna(0)


@pytest.mark.parametrize(
    'window_size',
    (1,)#, 3, 5)
)
@pytest.mark.parametrize(
    'n_estimators',
    (5, 15)
)
@pytest.mark.parametrize(
    'agg_func',
    ('sum', 'max')
)
def test_rf_regressor(train_test_data, validation_data,
                      n_estimators, window_size, agg_func, test_cases):
    SPLITS = 5
    X, y = train_test_data
    X = roll_x(X, window_size, agg_func)
    X_val, y_val = validation_data
    X_val = roll_x(X_val, window_size, agg_func)
    skf = KFold(n_splits=SPLITS, shuffle=True, random_state=0)
    for e, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rfg = RandomForestRegressor(verbose=1, n_estimators=n_estimators)
        rfg.fit(X_train, y_train)

        test_cases.append(TestCase(
            f'{n_estimators} Estimators '
            f'{window_size} W Size '
            f'({e}/{SPLITS}): '
            f'{rfg.score(X_train, y_train):.3f}/'
            f'{rfg.score(X_test, y_test):.3f}/'
            f'{rfg.score(X_val, y_val):.3f}',
            classname=f"RF Regressor Model w/ Rolling {agg_func}",
        ))
