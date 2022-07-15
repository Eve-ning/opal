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


@pytest.mark.parametrize(
    'n_estimators',
    (5, 15, 35)
)
def test_rf_regressor(train_test_data, validation_data, n_estimators,
                      test_cases):
    X, y = train_test_data
    X_val, y_val = validation_data
    skf = KFold(n_splits=5, shuffle=True, random_state=0)
    for e, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rfg = RandomForestRegressor(verbose=1, n_estimators=n_estimators)
        rfg.fit(X_train, y_train)

        test_cases.append(TestCase(
            f'{n_estimators} Estimators ({e}): '
            f'{rfg.score(X_train, y_train):.3f}/'
            f'{rfg.score(X_test, y_test):.3f}/'
            f'{rfg.score(X_val, y_val):.3f}',
            classname=f"RF Regressor Model",
        ))
