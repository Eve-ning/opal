from junit_xml import TestSuite, TestCase
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


def test_rf_regressor(rep_data):
    X, y = rep_data
    skf = KFold(n_splits=3, shuffle=True, random_state=0)

    test_cases = []

    for e, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rfg = RandomForestRegressor(verbose=1, n_estimators=3)
        rfg.fit(X_train, y_train)
        test_cases.append(TestCase(
            f'Test {e}', f'Score: {rfg.score(X_test, y_test)}'
        ))

    ts = TestSuite("RF Regressor suite", test_cases)
    with open('score.xml', 'w') as f:
        TestSuite.to_file(f, [ts])
