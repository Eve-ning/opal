import pandas as pd


def ecdf(x: pd.Series):
    counts = x.value_counts()
    x = counts.sort_index().cumsum() / len(counts)
    x.index = x.index.get_level_values(0)
    x /= x.max()
    return x