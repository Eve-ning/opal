from __future__ import annotations

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

from opal.collaborative_filtering.filter_df import Filter
from opal.collaborative_filtering.settings import Settings
from opal.dataset import Dataset
from opal.preprocessing_static.conf import OSU_DS_2022_04

# %%
ds = Dataset(OSU_DS_2022_04, score_set="scores_top10k")
s = Settings(unpopular_map_thres=0.15, unpopular_play_thres=0.15,
             score_filter=(0, 1000000))
df = Filter(s, ds.joined_filtered_df).filter_df()
# %%

count = df['countgeki'] + \
        df['count300'] + \
        df['countkatu'] + \
        df['count100'] + \
        df['count50'] + \
        df['countmiss']

df['accuracy'] = (
                     df['countgeki'] * 315 / 315 +
                     df['count300'] * 300 / 315 +
                     df['countkatu'] * 200 / 315 +
                     df['count100'] * 100 / 315 +
                     df['count50'] * 50 / 315
                 ) / count

# %%
import matplotlib.ticker as mtick


plt.figure(figsize=(20, 3))
interval = 0.0025
xs = np.arange(0, 1 + interval, interval)
plt.xticks(xs, rotation=90)

for e, x in enumerate(xs):
    plt.axvline(x=x, c='red' if e % 2 == 0 else 'orange',
                linestyle='dashed', linewidth=0.7)
plt.hist(df.accuracy, bins=1000)
plt.xlim([0.8, 1])
plt.xlabel("Accuracy (Lazer)")
plt.ylabel("Frequency")

plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.tight_layout()
plt.show()

#%%

plt.figure(figsize=(20, 3))
interval = 10000
xs = np.arange(0, 1000000 + interval, interval)
plt.xticks(xs, rotation=90)

for e, x in enumerate(xs):
    plt.axvline(x=x, c='red' if e % 2 == 0 else 'orange',
                linestyle='dashed', linewidth=0.7)
plt.hist(df.score, bins=1000)
plt.xlim([500000, 1000000])
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
# %%


counts, bins = np.histogram(
    df.score,
    bins=np.linspace(500000, 1000000, 501, ), )

counts = counts.astype(float)
counts /= counts.sum()
counts_rev_cum = np.cumsum(counts[::-1])[::-1]
x = bins[1:]
y = 1 - counts_rev_cum
plt.plot(x, y)
plt.xlabel("Score")
plt.ylabel("1 - Achievability")
plt.show()
# %%

counts, bins = np.histogram(
    df.accuracy,
    bins=np.linspace(0.8, 1, 501, ), )

counts = counts.astype(float)
counts /= counts.sum()
counts_rev_cum = np.cumsum(counts[::-1])[::-1]
x = bins[1:]
y = counts_rev_cum
plt.plot(x, y)
plt.xlabel("Accuracy")
plt.ylabel("Achievability")
plt.show()


# %%
# %%
# scipy.optimize.curve_fit(lambda )
def monoExp(x, m, t, b):
    return m * np.exp(-t * (x + b))


params, cv = scipy.optimize.curve_fit(monoExp, x[:], y[:], maxfev=5000)

m, t, b = params
m /= monoExp(1, m, t, b)

# plot the results
plt.plot(x, y, '-', label="data")
plt.plot(x, monoExp(x, m, t, b), '--', label="fitted")
plt.title("Fitted Exponential Curve")
plt.show()
# %%
print(f"{m:.3g}, {t:.3g}, {b:.3g}")


# %%

# scipy.optimize.curve_fit(lambda )
def poly(x, a, b, c):
    return b * x + c * x ** 2 + a


params, cv = scipy.optimize.curve_fit(poly, x, y, maxfev=5000)

a, b, c = params

# plot the results
plt.plot(x, y, '-', label="data")
plt.plot(x, poly(x, a, b, c), '--', label="fitted")
plt.title("Fitted Poly Curve")
plt.show()
# %%
plt.scatter(df.score, df.accuracy, s=1)
plt.ylim([0.992,1.0])
plt.xlim([995000,1000000])
plt.ylabel("Accuracy")
plt.xlabel("Score")
plt.show()
