import pandas as pd
from matplotlib import pyplot as plt


def plot_ct(df, col, title, figsize=(12, 3)):
    ct = (pd.crosstab(df[col], df['onshore_offshore'])
          [['onshore', 'offshore', 'onshore-offshore']]
          .sort_values("onshore", ascending=False))
    ct.plot(kind='bar', figsize=figsize, xlabel=col, ylabel='Кол-во месторождений', title=title)
    plt.show()
