from typing import Tuple

import pandas as pd
from matplotlib import pyplot as plt


def plot_ct(df: pd.DataFrame, col: str, title: str, figsize: Tuple[int, int]=(12, 3)):
    ct = (pd.crosstab(df[col], df['onshore_offshore'])
          [['onshore', 'offshore', 'onshore-offshore']]
          .sort_values("onshore", ascending=False))
    ct.plot(kind='bar', figsize=figsize, xlabel=col, ylabel='Кол-во месторождений', title=title)
    plt.show()


def plot_iqr_boxplot_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame, title: str):
    common_cols = train_df.columns.intersection(test_df.columns)

    fig, axes = plt.subplots(1, len(common_cols), figsize=(len(common_cols) * 3, 4))
    if len(common_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, common_cols):
        data = pd.DataFrame({
            'Train': train_df[col].dropna(),
            'Test': test_df[col].dropna()
        })
        data.boxplot(ax=ax)
        ax.set_title(col)
        ax.set_ylabel('Значение')

    plt.suptitle(f"Сравнение IQR: {title}")
    plt.tight_layout()
    plt.show()
