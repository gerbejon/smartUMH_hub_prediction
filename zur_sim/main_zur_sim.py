import numpy as np
import pandas as pd

from zur_sim.datasource import DataSource
from zur_sim.transition_matrix import TransitionMatrix
import geopandas as gpd
from progressbar import progressbar
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    hub_distributions = []
    date = '2026-01-1'
    ds = DataSource()
    # df = ds.df.loc[ds.df.AnzFahrzeuge.notna()]
    df = ds.df
    # df.loc[ds.df.AnzFahrzeuge.isna(), 'AnzFahrzeuge'] = np.inf
    datetimes = df.loc[df.MessungDatZeit.str.startswith(date)].MessungDatZeit.unique()
    for datetime in progressbar(datetimes):
        df_tmp = df.loc[df.MessungDatZeit == datetime]
        tm = TransitionMatrix(df_tmp, kind='nodecount')

        # tm = TransitionMatrix(df, kind='density')
        tm.main()
        res = tm.sample_multiple()
        tmp_dict = dict(Counter(res))
        tmp_dict['t'] = datetime
        hub_distributions.append(tmp_dict)

    df_dist = pd.DataFrame(hub_distributions)
    # df_dist.columns = [tm.nodes.index[tm.targets[0]], tm.nodes.index[tm.targets[1]], 't']
    # df_dist.set_index('t', inplace=True)
    df_dist.set_index('t', inplace=True)
    df_dist.columns = [tm.nodes.index[i] for i in df_dist.columns]
    sns.lineplot(data=df_dist)
    # sns.lineplot(data=df_dist.iloc[:,:-1])
    plt.title('Distribution of requests per hub')
    # plt.xticks(rotation=20)
    plt.grid(True)
    plt.show()