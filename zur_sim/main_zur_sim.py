import os

import numpy as np
import pandas as pd

from zur_sim.datasource import DataSource
from zur_sim.transition_matrix import TransitionMatrix
import geopandas as gpd
from progressbar import progressbar
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def main(hub_inds: tuple[str, str]):
    folder_dir = os.path.join('data', '_'.join([str(hub_inds[0]), str(hub_inds[1]), date]))
    print(folder_dir)
    # print(os.path.abspath(os.path.join(folder_dir, 'hub_distribution.csv')))
    # print(os.path.exists(os.path.abspath(os.path.join(folder_dir, 'hub_distribution.csv'))))

    if not os.path.exists(os.path.abspath(folder_dir)):
        os.mkdir(os.path.abspath(folder_dir))
    elif os.path.exists(os.path.abspath(os.path.join(folder_dir, 'hub_distribution.csv'))):
        return
    hub_distributions = []

    # df.loc[ds.df.AnzFahrzeuge.isna(), 'AnzFahrzeuge'] = np.inf
    datetimes = df.loc[df.MessungDatZeit.str.startswith(date)].MessungDatZeit.unique()
    for datetime in datetimes:
        df_tmp = df.loc[df.MessungDatZeit == datetime]
        tm = TransitionMatrix(df_tmp, kind='nodecount', target_ids=hub_inds)

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
    df_dist.to_csv(os.path.join(folder_dir, 'hub_distribution.csv'))
    return df_dist


date = '2026-02'
if __name__ == '__main__':
    ds = DataSource()
    df = ds.df
    zsid = df.ZSID.unique()
    hub_b = zsid[57]
    print(len(zsid))
    for hub_a in progressbar(zsid):

        if hub_a == hub_b:
            continue
        else:
            main((hub_a, hub_b))

