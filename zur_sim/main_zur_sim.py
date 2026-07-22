"""
Hub-distribution simulation driver
==================================
For a chosen pair of target hubs, walks every measurement timestamp of a day,
builds a node-count TransitionMatrix from that snapshot, and samples where
requests would be routed. Aggregating the samples per timestamp yields a time
series of how demand is distributed across hubs, which is plotted and optionally
written to `data/<hubA>_<hubB>_<date>/hub_distribution.csv`.
"""

import os

import numpy as np
import pandas as pd
import socket
if socket.gethostname() == 'berttrainer-large':
    from datasource import DataSource
    from transition_matrix import TransitionMatrix
else:
    from zur_sim.datasource import DataSource
    from zur_sim.transition_matrix import TransitionMatrix
import geopandas as gpd
from progressbar import progressbar
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def main(hub_inds: tuple[str, str]):
    """Simulate and plot the per-hub request distribution for one hub pair.

    `hub_inds` is a `(hub_a, hub_b)` pair of ZSID station ids used as the target
    hubs. For each measurement timestamp of the module-level `date` (drawn from
    the global `df`), a `nodecount` TransitionMatrix is built and sampled; the
    sampled hub counts are collected into a per-timestamp DataFrame that is drawn
    as a line plot and returned. If the output CSV for this pair already exists
    the function returns early without recomputing (the CSV write itself is
    currently commented out). Relies on the module globals `df` and `date`.
    """
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
    # df_dist.to_csv(os.path.join(folder_dir, 'hub_distribution.csv'))
    return df_dist


# date = '2026-02'
date = '2026-01-01T12:00:00'
if __name__ == '__main__':
    ds = DataSource()
    df = ds.df
    zsid = df.ZSID.unique()
    hub_b = zsid[57]
    print(len(zsid))
    for hub_a in progressbar(zsid)[:1]:

        if hub_a == hub_b:
            continue
        else:
            main((hub_a, hub_b))

