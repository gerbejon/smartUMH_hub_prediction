import os.path
import sys
import socket
import geopandas as gpd
import numpy as np
import pandas as pd
from progressbar import progressbar
if socket.gethostname() == 'berttrainer-large':
    from datasource import DataSource, cwd
    from tools import aggreate
    from create_raster import RasterCreater
else:
    from zur_sim.datasource import DataSource, cwd
    from zur_sim.tools import aggreate
    from zur_sim.create_raster import RasterCreater

from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
import random

class TransitionMatrix:
    def __init__(self, df, kind = None, target_ids: tuple[str, str] = None):
        if kind is None:
            print('please state underlying costcalculation: \n- vehicle count per node [nodecount]\n- density of vehicles [density]')
        elif kind == 'nodecount':
            self.z_col = 'AnzFahrzeuge'
        elif kind == 'density':
            self.z_col = 'AnzFahrzeugeSmooth'
        else:
            exit('kind must be either nodecount or density')

        df = self.handle_na_counts(df, nn=4, method='median')
        df = aggreate(df)
        rc = RasterCreater(df, kde_bw=0.3)
        rc.main_kde()
        # print(df.shape)
        self.raster = rc.grid_df
        self.df = df
        # self.df.set_index('ZSID', inplace=True)
        self.df['AnzFahrzeugeSmooth'] = rc.get_density_per_node(df)
        self.weight_distance = 0.5  # weight distance
        self.weight_traffic = 0.5  # weight traffic
        self.knn = 8
        self.target_ids = target_ids
        self.targets = tuple(np.where((self.df.ZSID.values == self.target_ids[0]) | (self.df.ZSID.values == self.target_ids[1]))[0])
        mask = [zsid in significant_hubs for zsid in self.df.ZSID.values]
        self.significant_targets = (
            tuple(
                np.where(
                    mask
                )[0]))

        # print('using: {} and {} as Target hubs'.format( target_ids[0], target_ids[1]))
        # print('using: {} and {} as Target hubs'.format( self.df.ZSID.iloc[self.targets[0]], self.df.ZSID.iloc[self.targets[1]]))
        self.timestamp = str(self.df.MessungDatZeit.loc[0]).replace('T', ' ')
        if not os.path.exists(f'{cwd}/plots/{self.timestamp.split(" ")[0]}'):
            os.makedirs(f'{cwd}/plots/{self.timestamp.split(" ")[0]}')

    def handle_na_counts(self, df, nn=4, method='median'):
        for i, row in df.loc[df.AnzFahrzeuge.isna()].iterrows():
            e0, n0 = row.EKoord, row.NKoord
            # compute distance
            dist = np.sqrt((df.loc[df.AnzFahrzeuge.notna()]["EKoord"] - e0) ** 2 + (df.loc[df.AnzFahrzeuge.notna()]["NKoord"] - n0) ** 2)
            # get n closest points

            if method == 'mean':
                df.loc[i, 'AnzFahrzeuge'] = df.loc[df.AnzFahrzeuge.notna()].loc[dist.nsmallest(nn).index]['AnzFahrzeuge'].mean()

            elif method == 'median':
                df.loc[i, 'AnzFahrzeuge'] = df.loc[df.AnzFahrzeuge.notna()].loc[dist.nsmallest(nn).index]['AnzFahrzeuge'].median()

            elif method == 'sample':
                df.loc[i, 'AnzFahrzeuge'] = np.random.choice(df.loc[df.AnzFahrzeuge.notna()].loc[dist.nsmallest(nn).index]['AnzFahrzeuge'].values, 1)

        return df

    def main(self, df = None):
        if df is None:
            df = self.df
        # df = self.aggregate(self.df)
        gdf = self.preprocess(df)
        self.cost_matrix = self.create_cost_trans_matrix(gdf)
        self.transition_matrix = self.define_transition_limits(self.cost_matrix, knn=self.knn)


    def calc_shortest_path(self, source, target, plot=True):
        self.get_shortest_path(self.matrix, source, target, plot=plot)


    def preprocess(self, df):
        df = df.loc[df[self.z_col].notna()]
        # df =  self.aggregate(df)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.EKoord, df.NKoord),
            crs="EPSG:2056"
        )

        # convert to lat/lon (required for Plotly)
        self.gdf = gdf.to_crs(3857)
        self.gdf["lat"] = self.gdf.geometry.y
        self.gdf["lon"] = self.gdf.geometry.x

        return self.gdf

    def create_cost_trans_matrix(self, gdf=None):
        if gdf is None:
            gdf = self.gdf
        self.nodes = gdf[['lon', 'lat', 'ZSID', self.z_col]].set_index('ZSID')
        self.nodes.columns = ['x', 'y', self.z_col]
        n_nodes = len(self.nodes)

        # Fully connected transition matrix (row sums to 1)
        transition_matrix = np.ones((n_nodes, n_nodes))  # / (n_nodes - 1)
        np.fill_diagonal(transition_matrix, 0)  # cannot stay at the same node

        # Extract coordinates as arrays
        coords = self.nodes[["x", "y"]].values
        traffic = self.nodes[self.z_col].values

        # Traffic cost = moving to a node with high traffic is costly
        traffic_matrix = np.tile(traffic, (n_nodes, 1))
        # traffic_matrix[i,j] = traffic at node j
        # optionally set diagonal to 0 or np.inf
        np.fill_diagonal(traffic_matrix, 0)
        # print(traffic_matrix.shape)

        # Compute distance matrix
        distance_matrix = cdist(coords, coords, metric="euclidean")
        # print(distance_matrix.shape)
        distance_matrix = distance_matrix ** 2
        # Optional: set diagonal to infinity (no self-loop)
        np.fill_diagonal(distance_matrix, 0)

        self.distance_norm = distance_matrix / distance_matrix.max()
        self.traffic_norm = traffic_matrix / traffic_matrix.max()

        return self.weight_distance * self.distance_norm + self.weight_traffic * self.traffic_norm

    def define_transition_limits(self, matrix, knn:int=None):
        if knn is None:
            return matrix

        else:
            # Create a sparse cost matrix initialized with infinities
            sparse_cost_matrix = np.full(self.distance_norm.shape, np.inf)
            for i in range(self.distance_norm.shape[0]):
                # indices of k closest nodes
                nearest_idx = np.argsort(self.distance_norm[i])[:knn]
                # copy total_cost values only for these neighbors
                sparse_cost_matrix[i, nearest_idx] = matrix[i, nearest_idx]
            return sparse_cost_matrix

    def get_shortest_path(self, matrix, source, targets, plot=True):

        total_cost0 = np.inf
        path0 = []
        for target in targets:
            # total_cost_matrix is your n_nodes x n_nodes numpy array
            G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            # G = nx.from_numpy_array(total_cost_matrix, create_using=nx.DiGraph)
            # DiGraph because cost[i,j] may not equal cost[j,i] if traffic is directional

            path = nx.dijkstra_path(G, source=source, target=target, weight="weight")
            total_cost = nx.dijkstra_path_length(G, source=source, target=target, weight="weight")
            if total_cost0 > total_cost:
                total_cost0 = total_cost
                path0 = path

        # print("Cheapest path:", path0)
        # print("Total cost:", total_cost0)

        return path0

    def plot(self, paths = None):

        fig, ax = plt.subplots(figsize=(8, 8))
        # X,Y,Z = self.raster['x'].values, self.raster['y'].values, self.raster['weight'].values

        nx = len(np.unique(self.raster["x"]))
        ny = len(np.unique(self.raster["y"]))

        Z = self.raster["weight"].values.reshape(ny, nx)

        im = ax.imshow(
            Z,
            extent=(self.raster["x"].min(), self.raster["x"].max(),
                    self.raster["y"].min(), self.raster["y"].max()),
            origin="lower",
            cmap="hot",
            alpha=0.4,
            zorder=1
        )


        # second axis on top
        # ax_graph = ax.twinx().twiny()


        x = self.gdf["lon"].values
        y = self.gdf["lat"].values

        # plt.figure(figsize=(6, 6))
        # plt.scatter(x, y, c='blue')
        # fig, ax = plt.subplots(figsize=(8, 8))


        sns.scatterplot(
            x=x,
            y=y,
            hue=self.nodes[self.z_col],  # whatever attribute you plot
            palette="hot",
            ax=ax,
            edgecolor="black",
            # hue_norm=norm,
            legend=False
            # title = df.MessungDatZeit.iloc[0].replace('T', ' '),
        )
        cx.add_basemap(ax)
        if paths is not None:
            for path in paths:
                # plot the cheapest path
                px_path, py_path = zip(*path)
                # px_path = [x[i] for i in x_path]
                # py_path = [y[i] for i in path]
                # plot the line (semi-transparent)
                ax.plot(px_path, py_path, color="black", lw=2, alpha=0.4)

                # plot the nodes of the path (transparent fill, visible border)
                ax.scatter(
                    px_path,
                    py_path,
                    facecolors="none",  # transparent center
                    edgecolors="none",
                    s=80,
                    linewidth=0.5,
                    zorder=5
                )
        # print(x)
        for i, txt in enumerate(range(len(x))):
            # if i in self.targets:
            if i in self.significant_targets:
            # if True:
                plt.text(x[i] + 0.3, y[i] + 0.3, str(self.nodes.index[i]))
        plt.title(self.timestamp)
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.savefig(f'{cwd}/plots/{self.timestamp.split(" ")[0]}/sim_map_{self.timestamp}.png')
        if not os.path.exists(f'{cwd}/zur_sim/plots/{self.timestamp.split(" ")[0]}/'):
            os.makedirs(f'{cwd}/zur_sim/plots/{self.timestamp.split(" ")[0]}/')
        # fig.savefig(f'{cwd}/zur_sim/plots/{self.timestamp.split(" ")[0]}/sim_map_{self.timestamp.replace(" ", "_")}.png', dpi=300, bbox_inches="tight")
        plt.show()

    def sample_multiple(self, nr_samples=100, targets=None, plot=False):
        if targets is None:
            targets = self.targets

        samples = random.choices(self.raster.index, k=nr_samples)
        paths = []
        nodeNr = pd.Series(range(self.nodes.shape[0]), index=self.nodes.index)
        best_hub = []
        for sample in samples:
            x0, y0 = self.raster.loc[sample, 'x'], self.raster.loc[sample, 'y']
            dist = np.sqrt((self.nodes["x"] - x0) ** 2 + (self.nodes["y"] - y0) ** 2)
            source = self.nodes.loc[dist.idxmin()].name
            path = self.get_shortest_path(matrix=self.transition_matrix, source=nodeNr[source], targets=targets)
            best_hub.append(path[-1])
            coord_path = [(x0, y0)]
            coord_path.extend([(self.gdf["lon"].values[i], self.gdf["lat"].values[i]) for i in path])
            paths.append(coord_path)
        if plot:
            self.plot(paths)
        return best_hub



significant_hubs = ['Z027', 'Z034', 'Z051', 'Z058', 'Z081', 'Z086', 'Z092', 'Z093', 'Z095', 'Z097', 'Z103', 'Z104', 'Z112', 'Z113', 'Z114']
zsids = 'Z034', 'Z059'
if __name__ == '__main__':
    # pass
    hub_distributions = []
    date = '2026-02-01T10:00:00'
    ds = DataSource()
    df = ds.df
    datetimes = df.loc[df.MessungDatZeit.str.startswith(date)].MessungDatZeit.unique()
    # for datetime in progressbar(datetimes[12:13]):
    datetime = datetimes[0]
    tmp_df = df.loc[df.AnzFahrzeuge.notna()]
    tmp_df = tmp_df.loc[tmp_df.MessungDatZeit == datetime]
    # tm = TransitionMatrix(df, kind='nodecount')

    tm = TransitionMatrix(tmp_df, kind='density', target_ids=zsids)
    tm.main()
    res = tm.sample_multiple()
    tm.plot()
