from zur_sim.datasource import DataSource
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from zur_sim.plots import interactive_plot
from zur_sim.tools import aggreate

class RasterCreater:
    def __init__(self, df, kde_bw=0.3):
        self.KDE = False
        self.df = df
        self.grid_res = 200
        self.kde_bw = kde_bw

    def main_kde(self, df=None):
        if df is None:
            df = self.df
        X,Y,Z = self.gaussian_kde(df)
        self.grid_df = self.postprocess(X, Y, Z)
        # self.plot(X, Y, Z)

    def main_gf(self, df=None):
        if df is None:
            df = self.df
        X,Y,Z = self.gaussan_filter(df)
        self.grid_df = self.postprocess(X, Y, Z)
        # self.plot(X, Y, Z)

    def preprocess(self, df, z_col='AnzFahrzeuge'):
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.EKoord, df.NKoord),
            crs="EPSG:2056"  # Swiss LV95
        )

        # convert to web mercator for map tiles
        gdf = gdf.to_crs(epsg=3857)
        self.x = gdf.geometry.x.values
        self.y = gdf.geometry.y.values
        self.z = gdf[z_col].values

        # create systematic grid
        xi = np.linspace(self.x.min(), self.x.max(), self.grid_res)
        yi = np.linspace(self.y.min(), self.y.max(), self.grid_res)

        X, Y = np.meshgrid(xi, yi)
        return X,Y

    def gaussan_filter(self, df):
        X, Y = self.preprocess(df, z_col='AnzFahrzeuge')
        Z = griddata(
            (self.x, self.y),
            self.z,
            (X, Y),
            method="nearest"
        )

        Z[np.isnan(Z)] = 0

        return X,Y,gaussian_filter(Z, sigma=3)

    def gaussian_kde(self, df):
        X, Y = self.preprocess(df, z_col='AnzFahrzeuge')
        # Kernel density estimate KDE
        positions = np.vstack([X.ravel(), Y.ravel()])
        coords = np.vstack([self.x, self.y])
        kde_traffic = gaussian_kde(coords, weights=self.z, bw_method=self.kde_bw)  # traffic and station density
        kde_stations = gaussian_kde(coords, bw_method=self.kde_bw)  # station density

        # gaussian_traffic =
        # Z = kde(positions)
        Z = kde_traffic(positions) / kde_stations(positions)
        return  X, Y, Z.reshape(X.shape)

    def postprocess(self, X,Y,Z):
        grid_df = pd.DataFrame({
            "x": X.ravel(),
            "y": Y.ravel(),
            "weight": Z.ravel()
        })
        grid_df["weight"] = grid_df["weight"] * self.z.sum()
        return grid_df

    def plot(self, X,Y,Z):
        fig, ax = plt.subplots(figsize=(8, 8))

        # plot heatmap
        im = ax.imshow(
            Z,
            extent=(X.min(), X.max(), Y.min(), Y.max()),
            origin="lower",
            cmap="hot",
            alpha=0.7
        )
        plt.show()

    def get_density_per_node(self, df=None):
        if df is None:
            df = self.df

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.EKoord, df.NKoord),
            crs="EPSG:2056"  # Swiss LV95
        )

        # convert to web mercator for map tiles
        gdf = gdf.to_crs(epsg=3857)
        AnzFahrzeugeSmooth=[]
        for i in range(df.shape[0]):
            qx, qy = gdf.geometry.x.values[i], gdf.geometry.y.values[i]
            dist = np.sqrt((self.grid_df["x"] - qx) ** 2 + (self.grid_df["y"] - qy) ** 2)
            AnzFahrzeugeSmooth.append(self.grid_df.loc[dist.idxmin(), "weight"])
        return np.array(AnzFahrzeugeSmooth)/np.sum(AnzFahrzeugeSmooth) * df.AnzFahrzeuge.sum()



if __name__ == '__main__':

    ds = DataSource()
    df = ds.df
    df = df.loc[df.AnzFahrzeuge.notna()]
    df_all =  df.loc[df.MessungDatZeit == '2026-01-08T12:00:00']
    df = aggreate(df_all)
    rc = RasterCreater(df, kde_bw=0.3)
    rc.main_kde()

    weights = rc.get_density_per_node()

    rc.grid_df

    #
    # df = df_all[['ZSID', 'ZSName', 'EKoord', 'Hoehe',
    #        'NKoord',  'AnzFahrzeuge', 'MessungDatZeit']].groupby('ZSID').agg(
    #             {'AnzFahrzeuge' : 'mean',
    #             'EKoord':'mean',
    #             'NKoord': 'mean',
    #             'ZSName': 'first',
    #              'Hoehe': 'mean',
    #              'MessungDatZeit': 'first'}
    #     ).reset_index()
    #
    #
    # # def create_grid(df,
    # grid_res = 200
    # #     ):
    #
    # df = df.loc[df.AnzFahrzeuge.notna()]
    #
    # # create GeoDataFrame using Swiss LV95 coordinates
    # gdf = gpd.GeoDataFrame(
    #     df,
    #     geometry=gpd.points_from_xy(df.EKoord, df.NKoord),
    #     crs="EPSG:2056"   # Swiss LV95
    # )
    #
    # # convert to web mercator for map tiles
    # gdf = gdf.to_crs(epsg=3857)
    # x = gdf.geometry.x.values
    # y = gdf.geometry.y.values
    # z = gdf["AnzFahrzeuge"].values
    #
    #
    # coords = np.vstack([x, y])
    #
    # # create systematic grid
    #
    # xi = np.linspace(x.min(), x.max(), grid_res)
    # yi = np.linspace(y.min(), y.max(), grid_res)
    #
    # X, Y = np.meshgrid(xi, yi)
    #
    #
    #
    # # gaussian filter
    # Z = griddata(
    #     (x, y),
    #     z,
    #     (X, Y),
    #     method="nearest"
    # )
    #
    # Z[np.isnan(Z)] = 0
    #
    # Z_smooth = gaussian_filter(Z, sigma=3)
    #
    # # Kernel density estimate KDE
    # positions = np.vstack([X.ravel(), Y.ravel()])
    #
    #
    # kde_traffic = gaussian_kde(coords, weights=z, bw_method=0.1) # traffic and station density
    # kde_stations = gaussian_kde(coords, bw_method=0.1) # station density
    #
    # # gaussian_traffic =
    # # Z = kde(positions)
    # if KDE:
    #     Z = kde_traffic(positions) / kde_stations(positions)
    # else:
    #     Z = Z_smooth
    #
    # Z = Z.reshape(X.shape)
    #
    #
    # grid_df = pd.DataFrame({
    #     "x": X.ravel(),
    #     "y": Y.ravel(),
    #     "weight": Z.ravel()
    # })
    # grid_df["weight"] = grid_df["weight"] * z.sum()
    # import seaborn as sns
    #
    #
    # fig, ax = plt.subplots(figsize=(8,8))
    #
    # # plot heatmap
    # im = ax.imshow(
    #     Z,
    #     extent=(X.min(), X.max(), Y.min(), Y.max()),
    #     origin="lower",
    #     cmap="hot",
    #     alpha=0.7
    # )
    # plt.show()
    #
