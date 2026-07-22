"""
Traffic density raster creation
===============================
Turns the point measurements of the Zurich detector network (vehicle counts at
discrete stations) into a smooth, regular grid ("raster") of traffic density.
Each station carries an ``AnzFahrzeuge`` (vehicle count) at fixed coordinates;
this module spreads those counts across a 2-D grid so that the simulation has a
continuous density surface to sample from and to paint as a heatmap.

Two smoothing strategies are offered: a weighted Gaussian kernel-density
estimate (KDE) that normalises traffic density by station density, and a simpler
nearest-neighbour interpolation followed by a Gaussian filter. Coordinates are
reprojected from Swiss LV95 (EPSG:2056) to Web Mercator (EPSG:3857) so the grid
lines up with standard map tiles.
"""

import socket
if socket.gethostname() == 'berttrainer-large':
    from datasource import DataSource
    from plots import interactive_plot
    from tools import aggreate
else:
    from zur_sim.datasource import DataSource
    from zur_sim.plots import interactive_plot
    from zur_sim.tools import aggreate

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


class RasterCreater:
    """Build a smooth traffic-density grid from point measurements.

    Given a DataFrame of detector stations (one row per station, with
    ``EKoord``/``NKoord`` coordinates and an ``AnzFahrzeuge`` vehicle count),
    this class produces ``grid_df`` -- a long-format DataFrame with columns
    ``x``, ``y`` and ``weight`` holding the estimated density on a regular
    ``grid_res`` x ``grid_res`` grid. It can also map that grid back onto each
    station to obtain a smoothed per-node density (``get_density_per_node``).
    """

    def __init__(self, df, kde_bw=0.3):
        """Store inputs and grid parameters.

        Parameters
        ----------
        df : pandas.DataFrame
            Station measurements with ``EKoord``, ``NKoord`` and
            ``AnzFahrzeuge`` columns.
        kde_bw : float, default 0.3
            Bandwidth passed to ``scipy.stats.gaussian_kde`` (``bw_method``);
            smaller values give a tighter, more peaked density.
        """
        self.KDE = False
        self.df = df
        self.grid_res = 200
        self.kde_bw = kde_bw

    def main_kde(self, df=None):
        """Compute the density grid with the KDE strategy.

        Runs ``gaussian_kde`` and stores the resulting long-format grid in
        ``self.grid_df``. Falls back to ``self.df`` when ``df`` is None.
        """
        if df is None:
            df = self.df
        X,Y,Z = self.gaussian_kde(df)
        self.grid_df = self.postprocess(X, Y, Z)
        # self.plot(X, Y, Z)

    def main_gf(self, df=None):
        """Compute the density grid with the Gaussian-filter strategy.

        Runs nearest-neighbour interpolation plus a Gaussian filter and stores
        the resulting long-format grid in ``self.grid_df``. Falls back to
        ``self.df`` when ``df`` is None.
        """
        if df is None:
            df = self.df
        X,Y,Z = self.gaussan_filter(df)
        self.grid_df = self.postprocess(X, Y, Z)
        # self.plot(X, Y, Z)

    def preprocess(self, df, z_col='AnzFahrzeuge'):
        """Reproject stations and build the empty target grid.

        Converts the station points from Swiss LV95 (EPSG:2056) to Web Mercator
        (EPSG:3857), caches the station coordinates and values in ``self.x``,
        ``self.y`` and ``self.z``, and builds a ``grid_res`` x ``grid_res``
        meshgrid spanning the station bounding box.

        Parameters
        ----------
        z_col : str, default 'AnzFahrzeuge'
            Column holding the value (traffic count) to be gridded.

        Returns
        -------
        (X, Y) : tuple of numpy.ndarray
            The meshgrid coordinate arrays of the target grid.
        """
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
        """Grid traffic via nearest-neighbour interpolation + Gaussian blur.

        Interpolates the station counts onto the grid using nearest-neighbour
        (filling any gaps with 0), then smooths with a Gaussian filter
        (``sigma=3``).

        Returns
        -------
        (X, Y, Z) : tuple of numpy.ndarray
            Grid coordinates and the smoothed density surface.
        """
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
        """Grid traffic via a station-normalised weighted KDE.

        Fits two Gaussian KDEs over the station coordinates: one weighted by the
        traffic counts and one unweighted (station density). The returned
        surface is the ratio ``kde_traffic / kde_stations``, i.e. traffic per
        station, so densely instrumented areas are not over-counted.

        Returns
        -------
        (X, Y, Z) : tuple of numpy.ndarray
            Grid coordinates and the density surface reshaped to the grid.
        """
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
        """Flatten the grid to a DataFrame and rescale to total traffic.

        Turns the 2-D grid arrays into a long-format DataFrame with columns
        ``x``, ``y`` and ``weight``, then rescales ``weight`` by the total
        vehicle count (``self.z.sum()``) so the surface carries absolute
        traffic units rather than a normalised density.

        Returns
        -------
        pandas.DataFrame
            The grid in long (one-row-per-cell) form.
        """
        grid_df = pd.DataFrame({
            "x": X.ravel(),
            "y": Y.ravel(),
            "weight": Z.ravel()
        })
        grid_df["weight"] = grid_df["weight"] * self.z.sum()
        return grid_df

    def plot(self, X,Y,Z):
        """Show the density surface ``Z`` as a hot-colormap heatmap."""
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
        """Sample the smoothed grid back onto each station.

        For every station, finds the nearest grid cell (in reprojected Web
        Mercator coordinates) and takes its ``weight``. The per-station values
        are then renormalised so they sum to the original total vehicle count
        (``df.AnzFahrzeuge.sum()``), yielding a smoothed vehicle count per node.

        Requires a grid to have been built first (``main_kde``/``main_gf``).

        Returns
        -------
        numpy.ndarray
            Smoothed vehicle count per station, aligned with ``df``'s rows.
        """
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
