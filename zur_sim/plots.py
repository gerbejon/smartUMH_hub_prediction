from zur_sim.datasource import DataSource, datadir
import pandas as pd
import geopandas as gpd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import contextily as cx
import os
from progressbar import progressbar
import imageio
import glob
import matplotlib.colors as mcolors
import numpy as np

def interactive_plot(df, vmin=None, vmax=None, kind=None):

    # create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.EKoord, df.NKoord),
        crs="EPSG:2056"
    )

    # convert to lat/lon (required for Plotly)
    gdf = gdf.to_crs(4326)

    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x
    # hover_cols = ['MSID', 'MSName', 'ZSID', 'ZSName', 'Achse', 'HNr', 'Hoehe', 'EKoord',
    #    'NKoord', 'Richtung', 'Knummer', 'Kname', 'AnzDetektoren', 'D1ID',
    #    'D2ID', 'D3ID', 'D4ID', 'AnzFahrzeuge']
    hover_cols = df.columns
    fig = px.scatter_mapbox(
        gdf,
        lat="lat",
        lon="lon",

        color="AnzFahrzeuge",
        color_continuous_scale="Hot",
        size="AnzFahrzeuge",
        size_max=15,
        zoom=12,
        hover_data=hover_cols
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        title=df.MessungDatZeit.iloc[0].replace('T', ' ')
    )

    fig.show(renderer="browser")


def plot(df, vmin=None, vmax=None, kind=None):
    if vmax is None:
        vmax=df.AnzFahrzeuge.max()
    if vmin is None:
        vmin=df.AnzFahrzeuge.min()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.hot

    # create GeoDataFrame using Swiss LV95 coordinates
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.EKoord, df.NKoord),
        crs="EPSG:2056"   # Swiss LV95
    )

    # convert to web mercator for map tiles
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8,8))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    if kind == 'scatter':

        sns.scatterplot(
            x=gdf.geometry.x,
            y=gdf.geometry.y,
            hue=gdf["AnzFahrzeuge"],   # whatever attribute you plot
            palette="hot",
            ax=ax,
            edgecolor = "black",
            hue_norm=norm,
            legend=False
            # title = df.MessungDatZeit.iloc[0].replace('T', ' '),
        )
        cx.add_basemap(ax)

        plt.title(df.MessungDatZeit.iloc[0].replace('T', ' '))
        plt.show()

    if kind == 'heatmap':
        sns.kdeplot(
            x=gdf.geometry.x,
            y=gdf.geometry.y,
            weights=gdf["AnzFahrzeuge"],
            fill=True,
            cmap="Reds",
            bw_adjust=0.5,
            levels=50,
            ax=ax,
            alpha=0.5,
            thresh=0
        )
        cx.add_basemap(ax)

        plt.title(df.MessungDatZeit.iloc[0].replace('T', ' '))
        plt.show()

    plt.savefig(os.path.join(datadir, 'plots', f"map_plot_{df.MessungDatZeit.iloc[0]}.png"), dpi=300, bbox_inches="tight")
    plt.close()



def create_frames_per_day(day='2026-01-01', kind=None):
    ds = DataSource()
    df = ds.df.loc[ds.df.MessungDatZeit.str.startswith(day)]
    timestamps = df.MessungDatZeit.unique()
    for timestamp in progressbar(timestamps):
        df_sub = df.loc[df.MessungDatZeit == timestamp]
        plot(df_sub, vmin=0, vmax=df.AnzFahrzeuge.max(), kind=kind)

def create_video_frames_per_day(day='2026-01-01', kind=None):
    frames = sorted(glob.glob(f"{datadir}/plots/map_plot_{day}*.png"))
    with imageio.get_writer(os.path.join(datadir, 'plots', f"zurich_traffic_{kind}_{day}.mp4"),
                            fps=6) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)



if __name__ == '__main__':
    date = '2026-01-08'
    # date = '2026-01-01'
    # kind = 'heatmap'
    kind = 'scatter'
    ds = DataSource()
    df = ds.df.loc[ds.df.MessungDatZeit == '2026-01-08T12:00:00']

    plot(df, kind=kind)


    # create_frames_per_day(day=date, kind=kind)
    # create_video_frames_per_day(day=date, kind=kind)

