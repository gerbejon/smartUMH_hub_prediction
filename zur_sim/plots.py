"""
Traffic map plotting and animation
==================================
Renders the Zurich counting-station data on a geographic map, colouring and
sizing stations by vehicle count (`AnzFahrzeuge`). Provides an interactive
Plotly map, static Matplotlib scatter/heatmap snapshots saved to disk, and
helpers that batch-render one frame per timestamp for a day and stitch them into
an MP4. Coordinates arrive in Swiss LV95 (EPSG:2056) and are reprojected to
WGS84 (Plotly) or Web Mercator (map tiles) as needed.
"""

import socket
if socket.gethostname() == 'berttrainer-large':
    from datasource import DataSource, datadir
else:
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
    """Open an interactive Plotly map of the stations in `df` in the browser.

    Builds a GeoDataFrame from the LV95 E/N coordinates, reprojects to WGS84
    lat/lon (required by Plotly), and draws a Mapbox scatter map where colour and
    marker size both encode `AnzFahrzeuge`. All DataFrame columns are shown on
    hover and the title is the first row's timestamp. `vmin`, `vmax` and `kind`
    are accepted for signature parity but not used here.
    """

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


def plot(df, vmin=None, vmax=None, kind=None, node_labels=None):
    """Render a static Matplotlib map of the stations and save it as a PNG.

    Reprojects the LV95 coordinates to Web Mercator so a basemap can be added.
    `kind='scatter'` draws points coloured by `AnzFahrzeuge`; `kind='heatmap'`
    draws a count-weighted KDE density surface. `vmin`/`vmax` fix the colour
    scale (defaulting to the data's min/max) so frames across a day stay
    comparable. The figure is always written to
    `<datadir>/plots/map_plot_<timestamp>.png` and then closed. `node_labels` is
    accepted but unused.
    """
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
    """Render one static map PNG per timestamp for the given `day`.

    Loads the full dataset, keeps rows whose timestamp starts with `day`, and
    calls `plot` for each unique timestamp using a shared colour scale
    (vmin=0, vmax = the day's maximum count) so the frames are comparable.
    `kind` selects the plot style ('scatter' or 'heatmap').
    """
    ds = DataSource()
    df = ds.df.loc[ds.df.MessungDatZeit.str.startswith(day)]
    timestamps = df.MessungDatZeit.unique()
    for timestamp in progressbar(timestamps):
        df_sub = df.loc[df.MessungDatZeit == timestamp]
        plot(df_sub, vmin=0, vmax=df.AnzFahrzeuge.max(), kind=kind)

def create_video_frames_per_day(day='2026-01-01', kind=None):
    """Stitch a day's saved `map_plot_<day>*.png` frames into an MP4.

    Reads the matching PNGs in sorted (chronological) order and writes
    `zurich_traffic_<kind>_<day>.mp4` into `<datadir>/plots/` at 6 fps. Assumes
    `create_frames_per_day` has already produced the frames.
    """
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

