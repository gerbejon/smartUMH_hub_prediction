"""
Traffic-data helper tools
=========================
Small utilities used by the Zurich simulation: aggregating the raw per-timestamp
counting-station table down to one row per station, and stitching per-day PNG map
frames into an MP4 animation.
"""

from datasource import cwd
import os
import glob
import imageio


def aggreate(df):
    """Collapse the raw count table to one row per counting station (ZSID).

    Groups by ZSID and averages the numeric fields that vary over time
    (vehicle count and E/N coordinates) while keeping the first value for the
    static descriptive fields (name, height, timestamp). Returns the aggregated
    DataFrame with ZSID as an ordinary column.
    """
    return df[['ZSID', 'ZSName', 'EKoord', 'Hoehe',
                     'NKoord', 'AnzFahrzeuge', 'MessungDatZeit']].groupby('ZSID').agg(
            {'AnzFahrzeuge': 'mean',
             'EKoord': 'mean',
             'NKoord': 'mean',
             'ZSName': 'first',
             'Hoehe': 'first',
             'MessungDatZeit': 'first'}
        ).reset_index()


def create_video_frames_per_day(folder=None):
    """Assemble the `sim_map_*.png` frames in a day's plot folder into an MP4.

    `folder` is a day sub-folder name under the hard-coded plots directory; when
    None it defaults to the `2026-01-06` folder. Frames are read in sorted
    (chronological) order and written to `zurich_traffic_simulation.mp4` in the
    same folder at 3 fps.
    """
    if folder is None:
        folder = os.path.join('/home/gerj/Documents/playground/smartUMH/zur_sim/plots', '2026-01-06')
    else:
        folder = os.path.join('/home/gerj/Documents/playground/smartUMH/zur_sim/plots', folder)
    frames = sorted(glob.glob(f"{folder}/sim_map_*.png"))

    with imageio.get_writer(os.path.join(folder,  f"zurich_traffic_simulation.mp4"), fps=3) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

if __name__ == '__main__':
    create_video_frames_per_day()