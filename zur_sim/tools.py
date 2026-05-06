from zur_sim.datasource import cwd
import os
import glob
import imageio


def aggreate(df):
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