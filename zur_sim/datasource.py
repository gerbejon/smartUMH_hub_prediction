"""
Traffic-count data source
==========================
Locates the project root and loads the raw Zurich MIV (motorized individual
traffic) counting-station CSV for a given year. Walking up the current working
directory until the `smartUMH` folder is found lets the module be imported from
anywhere in the repo while still resolving the shared `zur_sim/data/` directory.
"""

import pandas as pd
import os


# cwd = os.path.join([folder for folder in os.getcwd()])
cwd = '/'
for folder in os.getcwd().split('/'):
    cwd = os.path.join(cwd, folder)
    if folder == 'smartUMH':
        break

datadir = os.path.join(cwd, 'zur_sim/data/')

# cwd = '/home/gerj/Documents/playground/smartUMH'

class DataSource:
    """Loads and holds the yearly traffic-count table as a pandas DataFrame.

    On construction it immediately reads the CSV for `self.year` into `self.df`.
    The raw table contains per-station, per-timestamp vehicle counts with columns
    such as ZSID (station id), MessungDatZeit (measurement timestamp),
    AnzFahrzeuge (vehicle count), and E/N coordinates.
    """

    def __init__(self):
        """Set up file paths and eagerly load the data for the default year."""
        self.df = None
        self.data_path = os.path.join(datadir, 'sid_dav_verkehrszaehlung_miv_od2031_{}.csv')
        self.year = '2026'
        self.load_data()

    def load_data(self):
        """Read the CSV for `self.year` into `self.df`.

        `self.data_path` is a format template; the year is substituted to build
        the concrete filename before reading.
        """
        self.df = pd.read_csv(self.data_path.format(self.year))




if __name__ == '__main__':
    ds = DataSource()
    df = ds.df
    df.loc[df.MessungDatZeit == '2026-01-08T00:00:00']