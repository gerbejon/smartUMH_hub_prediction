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
    def __init__(self):
        self.df = None
        self.data_path = os.path.join(datadir, 'verkehrszaehlung_{}_clean.csv')
        self.year = '2026'
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.data_path.format(self.year))
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].str.strip('"')




if __name__ == '__main__':
    ds = DataSource()
    df = ds.df
    df.loc[df.MessungDatZeit == '2026-01-08T00:00:00']