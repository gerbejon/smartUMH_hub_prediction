import os
from datetime import timedelta, datetime
import numpy as np

from mockseries.trend import LinearTrend
from mockseries.seasonality import SinusoidalSeasonality
from mockseries.noise import RedNoise
from mockseries.utils import datetime_range, plot_timeseries
# from mockseries.utils import plot_timeseries, write_csv


class genTS:
    def __init__(self):
        # self.gen_one_series()
        # self.get_beds()
        self.round = False
        self.ts_a = None
        self.ts_b = None
        self.ts_c = None

        self.ts_index = datetime_range(
            granularity=timedelta(hours=1),
            start_time=datetime(2021, 5, 31),
            end_time=datetime(2021, 12, 30),
        )
        self.gen_ts_default()

    def get_abs_round(self, ts):
        return np.array([np.round(np.abs(item)) for item in ts])

    def gen_ts_default(self):
        self.gen_product_A()
        self.gen_product_B()
        self.gen_product_C()
        self.trend = self.get_trend()
        self.ts_all = self.ts_a + self.ts_b + self.ts_c
        self.ts_all_clean = self.ts_a_clean + self.ts_b_clean + self.ts_c_clean



    def get_trend(self, coef=2, time_unit= timedelta(days=4), flat_base=100):
        return LinearTrend(coefficient=coef, time_unit=time_unit, flat_base=flat_base)

    def get_seasonality(self, amplitude=20, period=timedelta(days=7)):
        return SinusoidalSeasonality(amplitude=amplitude, period=period)

    def gen_product_A(self):
        trend = self.get_trend( coef=2, time_unit= timedelta(days=4), flat_base=100)
        seasonality = self.get_seasonality(
            amplitude=5, period=timedelta(days=7)) #+ self.get_seasonality(
            #amplitude=4, period=timedelta(days=1))
        noise = RedNoise(mean=0, std=10, correlation=0.5)
        timeseries = trend + seasonality + noise

        ts_a = timeseries.generate(self.ts_index)
        ts_a_clean = (trend + seasonality).generate(self.ts_index)

        if self.round == True:
            ts_a = timeseries.generate(self.ts_index)
            self.ts_a = self.get_abs_round(ts_a)
            self.ts_a_clean = self.get_abs_round(ts_a_clean)
        else:
            self.ts_a = ts_a
            self.ts_a_clean = ts_a_clean
        # return ts_values, ts_index

    def gen_product_B(self):
        trend = self.get_trend( coef=-1, time_unit= timedelta(days=4), flat_base=100)
        seasonality = self.get_seasonality(
            amplitude=10, period=timedelta(days=28)
        ) #+ self.get_seasonality(
            # amplitude=1, period=timedelta(days=7))
        noise = RedNoise(mean=0, std=10, correlation=0.5)
        timeseries = trend + seasonality + noise


        ts_b = timeseries.generate(self.ts_index)
        ts_b_clean = (trend + seasonality).generate(self.ts_index)

        if self.round == True:
            self.ts_b = self.get_abs_round(ts_b)
            self.ts_b_clean = self.get_abs_round(ts_b_clean)
        else:
            self.ts_b = ts_b
            self.ts_b_clean = ts_b_clean

    def gen_product_C(self):
        trend = self.get_trend( coef=-1, time_unit= timedelta(days=10), flat_base=100)
        seasonality = self.get_seasonality(
            amplitude=15, period=timedelta(days=91)
        )# + self.get_seasonality(
            # amplitude=50, period=timedelta(days=28)
        # )
        noise = RedNoise(mean=0, std=10, correlation=0.5)
        timeseries = trend + seasonality + noise

        ts_c = timeseries.generate(self.ts_index)
        ts_c_clean = (trend + seasonality).generate(self.ts_index)

        if self.round == True:
            self.ts_c = self.get_abs_round(ts_c)
            self.ts_c_clean = self.get_abs_round(ts_c_clean)
        else:
            self.ts_c = ts_c
            self.ts_c_clean = ts_c_clean


# from mockseries.utils import plot_timeseries, write_csv
#
# print(ts_index, ts_values)
# plot_timeseries(ts_index, ts_values)
import pandas as pd
# import matplotlib.pyplot as plt
from plots.periodicity import _plot_lombscargle, _plot_acf, _plot_pacf
gents = genTS()
ts_all = gents.ts_all
ts_all_clean = gents.ts_all_clean
ts_index = gents.ts_index
from workbench.store_sales import WORK_DIR

def save_ts_plots(df):
    import matplotlib.pyplot as plt
    df.plot()
    # plt.scatter(df)
    plt.grid(True)
    plt.savefig(os.path.join(WORK_DIR, 'ts_example.png'))
    plt.show()

    # df = _plot_lombscargle(df.index, df.clean)
    from astropy.timeseries import LombScargle
    import matplotlib.pyplot as plt
    frequency, power = LombScargle(np.arange(0, len(df.index)), df.real).autopower()
    plt.plot(frequency, power)
    plt.title('Lomb-Scargle Periodogram')
    plt.grid(True)
    plt.savefig(os.path.join(WORK_DIR, 'ts_example_period.png'))
    plt.show()

    _plot_acf(df.real, 100, file_dir=f'{WORK_DIR}/ts_example_acf.png')
    _plot_pacf(df.real, 100, file_dir=f'{WORK_DIR}/ts_example_pacf.png')


def save_ts_diff_plots(df):
    import matplotlib.pyplot as plt
    df = df.diff().dropna()
    df.plot()
    # plt.scatter(df)
    plt.grid(True)
    plt.savefig(os.path.join(WORK_DIR, 'ts_diff_example.png'))
    plt.show()

    # df = _plot_lombscargle(df.index, df.clean)
    from astropy.timeseries import LombScargle
    import matplotlib.pyplot as plt
    frequency, power = LombScargle(np.arange(0, len(df.index)), df.real).autopower()
    plt.plot(frequency, power)
    plt.title('Lomb-Scargle Periodogram')
    plt.grid(True)
    plt.savefig(os.path.join(WORK_DIR, 'ts_diff_example_period.png'))
    plt.show()

    _plot_acf(df.real, 100, file_dir=f'{WORK_DIR}/ts_diff_example_acf_diff.png')
    plt.show()
    _plot_pacf(df.real, 100, file_dir=f'{WORK_DIR}/ts_diff_example_pacf.png')
    plt.show()

if __name__ == '__main__':
    # ts_values, ts_index = genTS().gen_one_series()
    gents = genTS()
    ts_all = gents.ts_all
    ts_all_clean = gents.ts_all_clean

    ts_index = gents.ts_index
    df = pd.DataFrame(
        [
            ts_all,
            ts_all_clean,
        ], columns= ts_index,
        index=['real', 'clean']
    ).transpose()

    save_ts_plots(df)
    save_ts_diff_plots(df)

    # plot_timeseries(ts_index, ts_all)
    # df.plot()
    # # plt.scatter(df)
    # plt.grid(True)
    # plt.savefig(os.path.join(WORK_DIR, 'ts_example.png'))
    # plt.show()
    #
    # # df = _plot_lombscargle(df.index, df.clean)
    # from astropy.timeseries import LombScargle
    # import matplotlib.pyplot as plt
    # frequency, power = LombScargle(np.arange(0,len(df.index)), df.real).autopower()
    # plt.plot(frequency, power)
    # plt.title('Lomb-Scargle Periodogram')
    # plt.grid(True)
    # plt.savefig(os.path.join(WORK_DIR, 'ts_example_period.png'))
    # plt.show()
    #
    #
    # _plot_acf(df.clean, 100, file_dir=f'{WORK_DIR}/ts_example_acf_diff.png')
    # _plot_pacf(df.clean, 100, file_dir=f'{WORK_DIR}/ts_example_pacf.png')


