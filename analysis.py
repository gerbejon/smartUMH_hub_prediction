import pandas as pd

from genTS import ts_all, ts_all_clean, ts_index
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# def plot_acf(data):
#     plt.figure(figsize=(10, 5))
#     plot_acf(data, lags=20, alpha=0.05)
#     plt.title('Autocorrelation Function (ACF)')
#     plt.show()
#
# def plot_pacf(data):
#     plt.figure(figsize=(10, 5))
#     plot_pacf(data, lags=20, alpha=0.05, method='ywm')  # Use method='ywm' for stable computation
#     plt.title('Partial Autocorrelation Function (PACF)')
#     plt.show()



if __name__ == '__main__':
    df = pd.DataFrame([ts_index, ts_all, ts_all_clean], index=["INDEX", "TS", "TS_clean"]).transpose()
    df.set_index("INDEX", inplace=True)
    df.plot()
    plt.grid()
    plt.show()

    # result = seasonal_decompose(df.TS_clean, model='multiplicative', extrapolate_trend='freq')
    # result.plot()
    # plt.suptitle('Seasonal Decomposition of Air Passengers Time Series')
    # plt.tight_layout()
    # plt.show()
    #
    # result1 = seasonal_decompose(df.TS, model='multiplicative', extrapolate_trend='freq')
    # result1.plot()
    # plt.suptitle('Seasonal Decomposition of Air Passengers Time Series')
    # plt.tight_layout()
    # plt.show()
    #
    # result2 = seasonal_decompose(result1.trend, model='multiplicative', extrapolate_trend='freq')
    # result2.plot()
    # plt.suptitle('Seasonal Decomposition of Air Passengers Time Series')
    # plt.tight_layout()
    # plt.show()
    #
    # result3 = seasonal_decompose(result2.trend, model='multiplicative', extrapolate_trend='freq')
    # result3.plot()
    # plt.suptitle('Seasonal Decomposition of Air Passengers Time Series')
    # plt.tight_layout()
    # plt.show()
    #
    # result4 = seasonal_decompose(result3.trend, model='multiplicative', extrapolate_trend='freq')
    # result4.plot()
    # plt.suptitle('Seasonal Decomposition of Air Passengers Time Series')
    # plt.tight_layout()
    # plt.show()
    #
    # df = pd.concat([
    #     result1.observed,
    #     result1.seasonal,
    #     result2.seasonal,
    #     result3.seasonal,
    #     result4.seasonal,
    #     result4.trend,
    #     result1.resid + result2.resid + result3.resid + result4.resid
    # ], axis=1
    # )
    # result4.plot()
    # plt.suptitle('Seasonal Decomposition of Air Passengers Time Series')
    # plt.tight_layout()
    # plt.show()

