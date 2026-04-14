from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import pandas as pd
import numpy as np

def _plot_acf(ts, lags=30, file_dir=None):
    # plot_acf(ts_all['Tables']['Quantity'], lags=100)
    plot_acf(ts, lags=lags)
    plt.grid()
    if file_dir is None:
        plt.show()
    else:
        plt.savefig(file_dir)

def _plot_pacf(ts, lags=30, file_dir=None):
    # plot_acf(ts_all['Tables']['Quantity'], lags=100)
    plot_pacf(ts, lags=lags)
    plt.grid()
    if file_dir is None:
        plt.show()
    else:
        plt.savefig(file_dir)

def _plot_fft(ts):
    # Perform FFT
    # values = ts_all['Tables']['Quantity']
    values = ts.values
    fft = np.fft.fft(values)
    frequencies = np.fft.fftfreq(len(values))

    # Plot the power spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[:len(frequencies) // 2], np.abs(fft)[:len(frequencies) // 2])
    plt.title("Power Spectrum (FFT)")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()


def _plot_lombscargle(timeseries, values, plot=True, title='', file_dir=None):

    # Example: Generate irregular time series
    # irregular_time = df['timestamp'].values.astype('float64')

    try:
        irregular_time = timeseries.values.astype('float64')
    except:
        irregular_time = pd.to_numeric(timeseries)

    try:
        values = values.values
    except:
        pass

    # Lomb-Scargle Periodogram
    frequency, power = LombScargle(irregular_time, values).autopower()
    if plot:
        # Plot the periodogram
        plt.figure(figsize=(10, 6))
        plt.plot(1 / frequency, power)
        plt.xlim((0,10))
        plt.grid()
        plt.title(f"Lomb-Scargle Periodogram {title}")
        plt.xlabel("Period")
        plt.ylabel("Power")
        if file_dir is None:
            plt.show()
        else:
            plt.savefig(file_dir)

    df = pd.DataFrame({'frequency': frequency, 'power': power})
    df['period'] = 1 /df['frequency']
    # df.sort_values('power', ascending=False, inplace=True)
    return df