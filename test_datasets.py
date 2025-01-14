import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import kagglehub
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plots.periodicity import _plot_lombscargle, _plot_acf, _plot_pacf

# Stock marked
ds = pm.datasets.load_msft()
ds['Date'] = pd.to_datetime(ds['Date'])
plt.plot(ds.Date, ds.Volume)
plt.show()

# Download latest version BigMart Product Sales Factors
path = kagglehub.dataset_download("thedevastator/bigmart-product-sales-factors")
print("Path to dataset files:", path)
df = pd.read_csv(os.path.join(path, 'data.csv'))

# Download latest version US Regional Sales Data
path = kagglehub.dataset_download("talhabu/us-regional-sales-data")
print("Path to dataset files:", path)
df = pd.read_csv(os.path.join(path, 'US_Regional_Sales_Data.csv'))
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
data = df[['_ProductID', 'OrderDate', '_StoreID', 'Discount Applied', 'Unit Price', 'Sales Channel', 'Order Quantity']]
prod12 = data.query('_ProductID == 12')


# Download latest version stores_sales_forecasting furniture
path = kagglehub.dataset_download("drrkkaushik/store-sales-forecasting-time-series-analysis")
print("Path to dataset files:", path)
df = pd.read_csv(os.path.join(path, 'stores_sales_forecasting.csv'), encoding='latin1')

df['Order Date'] = pd.to_datetime(df['Order Date'])
# ds = df[['Order Date', 'Region', 'Product Name', 'Sales', 'Discount', 'Sub-Category']]

ds = df[['Order Date', 'Sales', 'Profit', 'Sub-Category', 'Quantity']]
ds_agg = ds.groupby(['Order Date', 'Sub-Category']).sum().reset_index()



ds_agg = ds.groupby(['Order Date', 'Sub-Category']).sum()
all = []
for categ in set(ds_agg.reset_index()['Sub-Category']):
    for date in pd.date_range(start=ds_agg.reset_index()['Order Date'].min(), end=ds_agg.reset_index()['Order Date'].max(), freq='D'):
        try:
            all.append([date, categ] + list(ds_agg.loc[(date, categ)]))
        except:
            all.append([date, categ] + [0,0,0])
df_all = pd.DataFrame.from_records(all, columns=['Order Date', 'Sub-Category', 'Sales', 'Profit', 'Quantity'])

ts_all = {}

for categ in set(df_all['Sub-Category']):
    df_tmp = df_all.loc[df_all['Sub-Category'] == categ]
    ts_all[categ] = df_tmp
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x='Order Date',
        y='Quantity',
        hue='Sub-Category',
        palette='deep',  # You can customize the color palette
        data=df_tmp,
        # s=100  # Marker size
    )
    plt.show()

# plot autocorrelation
# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(ts_all['Tables']['Quantity'], lags=100)
# plt.show()

# plot fft
# Perform FFT
# values = ts_all['Tables']['Quantity']
# fft = np.fft.fft(values)
# frequencies = np.fft.fftfreq(len(values))
#
# # Plot the power spectrum
# plt.figure(figsize=(10, 6))
# plt.plot(frequencies[:len(frequencies)//2], np.abs(fft)[:len(frequencies)//2])
# plt.title("Power Spectrum (FFT)")
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude")
# plt.show()




# from astropy.timeseries import LombScargle
#
# # Example: Generate irregular time series
# # irregular_time = df['timestamp'].values.astype('float64')
# irregular_time = ts_all['Tables']['Order Date'].values.astype('float64')
# values = ts_all['Tables']['Quantity'].values
#
# # Lomb-Scargle Periodogram
# frequency, power = LombScargle(irregular_time, values).autopower()
#
# # Plot the periodogram
# plt.figure(figsize=(10, 6))
# plt.plot(1 / frequency, power)
# plt.grid()
# plt.title("Lomb-Scargle Periodogram")
# plt.xlabel("Period")
# plt.ylabel("Power")
# plt.show()
df = _plot_lombscargle(ts_all['Tables']['Order Date'], ts_all['Tables']['Quantity'])

_plot_acf(ts_all['Tables']['Quantity'], lags=21)
_plot_pacf(ts_all['Tables']['Quantity'], lags=21)