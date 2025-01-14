# from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import pandas as pd
from genTS import ts_all, ts_all_clean, ts_index
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import kagglehub
import os
import seaborn as sns
from astropy.timeseries import LombScargle
from plots.periodicity import _plot_lombscargle


path = kagglehub.dataset_download("drrkkaushik/store-sales-forecasting-time-series-analysis")
print("Path to dataset files:", path)
df = pd.read_csv(os.path.join(path, 'stores_sales_forecasting.csv'), encoding='latin1')

df['Order Date'] = pd.to_datetime(df['Order Date'])
# ds = df[['Order Date', 'Region', 'Product Name', 'Sales', 'Discount', 'Sub-Category']]

df = df.loc[df['Sub-Category'] == 'Tables']
df.sort_values('Order Date', inplace=True)
df['Order Date num'] = pd.to_numeric(df['Order Date']) / 10**10
ls = _plot_lombscargle(df['Order Date num'], df['Quantity'])
ls.sort_values('power', ascending=False, inplace=True)
ls['period_2pi'] = ls['period']
ls['period'] = ls['period_2pi'] / (2*np.pi)
print(ls[['period', 'frequency', 'power']])


plt.plot(pd.to_numeric(df['Order Date num']), df['Quantity'], color='orange', label='Quanity')
x = np.arange(
    pd.to_numeric(df['Order Date num']).min(),
    pd.to_numeric(df['Order Date num']).max(), 50000)
y = np.sin(ls['period_2pi'][0] * x)
plt.plot(x, y + df['Quantity'].mean(), color='red', label='Mean')
plt.grid()
plt.show()

