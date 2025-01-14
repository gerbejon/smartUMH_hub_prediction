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

# Run auto_arima to find the best parameters
def get_furniture_ds():
    path = kagglehub.dataset_download("drrkkaushik/store-sales-forecasting-time-series-analysis")
    print("Path to dataset files:", path)
    df = pd.read_csv(os.path.join(path, 'stores_sales_forecasting.csv'), encoding='latin1')

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    # ds = df[['Order Date', 'Region', 'Product Name', 'Sales', 'Discount', 'Sub-Category']]

    ds = df[['Order Date', 'Sales', 'Profit', 'Sub-Category', 'Quantity']]
    ds_agg = ds.groupby(['Order Date', 'Sub-Category']).sum()

    all = []
    for categ in set(ds_agg.reset_index()['Sub-Category']):
        for date in pd.date_range(start=ds_agg.reset_index()['Order Date'].min(),
                                  end=ds_agg.reset_index()['Order Date'].max(), freq='D'):
            try:
                all.append([date, categ] + list(ds_agg.loc[(date, categ)]))
            except:
                all.append([date, categ] + [0, 0, 0])
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
    return ts_all

def get_gen_ds():
    df = pd.DataFrame([ts_index, ts_all, ts_all_clean], index=["INDEX", "TS", "TS_clean"]).transpose()
    df['TS'] = pd.to_numeric(df['TS'], errors='coerce')
    df['TS_clean'] = pd.to_numeric(df['TS_clean'], errors='coerce')

def fit_model(df, y_col, test_n=10):
    df.reset_index(inplace=True)
    train = df.loc[:(df.shape[0]-test_n)]
    test = df.loc[(df.shape[0]-test_n):]
    model = auto_arima(train[y_col], seasonal=True, m=7, trace=True)

    # p, d, q = 7,1,0
    p, d, q = model.order
    # P, D, Q, m = 1,0,0,31
    P, D, Q, m = model.seasonal_order


    # print(model.summary())
    sarima_model = sm.tsa.statespace.SARIMAX(
        train[y_col],
        order=(p, d, q),  # Replace with your optimal (p, d, q)
        seasonal_order=(P, D, Q, m),  # Replace with your optimal (P, D, Q, m)
        enforce_stationarity=False,
        enforce_invertibility=False
    )


    # Fit the model
    result = sarima_model.fit()
    train['Predicted'] = result.predict(start=train.index[0], end=train.index[-1])
    forecast = result.forecast(steps=len(test))
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train[y_col], label='Train Data', color='blue')
    # plt.plot(df.index, df['TS_clean'], label='Clean Data', color='black')
    plt.plot(train.index, train['Predicted'], label='SARIMAX Fit', color='green')
    plt.plot(test.index, test[y_col], label='Test Data', color='orange')
    plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='dashed')
    plt.xlim((df.shape[0] - 100, df.shape[0]))
    plt.title('Time Series and SARIMAX Prediction')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

from plots.periodicity import _plot_lombscargle, _plot_acf, _plot_pacf
if __name__ == '__main__':

    # df = get_gen_ds()
    ds_dict = get_furniture_ds()
    df = ds_dict['Tables']

    ls = _plot_lombscargle(timeseries=ds_dict['Chairs']['Order Date'], values=ds_dict['Chairs']['Quantity'])
    fit_model(ds_dict['Chairs'], 'Quantity')
    from astropy.timeseries import LombScargle
    frequency, power = LombScargle(
        pd.to_numeric(ds_dict['Chairs']['Order Date']),
        ds_dict['Chairs']['Quantity']).autopower(
        minimum_frequency=0.1,
        maximum_frequency=5
    )
    plt.plot(frequency, power)
    plt.show()

    _plot_lombscargle(timeseries=ds_dict['Tables']['Order Date'], values=ds_dict['Tables']['Quantity'])
    fit_model(ds_dict['Tables'], 'Quantity')
    _plot_lombscargle(timeseries=ds_dict['Bookcases']['Order Date'], values=ds_dict['Bookcases']['Quantity'])
    fit_model(ds_dict['Bookcases'], 'Quantity')
    _plot_lombscargle(timeseries=ds_dict['Furnishings']['Order Date'], values=ds_dict['Furnishings']['Quantity'])
    fit_model(ds_dict['Furnishings'], 'Quantity')

    # train = df.loc[:df.shape[0]-10]
    # test = df.loc[df.shape[0]-10:]
    # model = auto_arima(train['Quantity'], seasonal=True, m=28, trace=True)
    #
    # # p, d, q = 7,1,0
    # p, d, q = model.order
    # # P, D, Q, m = 1,0,0,31
    # P, D, Q, m = model.seasonal_order
    #
    #
    # # print(model.summary())
    # sarima_model = sm.tsa.statespace.SARIMAX(
    #     train['Quantity'],
    #     order=(p, d, q),  # Replace with your optimal (p, d, q)
    #     seasonal_order=(P, D, Q, m),  # Replace with your optimal (P, D, Q, m)
    #     enforce_stationarity=False,
    #     enforce_invertibility=False
    # )
    #
    #
    # # Fit the model
    # result = sarima_model.fit()
    # train['Predicted'] = result.predict(start=train.index[0], end=train.index[-1])
    # forecast = result.forecast(steps=len(test))
    # plt.figure(figsize=(12, 6))
    # plt.plot(train.index, train['TS'], label='Train Data', color='blue')
    # plt.plot(df.index, df['TS_clean'], label='Clean Data', color='black')
    # plt.plot(train.index, train['Predicted'], label='SARIMAX Fit', color='green')
    # plt.plot(test.index, test['TS'], label='Test Data', color='orange')
    # plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='dashed')
    # plt.xlim((df.shape[0] - 100, df.shape[0]))
    # plt.title('Time Series and SARIMAX Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()



    # plot_acf(df.TS_clean)