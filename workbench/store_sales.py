import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from plots.periodicity import _plot_lombscargle, _plot_pacf, _plot_acf
# from models import fit_model
import statsmodels.api as sm
import numpy as np

def get_data():
    path = kagglehub.dataset_download("drrkkaushik/store-sales-forecasting-time-series-analysis")
    print("Path to dataset files:", path)
    df = pd.read_csv(os.path.join(path, 'stores_sales_forecasting.csv'), encoding='latin1')

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df
    # ds = df[['Order Date', 'Region', 'Product Name', 'Sales', 'Discount', 'Sub-Category']]
# df['Region']

def plot_ts(df_tmp, x, y):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=x,
        y=y,
        # hue='Sub-Category',
        palette='deep',  # You can customize the color palette
        data=df_tmp,
        # s=100  # Marker size
    )
    plt.show()

def get_time_series(sub_group = None):
    df = get_data()
    if sub_group is not None:
        df = df.loc[df[sub_group] == sub_group]
    df = df[['Order Date', 'Sales', 'Quantity', 'Discount', 'Profit']]
    ds_agg = df.groupby(['Order Date']).sum()
    all = []
    for date in pd.date_range(start=ds_agg.reset_index()['Order Date'].min(),
                              end=ds_agg.reset_index()['Order Date'].max(), freq='D'):
        try:
            all.append([date] + list(ds_agg.loc[date]))
        except:
            all.append([date] + [0, 0, 0, 0])
    return pd.DataFrame(all, columns=['Order Date', 'Sales', 'Quantity', 'Discount', 'Profit'])

def get_diff(ds):
    df = ds.set_index('Order Date')
    return df.diff().dropna().reset_index()

def ts_analyis(plot=True, y_col = 'Sales', **kwargs):
    ts = get_time_series(kwargs)
    if plot:
        plot_ts(ts.iloc[-30:, :], 'Order Date', y_col)
        _plot_lombscargle(ts['Order Date'], ts[y_col], plot=plot, title=y_col)
    ts_diff = get_diff(ts)
    # _plot_lombscargle(ts_diff['Order Date'], ts_diff[y_col])
    if plot:
        ls = _plot_lombscargle(ts_diff.index, ts_diff[y_col], title='sales returns')
        _plot_acf(ts_diff[y_col])
        _plot_pacf(ts_diff[y_col])
    return ts_diff, ts

def fit_model_rolling_prediction(df, y_col, test_n = 10):

    # df.reset_index(inplace=True)
    train = df.loc[:(df.shape[0]-test_n)]
    test = df.loc[(df.shape[0]-test_n+1):]
    model = auto_arima(train[y_col], seasonal=True, m=7, trace=True)

    p, d, q = model.order
    # p, d, q = 0,0,1
    P, D, Q, m = model.seasonal_order
    # P, D, Q, m = 0,0,0,0


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

    # Rolling Prediction
    predictions = []
    current_state = result
    for ind, new_obs in test.iterrows():
        # Forecast the next value
        # forecast = current_state.forecast(steps=1)
        forecast = current_state.forecast(steps=1)
        test.loc[forecast.index[0], 'forecast_sales'] = forecast.values[0]
        # predictions.append(forecast)

        # Update the model state with the new observation
        current_state = current_state.append([new_obs[y_col]], refit=False)

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train[y_col], label='Train Data', color='blue')
    # plt.plot(df.index, df['TS_clean'], label='Clean Data', color='black')
    plt.plot(train.index, train['Predicted'], label='SARIMAX Fit', color='green')
    plt.plot(test.index, test[y_col], label='Test Data', color='orange')
    plt.plot(test.index, test['forecast_sales'], label='Forecast', color='red', linestyle='dashed')
    plt.xlim((df.shape[0] - 100, df.shape[0]))
    plt.grid()
    plt.title('Time Series and SARIMAX Prediction')
    plt.suptitle('asdfasdf')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    # predicted_series = pd.Series(predictions, index=np.arange(len(train), len(train) + len(test)))

    # Optional: Compare predictions with actual test data


    # for i in range(test_n):
    #     train = df.loc[:(df.shape[0]-(test_n + i))]
    #     test = df.loc[(df.shape[0]-(test_n + i)):]


if __name__ == '__main__':
    df = get_data()
    for sg in df.groupby('Sub-Category')['Sub-Category'].count().index:
        ts_diff, ts_ts = ts_analyis(plot=False, sub_group=sg)
        fit_model_rolling_prediction(ts_diff, y_col='Sales', test_n = 50)