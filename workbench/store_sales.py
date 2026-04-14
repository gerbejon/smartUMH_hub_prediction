import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.defchararray import lower
from pmdarima import auto_arima
from plots.periodicity import _plot_lombscargle, _plot_pacf, _plot_acf
# from models import fit_model
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.mosaicplot import mosaic

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

def get_time_series(kwargs):
    df = get_data()
    if 'sub_group' in list(kwargs.keys()):
        sub_group = kwargs['sub_group']
        print("Sub-group:", sub_group)
        # sub_group = kwargs['sub_group']
        df = df.loc[df['Sub-Category'] == sub_group]
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

def fit_model_rolling_prediction(y_col, test_n = 10, timeseries_title='all'):
    if timeseries_title == 'all':
        ts_diff, ts_ts = ts_analyis(plot=False)
    else:
        ts_diff, ts_ts = ts_analyis(plot=False, sub_group=timeseries_title)
    df = ts_diff.copy()
    # df.reset_index(inplace=True)
    train_r = df.loc[:(df.shape[0]-test_n)]
    test_r = df.loc[(df.shape[0]-test_n+1):]
    # model = auto_arima(train_r[y_col], seasonal=True, m=7, trace=True, information_criterion='bic')

    # p, d, q = model.order
    p, d, q = 0,0,1
    # P, D, Q, m = model.seasonal_order
    P, D, Q, m = 0,0,0,0


    sarima_model = sm.tsa.statespace.SARIMAX(
        train_r[y_col],
        order=(p, d, q),  # Replace with your optimal (p, d, q)
        seasonal_order=(P, D, Q, m),  # Replace with your optimal (P, D, Q, m)
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    # Fit the model
    result = sarima_model.fit()
    train_r['Predicted'] = result.predict(start=train_r.index[0], end=train_r.index[-1])

    # Rolling Prediction
    predictions = []
    current_state = result
    for ind, new_obs in test_r.iterrows():
        # Forecast the next value
        # forecast = current_state.forecast(steps=1)
        forecast = current_state.forecast(steps=1)
        test_r.loc[forecast.index[0], 'Predicted'] = forecast.values[0]
        # predictions.append(forecast)

        # Update the model state with the new observation
        current_state = current_state.append([new_obs[y_col]], refit=False)



    # plot returns
    plt.figure(figsize=(12, 6))
    plt.plot(train_r.index, train_r[y_col], label='Train Data', color='blue')
    # plt.plot(df.index, df['TS_clean'], label='Clean Data', color='black')
    plt.plot(train_r.index, train_r['Predicted'], label='SARIMAX Fit', color='green')
    plt.plot(test_r.index, test_r[y_col], label='Test Data', color='orange')
    plt.plot(test_r.index, test_r['Predicted'], label='Forecast', color='red', linestyle='dashed')
    plt.xlim((df.shape[0] - 100, df.shape[0]))
    plt.grid()
    plt.suptitle(f'Time Series and SARIMAX Prediction of {timeseries_title} returns', fontsize=14)
    plt.title(f'SARIMA({p},{d},{q})({P},{D},{Q})[{m}]', )
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    # plt.show()
    if SAVE_FIG:
        plt.savefig(f'{DATASET_DIR}{y_col}/ts_{timeseries_title}_returns.png')
        _plot_acf(train_r[y_col], file_dir=f'{DATASET_DIR}{y_col}/ts_{timeseries_title}_returns_acf.png')
        _plot_pacf(train_r[y_col], file_dir=f'{DATASET_DIR}{y_col}/ts_{timeseries_title}_returns_pacf.png')
        _plot_lombscargle(train_r[y_col].index, train_r[y_col], file_dir=f'{DATASET_DIR}{y_col}/ts_{timeseries_title}_returns_lombscargle.png')

    else:
        plt.show()
        _plot_acf(train_r[y_col])
        _plot_pacf(train_r[y_col])


    # plot timeseries
    train, test = ts_ts.loc[train_r.index], ts_ts.loc[test_r.index]
    train.loc[train_r.index, 'Predicted'] = (train.loc[train_r.index, y_col] + train_r.loc[train_r.index, 'Predicted']).shift(1)
    test.loc[test_r.index, 'Predicted'] = (test.loc[test_r.index, y_col] + test_r.loc[test_r.index, 'Predicted']).shift(1)

    # cut negative sales
    train.loc[train['Predicted'] < 0, 'Predicted'] = 0
    test.loc[test['Predicted'] < 0, 'Predicted'] = 0

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train[y_col], label='Train Data', color='blue')
    # plt.plot(df.index, df['TS_clean'], label='Clean Data', color='black')
    plt.plot(train.index, train['Predicted'], label='SARIMAX Fit', color='green')
    plt.plot(test.index, test[y_col], label='Test Data', color='orange')
    plt.plot(test.index, test['Predicted'], label='Forecast', color='red', linestyle='dashed')
    plt.xlim((ts_ts.shape[0] - 2*test_n, ts_ts.shape[0]))
    plt.grid()
    plt.suptitle(f'Time Series and SARIMAX Prediction of {timeseries_title} ', fontsize=16)
    plt.title(f'SARIMA({p},{d},{q})({P},{D},{Q})[{m}]', )
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    # plt.show()
    if SAVE_FIG:
        plt.savefig(f'{DATASET_DIR}{y_col}/ts_{timeseries_title}.png')
        _plot_acf(train[y_col], file_dir=f'{DATASET_DIR}{y_col}/ts_{timeseries_title}_acf.png')
        _plot_pacf(train[y_col], file_dir=f'{DATASET_DIR}{y_col}/ts_{timeseries_title}_pacf.png')
        _plot_lombscargle(train[y_col].index, train_r[y_col], file_dir=f'{DATASET_DIR}{y_col}/ts_{timeseries_title}_lombscargle.png')
    else:
        plt.show()
        _plot_acf(train[y_col])
        _plot_pacf(train[y_col])



    mae =  np.mean(np.abs(test_r['Predicted'] - test_r[y_col]))
    mae_naive = np.mean(np.abs(test_r[y_col].shift() - test_r[y_col]))
    mse = np.mean((test_r['Predicted'] - test_r[y_col])**2)
    mape = np.mean((np.abs((test_r['Predicted'] - test_r[y_col]) / test_r[y_col])).replace([np.inf, -np.inf], np.nan).dropna()) * 100
    metrics = {
        'mae' : np.mean(np.abs(test_r['Predicted'] - test_r[y_col])),
        'mase' : mae_naive/mae,
        'mse': mse,
        'mape': mape

    }

    return train_r, test_r, train, test, metrics




    # predicted_series = pd.Series(predictions, index=np.arange(len(train), len(train) + len(test)))

    # Optional: Compare predictions with actual test data


    # for i in range(test_n):
    #     train = df.loc[:(df.shape[0]-(test_n + i))]
    #     test = df.loc[(df.shape[0]-(test_n + i)):]

def check_correlations(category=None):
    df_all_ts = None
    for sg in df.groupby('Sub-Category')['Sub-Category'].count().index:
        ts_diff, ts_ts = ts_analyis(plot=False, sub_group=sg)
        ts_ts.set_index('Order Date', inplace=True)
        # ts_ts.reset_index(inplace=True)
        ts_ts.columns = ts_ts.columns + f'_{sg}'
        if df_all_ts is None:
            df_all_ts = ts_ts
        else:
            df_all_ts = pd.concat([df_all_ts, ts_ts], axis=1)
    if category is not None:
        df_ts = df_all_ts.loc[:, [col for col in df_all_ts.columns if col.__contains__(category)]]
        return df_ts
    else:
        return df_all_ts

def plot_correlations(df):
    labels = [col.split('_')[1] for col in df.columns]
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.title(f"Correlation of {df.columns[0].split('_')[0]}")
    if SAVE_FIG:
        plt.savefig(f'{DATASET_DIR}/corr_{df.columns[0].split("_")[0].lower()}_.png')
    else:
        plt.show()
def run_one_timeseries(y_col='Sales', test_n=50, timeseries_title='Furnishings'):
    train_r, test_r, train, test, metrics = fit_model_rolling_prediction(y_col=y_col, test_n=test_n, timeseries_title=timeseries_title)

def main():
    y_cols = ['Sales', 'Quantity',]
    df = get_data()
    ts_diff, ts_ts = ts_analyis(plot=False)

    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)
    for y_col in y_cols:

        if not os.path.exists(f'{DATASET_DIR}{y_col}'):
            os.mkdir(f'{DATASET_DIR}{y_col}')
        train_r, test_r, train, test, metrics = fit_model_rolling_prediction(y_col=y_col, test_n=70)
        metric_dict = {
            'all' : metrics
        }
        for sg in df.groupby('Sub-Category')['Sub-Category'].count().index:
            ts_diff, ts_ts = ts_analyis(plot=False, sub_group=sg)
            train_r, test_r, train, test, metrics = fit_model_rolling_prediction(y_col=y_col, test_n = 50, timeseries_title=sg)
            metric_dict[sg] = metrics

        df_metrics = pd.DataFrame(metric_dict).round(2)
        print(df_metrics)
        df_metrics.to_csv(f'{DATASET_DIR}{y_col}/metrics_{y_col}.csv')




timeseries_title = 'all'
# model = 'asdf'
DATASET = 'furniture_sales'
WORK_DIR = f'/home/gerj/Documents/Init/SmartUMH/meetings/250122_kickoff/'
DATASET_DIR = os.path.join(WORK_DIR, DATASET)
SAVE_FIG = True


if __name__ == '__main__':
    # pass
    # main()
    run_one_timeseries()
    # ts_diff, ts_ts = ts_analyis(plot=False, sub_group='Chairs')
