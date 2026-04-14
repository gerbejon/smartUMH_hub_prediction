from matplotlib import pyplot as plt
from pmdarima import auto_arima
from workbench.store_sales import ts_analyis
import statsmodels.api as sm
import pandas as pd
ts_diff, ts_ts = ts_analyis(plot=False, sub_group='Chairs')
y_col = 'Sales'

train_r = ts_diff.loc[ :(ts_diff.shape[0]-100)]
test_r = ts_diff.loc[ (ts_diff.shape[0]-100+1):]
test = ts_ts.loc[(ts_ts.shape[0]-100+1):]


model = auto_arima(train_r[y_col], seasonal=True, m=7, trace=True)
p, d, q = model.order
# p, d, q = 0,0,1
P, D, Q, m = model.seasonal_order
# P, D, Q, m = 0,0,0,0


sarima_model = sm.tsa.statespace.SARIMAX(
    train_r[y_col],
    order=(p, d, q),  # Replace with your optimal (p, d, q)
    seasonal_order=(P, D, Q, m),  # Replace with your optimal (P, D, Q, m)
    enforce_stationarity=False,
    enforce_invertibility=False
)
result = sarima_model.fit()
current_state = result
# all_predictions = []
for i, row  in test_r.iterrows():

    forecast = current_state.forecast(steps=1)
    test_r.loc[forecast.index[0], 'Sales_hat'] = forecast.values[0]
    current_state = current_state.append(
        [row[y_col]], refit=False
    )
    # _step = {
    #     'ind' : index,
    #     # 'y_hat' : forecast.values[0] + ts_ts.loc[index-1, y_col],
    #     # 'y_1' : ts_ts.loc[index-1, y_col],
    #     'y_hat_diff' : forecast.values[-1],
    #     'y_diff': ts_diff.loc[index, y_col],
    #     'y': ts_ts.loc[index, y_col]
    # }
    # all_predictions.append(_step)

# pred_df = pd.DataFrame(all_predictions).set_index('ind')
# (pred_df['y'] + pred_df['y_diff']).shift()
# pred_df['y_hat'] = (pred_df['y'] + pred_df['y_hat_diff']).shift()
# pred_df[['y_hat', 'y']].plot()
test_r[['Sales_hat', 'Sales']].plot()


df_pred = test[['Order Date', 'Sales']]
df_pred['Sales_hat_r'] = test_r['Sales_hat']
df_pred['Sales_r'] = test_r['Sales']
df_pred['Sales_hat'] = (test['Sales'] + test_r['Sales_hat']).shift(1)
df_pred['Sales_'] = (test['Sales'] + test_r['Sales']).shift(1)
# df_pred.loc[df_pred['Sales_hat'] < 0, 'Sales_hat'] = 0
print(df_pred)
df_pred.set_index('Order Date', inplace=True)
df_pred[['Sales', 'Sales_', 'Sales_hat']].plot()
# test_r[['Sales', 'Sales_hat']].plot()
plt.show()