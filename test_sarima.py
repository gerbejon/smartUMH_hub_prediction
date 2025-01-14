import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Example: Generate synthetic time series data
date_range = pd.date_range(start='2020-01-01', periods=36, freq='M')
data = pd.DataFrame({
    'Date': date_range,
    'Value': np.sin(2 * np.pi * date_range.month / 12) + np.random.normal(0, 0.2, len(date_range))
})
data.set_index('Date', inplace=True)

# Split into train and test sets
train = data.iloc[:-6]
test = data.iloc[-6:]

# Fit the SARIMAX model
model = SARIMAX(train['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit()

# Generate predictions
train['Predicted'] = result.predict(start=train.index[0], end=train.index[-1])
forecast = result.forecast(steps=len(test))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Value'], label='Train Data', color='blue')
plt.plot(train.index, train['Predicted'], label='SARIMAX Fit', color='green')
plt.plot(test.index, test['Value'], label='Test Data', color='orange')
plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='dashed')
plt.title('Time Series and SARIMAX Prediction')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()