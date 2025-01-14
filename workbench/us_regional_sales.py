import kagglehub
import pandas as pd
import os



# Download latest version US Regional Sales Data
path = kagglehub.dataset_download("talhabu/us-regional-sales-data")
print("Path to dataset files:", path)
df = pd.read_csv(os.path.join(path, 'US_Regional_Sales_Data.csv'))
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
data = df[['_ProductID', 'OrderDate', '_StoreID', 'Discount Applied', 'Unit Price', 'Sales Channel', 'Order Quantity']]
prod12 = data.query('_ProductID == 12')