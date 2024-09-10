import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from datetime import date, timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler
import talib
from itertools import combinations

# Connect to database
host = "ec2-52-6-117-96.compute-1.amazonaws.com"
dbname = "dftej5l5m1cl78"
user = "aiuhlrpcnftsjs"
password = "8b2220cd5b6da572369545d91f6b435dfc37a42bfec6b6e2a5c9f236dfb65f42"

conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
cur = conn.cursor()

query = "SELECT date, symbol, CASE WHEN (close-open)/open*100 > 0.25 then 2 when (close-open)/open*100 < -0.25 then 0 else 1 END as category, close, open, ema_200, ema_50, ema_12, ema_26, high, low, volume, upper_band, lower_band, macd_results*100, rsi, stochastic_oscillator, ((close - open) / open) * 100 AS daily_percent_change, ((open - LAG(close) OVER (ORDER BY symbol, date)) / LAG(close) OVER (ORDER BY symbol, date)) * 100 AS overnight_percent_change, ((high - low) / low) * 100 AS percentage_volatility, ABS((close - open) / open) * 100 AS absolute_percent_change, EXTRACT(DOW FROM date) AS day_of_week, case when grp = 1 and lag(grp) over(partition by symbol order by date) = 0 then 1 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 0 then 2 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 0 then 3 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 0 then 4 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 0 then 5 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 0 then 6 when  grp = 1 and lag(grp) over(partition by symbol order by date) = 1 and lag(grp,2) over(partition by symbol order by date) = 1 and lag(grp,3) over(partition by symbol order by date) = 1 and lag(grp,4) over(partition by symbol order by date) = 1 and lag(grp,5) over(partition by symbol order by date) = 1 and lag(grp,6) over(partition by symbol order by date) = 1 and lag(grp,7) over(partition by symbol order by date) = 0 then 7 else 0 end as consecutive_green_days, case when grp = 0 and lag(grp) over(partition by symbol order by date) = 1 then 1 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 1 then 2 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 1 then 3 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 1 then 4 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 1 then 5 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 1 then 6 when  grp = 0 and lag(grp) over(partition by symbol order by date) = 0 and lag(grp,2) over(partition by symbol order by date) = 0 and lag(grp,3) over(partition by symbol order by date) = 0 and lag(grp,4) over(partition by symbol order by date) = 0 and lag(grp,5) over(partition by symbol order by date) = 0 and lag(grp,6) over(partition by symbol order by date) = 0 and lag(grp,7) over(partition by symbol order by date) = 1 then 7 else 0 end as consecutive_red_days,     volume / AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW)*100 AS relative_volume FROM (SELECT *, CASE WHEN close > open THEN 1 when close <= open then 0 END AS grp FROM daily_stock_price_fin) as test ORDER BY symbol, date;"

cur.execute(query)
data = cur.fetchall()

data = pd.DataFrame(data, columns=[ "date", "symbol", "category", "close", "open", "ema_200", "ema_50", "ema_12", "ema_26", "high", "low", "volume", "upper_band", "lower_band", "macd_results", "rsi", "stochastic_oscillator", "daily_percent_change", "overnight_percent_change", "percentage_volatility", "absolute_percent_change", "day_of_week", "consecutive_green_days",  "consecutive_red_days", "relative_volume"])

data = data.sort_values(by=['symbol', 'date'], ascending=[True, True])
data = data.iloc[:-1]

scaled_data_2 = data.iloc[100:]
print(scaled_data_2.isna().sum())

pattern_functions = [func for func in dir(talib) if func.startswith('CDL')]

for pattern_func in pattern_functions:
    scaled_data_2[pattern_func] = getattr(talib, pattern_func)(scaled_data_2['open'].values, scaled_data_2['high'].values, scaled_data_2['low'].values, scaled_data_2['close'].values)
scaled_data_2 = scaled_data_2[scaled_data_2[pattern_functions].any(axis=1)]

for column in pattern_functions:
    scaled_data_2[column] = np.where(scaled_data_2[column] != 0, pattern_functions.index(column) + 1, 0)
scaled_data_2['pattern_comb'] = scaled_data_2[pattern_functions].apply(
    lambda row: int(''.join(str(int(val)) for val in row if val != 0)), axis=1
)

scaled_data_2.drop(pattern_functions, axis=1, inplace=True)
scaled_data_3 = pd.DataFrame(scaled_data_2, columns=[ "date", "symbol", "category", "pattern_comb", "macd_results", "rsi", "stochastic_oscillator", "daily_percent_change", "overnight_percent_change", "percentage_volatility", "absolute_percent_change", "consecutive_green_days",  "consecutive_red_days", "relative_volume"])
numeric_columns = scaled_data_3.select_dtypes(include=['float64', 'int64']).columns

columns_to_round = [col for col in numeric_columns if col not in ['date', 'symbol']]
scaled_data_3[columns_to_round] = scaled_data_3[columns_to_round].replace([np.inf, -np.inf, np.nan], 0)

scaled_data_3[columns_to_round] = scaled_data_3[columns_to_round].round()
scaled_data_3[columns_to_round] = scaled_data_3[columns_to_round].astype(int)

sc = MinMaxScaler()

start_date = date(2000, 1, 1)
end_date = date(2023, 1, 1)

if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    start_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)
if start_date not in scaled_data_2['date'].values:
    end_date += timedelta(days=1)

symbols = data['symbol'].unique()
X_train = []
y_train = []
test_value_list = []
test_value_list_2 = []

for symbol in symbols:
    symbol_data = scaled_data_3[scaled_data_3['symbol'] == symbol]
    symbol_data_2 = scaled_data_2[scaled_data_2['symbol'] == symbol]

    date_range_data = symbol_data[(symbol_data['date'] >= start_date) & (symbol_data['date'] <= end_date)]
    date_range_data_2 = symbol_data_2[(symbol_data_2['date'] >= start_date) & (symbol_data_2['date'] <= end_date)]

    target_variable = 'category'

    data_points_in_group = 3

    for i in range(len(date_range_data) - data_points_in_group - 1):
        current_group_features = date_range_data.iloc[i:i + data_points_in_group, 2:].values

        target_value = date_range_data.iloc[i + data_points_in_group + 1, 2]
        test_value = date_range_data_2.iloc[i + data_points_in_group + 1, 17]
        test_value_2 = date_range_data.iloc[i + data_points_in_group + 1, 8]
        X_train.append(current_group_features)

        y_train.append(target_value)
        test_value_list.append(test_value)
        test_value_list_2.append(test_value_2)

X_train_ema_2 = np.array(X_train)
print(X_train_ema_2.shape)

X_train_ema_2 = X_train_ema_2.reshape((X_train_ema_2.shape[0], -1))
print(X_train_ema_2.shape)

test_value_list_2 = np.array(test_value_list_2)
print(test_value_list_2.shape)
Y_train_ema_2 = np.array(y_train)

X_train_ema_2 = np.concatenate((X_train_ema_2, test_value_list_2[:, np.newaxis]), axis=1)
print(X_train_ema_2.shape)

X_train_ema_2, Y_train_ema_2 = np.array(X_train_ema_2), np.array(Y_train_ema_2)
print("X_train_ema_2 shape:", X_train_ema_2.shape)
print("Y_train_ema_2 shape:", Y_train_ema_2.shape)


X_train, X_val, y_train, y_val = train_test_split(X_train_ema_2, Y_train_ema_2, test_size=0.3, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

clf = xgb.XGBClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

with open('xgboost_model_pos_fin.pkl', 'wb') as pkl_file:
    pickle.dump(clf, pkl_file)


raw_predictions = clf.predict_proba(X_val)  
print(raw_predictions)

predicted_classes = np.argmax(raw_predictions, axis=1)
print(predicted_classes)

custom_threshold = 0.6

mask_greater_than_threshold = np.max(raw_predictions, axis=1) > custom_threshold

final_predictions = np.ones_like(predicted_classes)

final_predictions[mask_greater_than_threshold] = predicted_classes[mask_greater_than_threshold]

print(final_predictions)

category_0_count = np.sum(final_predictions == 0)
category_1_count = np.sum(final_predictions == 1)
category_2_count = np.sum(final_predictions == 2)

print(f'Predicted Category 0 count: {category_0_count}')
print(f'Predicted Category 1 count: {category_1_count}')
print(f'Predicted Category 2 count: {category_2_count}')

total_category_0 = np.sum(final_predictions == 0)
correct_category_0 = np.sum((y_val == 0) & (final_predictions == 0))
in_category_0 = np.sum((y_val == 2) & (final_predictions == 0))

total_category_1 = np.sum(final_predictions == 2)
correct_category_1 = np.sum((y_val == 2) & (final_predictions == 2))
in_category_1 = np.sum((y_val == 0) & (final_predictions == 2))

category_0_accuracy = correct_category_0 / total_category_0
category_1_accuracy = correct_category_1 / total_category_1

total_percent = (correct_category_1+correct_category_0)/(correct_category_0 +in_category_0+correct_category_1+in_category_1)

print(f'total_percent: {total_percent}')
print(f'correct_category_0: {correct_category_0}')
print(f'in_category_0: {in_category_0}')
print(f'correct_category_1: {correct_category_1}')
print(f'in_category_1: {in_category_1}')
