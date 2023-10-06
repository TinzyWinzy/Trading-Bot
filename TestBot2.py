import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load historical data from CSV (adjust the file path accordingly)
data = pd.read_csv('../BTC-2018min.csv')

# Convert the 'date' column to a datetime format
data['date'] = pd.to_datetime(data['date'])

# Sort data by date
data.sort_values(by='date', inplace=True)

# Select features and target variable
features = ['open', 'high', 'low', 'Volume_BTC', 'Volume_USD']
target = 'close'

# Normalize the features
for feature in features:
    data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Reshape data for LSTM input (samples, time steps, features)
X_train = X_train.values.reshape(-1, len(features), 1)
X_test = X_test.values.reshape(-1, len(features), 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(len(features), 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=1000, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Reshape y_pred and y_test to match the shape of the original input data
y_pred = y_pred.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Inverse transform the predictions and actual values
y_pred = (y_pred * (data[target].max() - data[target].min())) + data[target].min()
y_test = (y_test * (data[target].max() - data[target].min())) + data[target].min()

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(data['date'][-len(y_test):], y_test, label='Actual Prices', linewidth=2)
plt.plot(data['date'][-len(y_test):], y_pred, label='Predicted Prices', linestyle='--')
plt.xlabel('Date')
plt.ylabel('BTC/USD Price')
plt.legend()
plt.title('BTC/USD Price Prediction with LSTM')
plt.show()
