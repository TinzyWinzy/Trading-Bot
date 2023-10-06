import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset from a CSV file (adjust the file path accordingly)
file_path = 'BTC-2018min.csv'
data = pd.read_csv(file_path)

# Data Cleaning
# data.dropna(inplace=True)

# Column Analysis
numeric_columns = ['open', 'high', 'low', 'close', 'Volume_BTC', 'Volume_USD']
non_numeric_columns = ['date', 'symbol']

# Check for outliers, distributions, and summary statistics for numeric columns
numeric_summary = data[numeric_columns].describe()
print("Numeric Columns Summary:")
print(numeric_summary)

# Assess unique values, data types, and patterns for non-numeric columns
non_numeric_summary = data[non_numeric_columns].describe(include='all')
print("\nNon-Numeric Columns Summary:")
print(non_numeric_summary)

# Time Series Analysis (if 'date' column is a datetime)
if 'date' in data:
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    plt.figure(figsize=(12, 6))
    plt.title("Cryptocurrency Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    for symbol in data['symbol'].unique():
        symbol_data = data[data['symbol'] == symbol]['close']
        plt.plot(symbol_data.index, symbol_data, label=symbol)
    plt.legend()
    plt.show()

    # Perform ADFuller test for stationarity
    for symbol in data['symbol'].unique():
        symbol_data = data[data['symbol'] == symbol]['close']
        result = adfuller(symbol_data)
        print(f"ADF Test for {symbol}:")
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        print("Stationary" if result[1] <= 0.05 else "Non-Stationary")

    # Plot ACF and PACF
    for symbol in data['symbol'].unique():
        symbol_data = data[data['symbol'] == symbol]['close']
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plot_acf(symbol_data, ax=plt.gca(), lags=40)
        plt.title(f"ACF for {symbol}")
        plt.subplot(212)
        plot_pacf(symbol_data, ax=plt.gca(), lags=40)
        plt.title(f"PACF for {symbol}")
        plt.show()

# Correlation Analysis
correlation_matrix = data[numeric_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Feature Engineering (Example: Calculate Moving Averages)
data['MA30'] = data.groupby('symbol')['close'].transform(lambda x: x.rolling(30).mean())
data['MA7'] = data.groupby('symbol')['close'].transform(lambda x: x.rolling(7).mean())

# Visualization
for symbol in data['symbol'].unique():
    symbol_data = data[data['symbol'] == symbol]
    plt.figure(figsize=(12, 6))
    plt.title(f"Price and Moving Averages for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.plot(symbol_data.index, symbol_data['close'], label='Close Price', color='blue')
    plt.plot(symbol_data.index, symbol_data['MA7'], label='7-Day MA', linestyle='--', color='green')
    plt.plot(symbol_data.index, symbol_data['MA30'], label='30-Day MA', linestyle='--', color='red')
    plt.legend()
    plt.show()

# Statistical Analysis (Example: T-Test for Mean Closing Price between Symbols)
symbols = data['symbol'].unique()
for i in range(len(symbols)):
    for j in range(i + 1, len(symbols)):
        symbol1_data = data[data['symbol'] == symbols[i]]['close']
        symbol2_data = data[data['symbol'] == symbols[j]]['close']
        t_stat, p_value = ttest_ind(symbol1_data, symbol2_data)
        print(f"T-Test for {symbols[i]} vs. {symbols[j]}:")
        print(f"T-Statistic: {t_stat}")
        print(f"P-Value: {p_value}")
        if p_value <= 0.05:
            print("Mean Closing Prices are significantly different.")
        else:
            print("No significant difference in mean closing prices.")

# Domain-Specific Analysis (e.g., sentiment analysis - not shown in this script)

# Machine Learning and Predictive Analysis (Example: Predict Close Price)
# Define target variable and features
target_column = 'close'
feature_columns = ['open', 'high', 'low', 'Volume BTC', 'Volume USD', 'MA7', 'MA30']

# Drop rows with missing values
data.dropna(subset=[target_column] + feature_columns, inplace=True)

# Split data into train and test sets
X = data[feature_columns]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Principal Component Analysis (PCA) as an example feature reduction technique
pca = PCA(n_components=5)  # Choose the number of components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train a machine learning model (Random Forest as an example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_pca)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) for Predictive Model: {mse}")
