import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def load_data(file_path):
    """Loads historical data from a CSV file."""
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    return data


def prepare_features(data):
    """Prepares the features for the linear regression model."""
    features = ['open', 'high', 'low', 'Volume_BTC', 'Volume_USD']
    for feature in features:
        data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())
    
    return data  



def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Trains a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the performance of a linear regression model on a test set."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def plot_results(data, y_pred):
    """Plots the actual vs. predicted prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['close'], label='Actual Prices')
    plt.plot(X_test.index, y_pred, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('BTC/USD Price')
    plt.legend()
    plt.title('BTC/USD Price Prediction')
    plt.show()


if __name__ == '__main__':

    file_path = 'BTC-Hourly.csv'
    data = load_data(file_path)
    data = prepare_features(data)

    target_col = 'close'
    features = ['open', 'high', 'low', 'Volume_BTC', 'Volume_USD']

    X = data[features]
    y = data[target_col]

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    mse = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')

    plot_results(data, model.predict(X_test))
