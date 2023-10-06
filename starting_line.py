import pandas as pd

# Load the dataset from the CSV file
file_path = 'BTC-2017min.csv'
data = pd.read_csv(file_path)

# Data Cleaning
# Remove rows with missing values (NaN)
data.dropna(inplace=True)

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Describe the dataset
description = data.describe()

# Display the first few rows of the cleaned dataset
print("Cleaned Dataset:")
print(data.head())

# Display dataset description
print("\nDataset Description:")
print(description)

# Display unique symbols in the dataset
unique_symbols = data['symbol'].unique()
print("\nUnique Symbols:")
print(unique_symbols)
