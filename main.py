from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/sales_data.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Display data
print("Sales Data:\n")
print(df.head())

# -----------------------------
# DATA CLEANING
# -----------------------------

print("\nChecking Missing Values:\n")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing sales values with mean
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())

print("\nData After Cleaning:\n")
print(df)

# -----------------------------
# STATISTICS
# -----------------------------

print("\nStatistics:\n")
print(df.describe())

# -----------------------------
# POSTGRESQL INTEGRATION
# -----------------------------

username = "postgres"
password = "prarts321"
host = "localhost"
port = "5432"
database = "sales_forecasting"

engine = create_engine(
    f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
)

# Store dataframe
df.to_sql(
    'sales_data',
    engine,
    if_exists='replace',
    index=False
)

# Retrieve data
sql_data = pd.read_sql(
    'SELECT * FROM sales_data',
    engine
)

print("\nData Retrieved From PostgreSQL:\n")
print(sql_data.head())


# -----------------------------
# FORECASTING
# -----------------------------

# Create numerical day values
df['Day'] = np.arange(len(df))

# Create best-fit linear trend line
z = np.polyfit(df['Day'], df['Sales'], 1)

# Create forecasting equation
p = np.poly1d(z)

# Predict sales values
df['Forecast'] = p(df['Day'])

print("\nForecast Data:\n")
print(df[['Date', 'Sales', 'Forecast']])

# -----------------------------
# FUTURE PREDICTIONS
# -----------------------------

future_days = np.arange(len(df), len(df) + 5)

future_forecast = p(future_days)

print("\nFuture Sales Predictions:\n")

for i, value in enumerate(future_forecast, start=1):
    print(f"Future Day {i}: {value:.2f}")

# -----------------------------
# VISUALIZATION
# -----------------------------

plt.figure(figsize=(12,6))

# Actual sales line
plt.plot(df['Date'], df['Sales'],
         marker='o',
         label='Actual Sales')

# Forecast line
plt.plot(df['Date'], df['Forecast'],
         linestyle='--',
         marker='x',
         label='Forecast')

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")

plt.legend()

plt.grid(True)

plt.show()