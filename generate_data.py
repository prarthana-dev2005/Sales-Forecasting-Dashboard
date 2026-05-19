import pandas as pd
import numpy as np

# Reproducible randomness
np.random.seed(42)

# Generate 365 dates
dates = pd.date_range(
    start='2025-01-01',
    periods=365
)

# Base sales trend
trend = np.linspace(100, 500, 365)

# Seasonal effect
seasonality = 50 * np.sin(
    np.linspace(0, 12*np.pi, 365)
)

# Random noise
noise = np.random.normal(0, 20, 365)

# Final sales values
sales = trend + seasonality + noise

# Create dataframe
df = pd.DataFrame({
    'Date': dates,
    'Sales': sales.astype(int)
})

# Save CSV
df.to_csv(
    'data/sales_data.csv',
    index=False
)

print("Dataset generated successfully!")