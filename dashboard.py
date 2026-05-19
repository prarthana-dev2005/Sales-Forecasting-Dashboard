import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# LOAD DATASET
# -----------------------------

from sqlalchemy import create_engine

username = "postgres"
password = "YOUR_PASSWORD"
host = "localhost"
port = "5432"
database = "sales_forecasting"

engine = create_engine(
    f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
)

# Read data directly from PostgreSQL
df = pd.read_sql(
    "SELECT * FROM sales_data",
    engine
)

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.header("Filter Data")

start_date = st.sidebar.date_input(
    "Start Date",
    df['Date'].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    df['Date'].max()
)

# Filter dataframe
filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

st.sidebar.markdown("---")

st.sidebar.subheader("Project Information")

st.sidebar.write("Sales Forecasting System")
st.sidebar.write("Using Python, PostgreSQL & Streamlit")

# -----------------------------
# FORECASTING
# -----------------------------

filtered_df = filtered_df.copy()

# Create numerical day values
filtered_df['Day'] = np.arange(len(filtered_df))

# Linear forecasting
z = np.polyfit(
    filtered_df['Day'],
    filtered_df['Sales'],
    1
)

p = np.poly1d(z)

# Forecast values
filtered_df['Forecast'] = p(
    filtered_df['Day']
)

# -----------------------------
# DASHBOARD TITLE
# -----------------------------

st.title("Sales Forecasting Dashboard")

st.markdown("""
This dashboard analyzes historical sales data,
visualizes business trends,
and predicts future sales performance using
linear forecasting techniques.
""")

# -----------------------------
# SHOW DATASET
# -----------------------------

st.subheader("Sales Data")

st.dataframe(filtered_df.head())

# -----------------------------
# METRICS
# -----------------------------

st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Sales",
    int(filtered_df['Sales'].sum())
)

col2.metric(
    "Average Sales",
    round(filtered_df['Sales'].mean(), 2)
)

col3.metric(
    "Maximum Sales",
    int(filtered_df['Sales'].max())
)

col4.metric(
    "Minimum Sales",
    int(filtered_df['Sales'].min())
)

# -----------------------------
# GRAPH
# -----------------------------

st.subheader("Sales Forecast Graph")

fig, ax = plt.subplots(figsize=(14,6))

# Actual Sales
ax.plot(
    filtered_df['Date'],
    filtered_df['Sales'],
    label='Actual Sales',
    linewidth=2
)

# Forecast Line
ax.plot(
    filtered_df['Date'],
    filtered_df['Forecast'],
    linestyle='--',
    linewidth=2,
    label='Forecast'
)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Sales Trend & Forecast")

ax.legend()

ax.grid(True)

st.pyplot(fig)

# -----------------------------
# MONTHLY SALES ANALYSIS
# -----------------------------

st.subheader("Monthly Sales Analysis")

monthly_sales = filtered_df.resample(
    'M',
    on='Date'
)['Sales'].sum().reset_index()

fig_monthly = px.line(
    monthly_sales,
    x='Date',
    y='Sales',
    title='Monthly Sales Trend',
    markers=True
)

st.plotly_chart(
    fig_monthly,
    use_container_width=True
)

# -----------------------------
# FUTURE PREDICTIONS
# -----------------------------

st.subheader("Future Predictions")

future_days = np.arange(
    len(filtered_df),
    len(filtered_df) + 5
)

future_forecast = p(future_days)

future_df = pd.DataFrame({
    'Future Day': range(1, 6),
    'Predicted Sales': future_forecast.astype(int)
})

st.table(future_df)

# -----------------------------
# DOWNLOAD FILTERED DATA
# -----------------------------

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_sales_data.csv',
    mime='text/csv'
)

st.markdown("---")

st.markdown(
    "Developed using Python, Pandas, PostgreSQL, Matplotlib and Streamlit"
)