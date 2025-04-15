import pandas as pd
import streamlit as st
import plotly.express as px
from eda_plots import (
    plot_price_distribution,
    plot_count_by_year,
    plot_suv_scatter,
    plot_sedan_scatter)

df = pd.read_csv('vehicles_us.csv')

# Clean data
df = df.dropna(subset=['model_year'])
df['model_year'] = df['model_year'].astype(int)

# Sidebar filters
st.sidebar.header("Filter Options")
min_year, max_year = int(df['model_year'].min()), int(df['model_year'].max())
year_range = st.sidebar.slider("Model Year Range", min_year, max_year, (2016, max_year))
colors = st.sidebar.multiselect("Paint Colors", options=df['paint_color'].dropna().unique(), default=['black', 'white'])
conditions = st.sidebar.multiselect("Condition", options=df['condition'].dropna().unique(), default=['like new', 'good', 'excellent'])

# Apply filters
filtered_df = df[
    (df['model_year'] >= year_range[0]) &
    (df['model_year'] <= year_range[1]) &
    (df['paint_color'].isin(colors)) &
    (df['condition'].isin(conditions))
]

# Main title
st.title("🚗 Used Cars Dashboard")

# Interactive Plotly charts
fig_price = px.histogram(filtered_df, x='price', nbins=50, title='Price Distribution')
st.plotly_chart(fig_price)

fig_year_price = px.scatter(filtered_df, x='model_year', y='price', color='condition', title='Model Year vs Price by Condition')
st.plotly_chart(fig_year_price)

# Static Matplotlib/Seaborn charts
st.subheader("Histograms & Count Plots")
plot_price_distribution()
plot_count_by_year()

st.subheader("SUV Scatter Plot (Red & Blue)")
plot_suv_scatter()

st.subheader("Sedan Scatter Plot (Black & White)")
plot_sedan_scatter()
