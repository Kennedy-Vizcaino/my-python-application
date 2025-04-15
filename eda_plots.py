import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

df = pd.read_csv('vehicles_us.csv')  # or pass this in via function args for better modularity

def plot_price_distribution():
    plt.figure(figsize=(10, 4))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Price Distribution')
    st.pyplot(plt.gcf())

def plot_count_by_year():
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x='model_year', order=sorted(df['model_year'].dropna().astype(int).unique()))
    plt.title('Car Counts by Model Year')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

def plot_suv_scatter():
    suv_df = df[df['type'] == 'SUV']
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=suv_df, x='model_year', y='price', hue='paint_color', palette='RdBu')
    plt.title('SUV Prices by Year and Color')
    st.pyplot(plt.gcf())

def plot_sedan_scatter():
    sedan_df = df[df['type'] == 'sedan']
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sedan_df, x='model_year', y='price', hue='paint_color', palette='gray')
    plt.title('Sedan Prices by Year and Color')
    st.pyplot(plt.gcf())
