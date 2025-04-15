import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import distinctipy

# App title
st.title("🚗 Car Sales Visualizations Dashboard 🏁 ")

# Load your dataset (update the path as needed)
@st.cache_data
def load_data():
    df = pd.read_csv("car_data.csv")  # Change this to match your dataset name

    # Create mileage category
    def categorize_mileage(odometer):
        if odometer < 40000:
            return 'low'
        elif odometer <= 100000:
            return 'medium'
        else:
            return 'high'
    df['mileage_category'] = df['odometer'].apply(categorize_mileage)
    return df

df = load_data()

# ========== 1. Histogram: Faceted by Model Year ==========
st.subheader("1. Price Distribution by Year (Black & White Cars, Low/Medium Mileage)")
filtered_df = df[
    (df['model_year'] >= 2016) &
    (df['paint_color'].isin(['black', 'white'])) &
    (df['mileage_category'].isin(['low', 'medium']))
].copy()

sns.set(style="whitegrid")
g = sns.FacetGrid(
    filtered_df,
    col="model_year", col_wrap=2, height=4,
    hue="paint_color", palette=["lightcoral", "lightblue"],
    sharex=False, sharey=True
)
g.map(sns.histplot, "price", stat="density", common_norm=False, alpha=0.6, kde=True)
g.add_legend(title="Paint Color")
g.set_titles("Year: {col_name}")
g.set_axis_labels("Price", "Density")
g.fig.suptitle("Price Distribution by Model Year", fontsize=16)
g.fig.subplots_adjust(top=0.9)

st.pyplot(g.fig)

# ========== 2. Histogram: Overlaid ==========
st.subheader("2. Price Distribution (Black & White Cars, Low/Medium Mileage)")
bw_low_med_df = df[
    (df['paint_color'].isin(['black', 'white'])) &
    (df['mileage_category'].isin(['low', 'medium'])) &
    (df['price'] > 0)
]

plt.figure(figsize=(12, 6))
colors = {'black': '#ff9999', 'white': '#99ccff'}
for color in ['black', 'white']:
    subset = bw_low_med_df[bw_low_med_df['paint_color'] == color]
    sns.histplot(
        data=subset,
        x='price',
        bins=30,
        color=colors[color],
        label=f'{color.capitalize()} Cars',
        kde=True,
        stat="count",
        alpha=0.7
    )

plt.title('Price Distribution: Black & White Cars with Low/Medium Mileage', fontsize=14)
plt.xlabel('Price ($)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
st.pyplot(plt.gcf())

# ========== 3. Bar Plot: Count by Model Year and Mileage ==========
st.subheader("3. Count of Cars by Model Year and Mileage Category")
filtered_df = df[
    (df['model_year'] >= 2016) &
    (df['paint_color'].isin(['black', 'white'])) &
    (df['condition'].isin(['like new', 'good', 'excellent']))
].copy()

plt.figure(figsize=(12, 6))
sns.countplot(
    data=filtered_df,
    x='model_year',
    hue='mileage_category',
    palette='pastel'
)
plt.title('Count of Cars by Model Year and Mileage Category (Black & White Cars Only)', fontsize=14)
plt.xlabel('Model Year')
plt.ylabel('Number of Cars')
plt.legend(title='Mileage Category')
plt.tight_layout()
st.pyplot(plt.gcf())

# ========== 4. Scatter Plot: SUV ==========
st.subheader("4. Price vs. Mileage – SUV (Red & Blue, 2016–2019)")
filtered_df = df[
    (df['model_year'] >= 2016) &
    (df['model_year'] <= 2019) &
    (df['paint_color'].isin(['red', 'blue'])) &
    (df['mileage_category'].isin(['low', 'medium', 'high'])) &
    (df['condition'].isin(['like new', 'good', 'excellent'])) &
    (df['type'] == 'SUV')
].copy()

unique_models = sorted(filtered_df['model'].unique())
palette = dict(zip(unique_models, [distinctipy.get_hex(c) for c in distinctipy.get_colors(len(unique_models))]))
fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4, wspace=0.25)

colors = ['red', 'blue']
years = [2016, 2017, 2018, 2019]

for row_idx, color in enumerate(colors):
    for col_idx, year in enumerate(years):
        ax = axes[row_idx, col_idx]
        subset = filtered_df[
            (filtered_df['paint_color'] == color) &
            (filtered_df['model_year'] == year)
        ]
        if not subset.empty:
            sns.scatterplot(
                data=subset,
                x='price',
                y='odometer',
                hue='model',
                palette=palette,
                alpha=0.75,
                s=80,
                ax=ax,
                legend=False
            )
            ax.text(0.95, 0.95, f'N = {len(subset)}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, color='gray')
        ax.set_title(f"{color.title()} SUV – {year}", fontsize=12)
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Mileage")

fig.suptitle("Price vs Mileage by Model (SUV, Red & Blue, 2016–2019)", fontsize=16)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
st.pyplot(fig)

# ========== 5. Scatter Plot: Sedan ==========
st.subheader("5. Price vs. Mileage – Sedan (Black & White, 2016–2019)")
filtered_df = df[
    (df['model_year'] >= 2016) &
    (df['model_year'] <= 2019) &
    (df['paint_color'].isin(['black', 'white'])) &
    (df['mileage_category'].isin(['low', 'medium', 'high'])) &
    (df['condition'].isin(['like new', 'good', 'excellent'])) &
    (df['type'] == 'sedan')
].copy()

unique_models = sorted(filtered_df['model'].unique())
palette = dict(zip(unique_models, [distinctipy.get_hex(c) for c in distinctipy.get_colors(len(unique_models))]))
fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.4, wspace=0.25)

colors = ['black', 'white']

for row_idx, color in enumerate(colors):
    for col_idx, year in enumerate(years):
        ax = axes[row_idx, col_idx]
        subset = filtered_df[
            (filtered_df['paint_color'] == color) &
            (filtered_df['model_year'] == year)
        ]
        if not subset.empty:
            sns.scatterplot(
                data=subset,
                x='price',
                y='odometer',
                hue='model',
                palette=palette,
                alpha=0.75,
                s=80,
                ax=ax,
                legend=False
            )
            ax.text(0.95, 0.95, f'N = {len(subset)}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, color='gray')
        ax.set_title(f"{color.title()} Sedan – {year}", fontsize=12)
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Mileage")

fig.suptitle("Price vs Mileage by Model (Sedan, Black & White, 2016–2019)", fontsize=16)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
st.pyplot(fig)
