import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import distinctipy
import plotly.express as px


# Application title
st.title("🚗 Car Sales Visualizations Dashboard 🏁 ")

#the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("vehicles_us.csv")  

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

# ➕ Add Shared Legend
legend_fig, legend_ax = plt.subplots(figsize=(min(len(unique_models) * 0.7, 20), 2))
legend_ax.axis("off")
legend_handles = [Patch(color=palette[model], label=model) for model in unique_models]
legend_ax.legend(
    handles=legend_handles,
    title="Car Model",
    loc="center",
    ncol=6 if len(unique_models) <= 30 else 10,
    fontsize=9,
    title_fontsize=11,
    frameon=False
)
st.pyplot(legend_fig)

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
        ax.set_title(f"{color.title()} Sedan – {year}", fontsize=12)
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Mileage")

fig.suptitle("Price vs Mileage by Model (Sedan, Black & White, 2016–2019)", fontsize=16)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
st.pyplot(fig)

# ➕ Add Shared Legend
legend_fig, legend_ax = plt.subplots(figsize=(min(len(unique_models) * 0.7, 20), 2))
legend_ax.axis("off")
legend_handles = [Patch(color=palette[model], label=model) for model in unique_models]
legend_ax.legend(
    handles=legend_handles,
    title="Car Model",
    loc="center",
    ncol=6 if len(unique_models) <= 30 else 10,
    fontsize=9,
    title_fontsize=11,
    frameon=False
)
st.pyplot(legend_fig)

# ========== 6. Plotly Express Histogram with Checkbox ==========

st.subheader("6. Plotly Express Histogram: Price Distribution")

use_checkbox = st.checkbox("Show only cars from 2018 and later")

if use_checkbox:
    data_plotly = df[df['model_year'] >= 2018]
else:
    data_plotly = df.copy()

fig = px.histogram(
    data_plotly,
    x="price",
    nbins=30,
    title="Histogram of Car Prices",
    labels={"price": "Price ($)"},
    color_discrete_sequence=["indianred"]
)
fig.update_layout(bargap=0.1)

st.plotly_chart(fig, use_container_width=True)


# ========== 7. Plotly Express Scatter Plot with Filters ==========

st.subheader("7. Plotly Scatter Plot: Price vs. Odometer with Filters")

# Create multiselects for filter options
selected_type = st.multiselect("Select Car Type", options=sorted(df['type'].dropna().unique()), default=None)
selected_year = st.multiselect("Select Model Year", options=sorted(df['model_year'].dropna().unique()), default=None)
selected_mileage = st.multiselect("Select Mileage Category", options=sorted(df['mileage_category'].dropna().unique()), default=None)
selected_condition = st.multiselect("Select Condition", options=sorted(df['condition'].dropna().unique()), default=None)
selected_color = st.multiselect("Select Paint Color", options=sorted(df['paint_color'].dropna().unique()), default=None)

# Apply filters
filtered_plot_df = df.copy()
if selected_type:
    filtered_plot_df = filtered_plot_df[filtered_plot_df['type'].isin(selected_type)]
if selected_year:
    filtered_plot_df = filtered_plot_df[filtered_plot_df['model_year'].isin(selected_year)]
if selected_mileage:
    filtered_plot_df = filtered_plot_df[filtered_plot_df['mileage_category'].isin(selected_mileage)]
if selected_condition:
    filtered_plot_df = filtered_plot_df[filtered_plot_df['condition'].isin(selected_condition)]
if selected_color:
    filtered_plot_df = filtered_plot_df[filtered_plot_df['paint_color'].isin(selected_color)]

# Show scatter plot
if not filtered_plot_df.empty:
    scatter_fig = px.scatter(
        filtered_plot_df,
        x='odometer',
        y='price',
        color='paint_color',
        hover_data=['model', 'type', 'model_year', 'condition'],
        title="Price vs. Odometer (Filtered)",
        labels={'odometer': 'Mileage', 'price': 'Price ($)'}
    )
    scatter_fig.update_traces(marker=dict(size=7, opacity=0.6))
    st.plotly_chart(scatter_fig, use_container_width=True)
else:
    st.warning("No data matches the selected filters.")


