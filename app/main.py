import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
from utils import (
    load_data, 
    filter_data_by_countries, 
    filter_data_by_date_range,
    filter_data_by_metric_range,
    get_top_regions,
    calculate_solar_potential_score,
    format_metric_name,
    get_correlation_insights
)

# Set page configuration
st.set_page_config(
    page_title="Solar Farm Potential Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - optimized for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFA726;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #E0E0E0;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #E0E0E0;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFA726;
    }
    .metric-label {
        font-size: 1rem;
        color: #BDBDBD;
    }
    .highlight {
        color: #FFA726;
        font-weight: 600;
    }
    .stPlotlyChart {
        background-color: rgba(30, 30, 30, 0.5);
        border-radius: 5px;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(0, 0, 0, 0.2);
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stDataFrame {
        background-color: rgba(30, 30, 30, 0.5);
        border-radius: 5px;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(0, 0, 0, 0.2);
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .info-box {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
        margin-bottom: 1rem;
    }
    /* Improve table readability on dark background */
    .dataframe {
        color: #E0E0E0 !important;
    }
    .dataframe th {
        background-color: rgba(50, 50, 50, 0.8) !important;
        color: #FFA726 !important;
    }
    .dataframe td {
        background-color: rgba(40, 40, 40, 0.6) !important;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: #BDBDBD;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 167, 38, 0.2) !important;
        color: #FFA726 !important;
        border-bottom: 2px solid #FFA726;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="main-header">Solar Farm Potential Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
This dashboard visualizes solar potential metrics across Benin, Sierra Leone, and Togo to identify high-potential regions for solar installations. 
Use the filters in the sidebar to customize your analysis.
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Check if data is loaded successfully
if df.empty:
    st.error("Failed to load data. Please check that the CSV files exist in the data/ directory.")
    st.stop()

# Extract date range from data
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

# Sidebar for filters
st.sidebar.markdown("## Data Filters")

# Country selector
countries = sorted(df['country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=countries
)

# Define all available metrics based on the data structure
all_metrics = [
    "GHI", "DNI", "DHI", "ModA", "ModB", "Tamb", "RH", "WS", 
    "WSgust", "WSstdev", "WD", "WDstdev", "BP", "Precipitation", 
    "TModA", "TModB"
]

# Filter to only include metrics that exist in the dataframe
available_metrics = [metric for metric in all_metrics if metric in df.columns]

# Metric selector
primary_metric = st.sidebar.selectbox(
    "Select Primary Metric",
    options=available_metrics,
    index=0 if "GHI" in available_metrics else 0
)

# Date range selector
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Handle single date selection
if len(date_range) == 1:
    date_range = (date_range[0], max_date)
elif len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Filter data based on selections
filtered_df = df.copy()

if selected_countries:
    filtered_df = filter_data_by_countries(filtered_df, selected_countries)

filtered_df = filter_data_by_date_range(filtered_df, start_date, end_date)

# Check if we have data after filtering
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# Calculate key metrics for the filtered data
avg_primary = filtered_df[primary_metric].mean()
max_primary = filtered_df[primary_metric].max()
min_primary = filtered_df[primary_metric].min()

# Display key metrics
st.markdown('<div class="sub-header">Key Metrics Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average {format_metric_name(primary_metric)}</div>
        <div class="metric-value">{avg_primary:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Maximum {format_metric_name(primary_metric)}</div>
        <div class="metric-value">{max_primary:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Minimum {format_metric_name(primary_metric)}</div>
        <div class="metric-value">{min_primary:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Calculate solar potential score
    potential_score = calculate_solar_potential_score(filtered_df)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Solar Potential Score</div>
        <div class="metric-value">{potential_score:.1f}/10</div>
    </div>
    """, unsafe_allow_html=True)

# Main visualizations
st.markdown('<div class="sub-header">Data Visualizations</div>', unsafe_allow_html=True)

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Comparison Analysis", "Time Series Analysis", "Regional Rankings"])

with tab1:
    st.markdown('<div class="section-header">Country Comparison</div>', unsafe_allow_html=True)
    
    # Boxplot comparing the selected metric across countries
    fig_boxplot = px.box(
        filtered_df, 
        x='country', 
        y=primary_metric,
        color='country',
        title=f"{format_metric_name(primary_metric)} Distribution by Country",
        labels={
            'country': 'Country',
            primary_metric: format_metric_name(primary_metric)
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark"
    )
    
    fig_boxplot.update_layout(
        xaxis_title="Country",
        yaxis_title=format_metric_name(primary_metric),
        legend_title="Country",
        font=dict(size=12),
        height=500,
        boxmode='group'
    )
    
    st.plotly_chart(fig_boxplot, use_container_width=True)
    
    # Bar chart showing average metric by country and region
    avg_by_country_region = filtered_df.groupby(['country', 'region'])[primary_metric].mean().reset_index()
    
    fig_bar = px.bar(
        avg_by_country_region,
        x='region',
        y=primary_metric,
        color='country',
        barmode='group',
        title=f"Average {format_metric_name(primary_metric)} by Region",
        labels={
            'region': 'Region',
            primary_metric: f'Average {format_metric_name(primary_metric)}',
            'country': 'Country'
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark"
    )
    
    fig_bar.update_layout(
        xaxis_title="Region",
        yaxis_title=f"Average {format_metric_name(primary_metric)}",
        legend_title="Country",
        font=dict(size=12),
        height=500
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)
    
    # Time aggregation selector
    time_agg = st.radio(
        "Time Aggregation",
        options=["Daily", "Weekly", "Monthly"],
        horizontal=True
    )
    
    # Prepare time series data based on selected aggregation
    if time_agg == "Daily":
        time_df = filtered_df.copy()
        time_df['period'] = time_df['timestamp'].dt.date
    elif time_agg == "Weekly":
        time_df = filtered_df.copy()
        time_df['period'] = time_df['timestamp'].dt.to_period('W').dt.start_time.dt.date
    else:  # Monthly
        time_df = filtered_df.copy()
        time_df['period'] = time_df['timestamp'].dt.to_period('M').dt.start_time.dt.date
    
    # Group by period and country
    time_series = time_df.groupby(['period', 'country'])[primary_metric].mean().reset_index()
    
    # Create time series plot
    fig_time = px.line(
        time_series,
        x='period',
        y=primary_metric,
        color='country',
        title=f"{format_metric_name(primary_metric)} Trends Over Time",
        labels={
            'period': 'Date',
            primary_metric: format_metric_name(primary_metric),
            'country': 'Country'
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark"
    )
    
    fig_time.update_layout(
        xaxis_title="Date",
        yaxis_title=format_metric_name(primary_metric),
        legend_title="Country",
        font=dict(size=12),
        height=500
    )
    
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Monthly heatmap
    st.markdown('<div class="section-header">Monthly Patterns</div>', unsafe_allow_html=True)
    
    # Prepare data for heatmap
    filtered_df['month'] = filtered_df['timestamp'].dt.month
    filtered_df['month_name'] = filtered_df['timestamp'].dt.month_name()
    
    # Group by month and country
    monthly_data = filtered_df.groupby(['month', 'month_name', 'country'])[primary_metric].mean().reset_index()
    
    # Create a pivot table for the heatmap
    heatmap_data = monthly_data.pivot(index='country', columns='month', values=primary_metric)
    
    # Replace month numbers with month names
    month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}
    heatmap_data.columns = [month_names[col] for col in heatmap_data.columns]
    
    # Create heatmap
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Country", color=format_metric_name(primary_metric)),
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index),
        color_continuous_scale="YlOrRd",
        title=f"Monthly {format_metric_name(primary_metric)} Patterns by Country",
        template="plotly_dark"
    )
    
    fig_heatmap.update_layout(
        xaxis_title="Month",
        yaxis_title="Country",
        font=dict(size=12),
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Top Regions by Solar Potential</div>', unsafe_allow_html=True)
    
    # Get top regions based on the primary metric
    top_regions = get_top_regions(filtered_df, primary_metric, n=10)
    
    # Create a bar chart for top regions
    fig_top = px.bar(
        top_regions,
        x='region',
        y='mean_value',
        color='country',
        title=f"Top Regions by {format_metric_name(primary_metric)}",
        labels={
            'region': 'Region',
            'mean_value': f'Average {format_metric_name(primary_metric)}',
            'country': 'Country'
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark"
    )
    
    fig_top.update_layout(
        xaxis_title="Region",
        yaxis_title=f"Average {format_metric_name(primary_metric)}",
        legend_title="Country",
        font=dict(size=12),
        height=500
    )
    
    st.plotly_chart(fig_top, use_container_width=True)
    
    # Display top regions table
    st.markdown('<div class="section-header">Detailed Metrics for Top Regions</div>', unsafe_allow_html=True)
    
    # Enhance the table with more metrics
    detailed_metrics = []
    
    for _, row in top_regions.iterrows():
        region_data = filtered_df[(filtered_df['country'] == row['country']) & 
                                 (filtered_df['region'] == row['region'])]
        
        region_metrics = {
            'Country': row['country'],
            'Region': row['region'],
            f'Mean {primary_metric}': round(row['mean_value'], 2),
            f'Median {primary_metric}': round(region_data[primary_metric].median(), 2),
            f'Std Dev {primary_metric}': round(region_data[primary_metric].std(), 2),
            'Solar Potential Score': round(calculate_solar_potential_score(region_data), 1)
        }
        
        # Add additional metrics if available
        for metric in ['Tamb', 'RH', 'WS']:
            if metric in region_data.columns:
                region_metrics[f'Mean {metric}'] = round(region_data[metric].mean(), 2)
        
        detailed_metrics.append(region_metrics)
    
    detailed_df = pd.DataFrame(detailed_metrics)
    
    # Display the table
    st.dataframe(detailed_df, use_container_width=True)

# Display Key Insights
st.markdown('<div class="sub-header">Key Insights</div>', unsafe_allow_html=True)

# Data Summary
summary = f"""
<div class="info-box">
    <p><strong>Data Summary:</strong></p>
    <p>The dataset contains measurements from {len(countries)} countries and spans from {min_date} to {max_date}.</p>
    <p>The highest {primary_metric} value ({max_primary:.2f}) was recorded in {filtered_df.loc[filtered_df[primary_metric].idxmax(), 'country']}.</p>
</div>
"""

# Correlation Insights
correlation_insights = get_correlation_insights(filtered_df, primary_metric)
correlation_summary = f"""
<div class="info-box">
    <p><strong>Key Correlations:</strong></p>
    <ul>
        {' '.join([f'<li>{insight}</li>' for insight in correlation_insights])}
    </ul>
</div>
"""

# Combine and display insights
st.markdown(summary + correlation_summary, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2023 Solar Farm Potential Analysis Dashboard")
st.caption("Dashboard created for analyzing solar potential in West Africa.")