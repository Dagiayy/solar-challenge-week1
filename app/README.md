import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
from scipy import stats
from utils import (
    load_data, 
    filter_data_by_countries, 
    filter_data_by_date_range,
    filter_data_by_metric_range,
    get_top_regions,
    generate_mock_data,
    calculate_solar_potential_score,
    format_metric_name,
    perform_statistical_test,
    get_correlation_insights,
    get_seasonal_insights,
    get_actionable_recommendations
)

# Set page configuration
st.set_page_config(
    page_title="Solar Farm Potential Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - adjusted for dark background
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
    .warning-box {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 0 5px 5px 0;
        margin-bottom: 1rem;
    }
    .data-source-info {
        background-color: rgba(156, 39, 176, 0.1);
        border-left: 5px solid #9C27B0;
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

# Sidebar for filters
st.sidebar.markdown("## Data Filters")

# Load data (actual or mock)
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Check if we're using actual or mock data
is_mock_data = df.equals(generate_mock_data())
if is_mock_data:
    st.markdown("""
    <div class="data-source-info">
    <strong>Note:</strong> Using simulated data for demonstration. Actual data files (benin_clean.csv, sierraleone_clean.csv, togo_clean.csv) were not found.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="success-box">
    <strong>Data Source:</strong> Using actual data from Benin, Sierra Leone, and Togo.
    </div>
    """, unsafe_allow_html=True)

# Extract date range from data
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

# Country selector
countries = sorted(df['country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=countries
)

# Metric selector
metrics = ["GHI", "DNI", "DHI", "Tamb", "RH", "WS", "BP", "Precipitation"]
primary_metric = st.sidebar.selectbox(
    "Select Primary Metric",
    options=metrics,
    index=0
)

# Secondary metric for bubble chart
secondary_metric = st.sidebar.selectbox(
    "Select Secondary Metric (for Bubble Chart)",
    options=[m for m in metrics if m != primary_metric],
    index=1 if primary_metric != "DNI" else 0
)

# Tertiary metric for bubble size
tertiary_metric = st.sidebar.selectbox(
    "Select Tertiary Metric (for Bubble Size)",
    options=[m for m in metrics if m not in [primary_metric, secondary_metric]],
    index=0
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

# Advanced filters
st.sidebar.markdown("## Advanced Filters")
show_advanced = st.sidebar.checkbox("Show Advanced Filters", value=False)

if show_advanced:
    # Get min/max values for each metric
    metric_ranges = {}
    for metric in metrics:
        min_val = float(df[metric].min())
        max_val = float(df[metric].max())
        metric_ranges[metric] = st.sidebar.slider(
            f"{format_metric_name(metric)} Range",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            format=None
        )

# Filter data based on selections
filtered_df = df.copy()

if selected_countries:
    filtered_df = filter_data_by_countries(filtered_df, selected_countries)

filtered_df = filter_data_by_date_range(filtered_df, start_date, end_date)

if show_advanced:
    for metric, (min_val, max_val) in metric_ranges.items():
        filtered_df = filter_data_by_metric_range(filtered_df, metric, min_val, max_val)

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

# Summary Statistics
st.markdown('<div class="sub-header">Summary Statistics</div>', unsafe_allow_html=True)

# Create summary statistics for the primary metric
summary_stats = filtered_df[primary_metric].describe().reset_index()
summary_stats.columns = ['Statistic', 'Value']

# Format the statistics
summary_stats['Value'] = summary_stats['Value'].round(2)

# Display the statistics in a more visually appealing way
col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(summary_stats, use_container_width=True)

with col2:
    # Create a histogram of the primary metric
    fig_hist = px.histogram(
        filtered_df, 
        x=primary_metric,
        color='country',
        nbins=30,
        opacity=0.7,
        title=f"Distribution of {format_metric_name(primary_metric)}",
        template="plotly_dark"
    )
    
    fig_hist.update_layout(
        xaxis_title=format_metric_name(primary_metric),
        yaxis_title="Frequency",
        legend_title="Country",
        font=dict(size=12),
        height=300
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

# Statistical Testing
st.markdown('<div class="sub-header">Statistical Analysis</div>', unsafe_allow_html=True)

# Perform statistical test
test_result = perform_statistical_test(filtered_df, primary_metric, selected_countries)

if test_result:
    test_name, statistic, p_value, interpretation = test_result
    
    st.markdown(f"""
    <div class="info-box">
        <p><strong>{test_name} Results:</strong></p>
        <p>Test statistic: {statistic:.2f}</p>
        <p>p-value: {p_value:.4f}</p>
        <p>{interpretation}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Statistical testing requires at least two countries with data to compare.")

# Main visualizations
st.markdown('<div class="sub-header">Data Visualizations</div>', unsafe_allow_html=True)

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Comparison Analysis", "Time Series Analysis", "Correlation Analysis", "Regional Rankings"])

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
    filtered_df['year'] = filtered_df['timestamp'].dt.year
    
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
    
    # Get seasonal insights
    seasonal_insights = get_seasonal_insights(filtered_df, primary_metric)
    
    st.markdown('<div class="section-header">Seasonal Patterns</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <p><strong>Seasonal Analysis:</strong></p>
        <p>Best month for {primary_metric}: <span class="highlight">{seasonal_insights['best_month']}</span> with average value of {seasonal_insights['best_month_value']:.2f}</p>
        <p>Worst month for {primary_metric}: <span class="highlight">{seasonal_insights['worst_month']}</span> with average value of {seasonal_insights['worst_month_value']:.2f}</p>
        <p><strong>Seasonal Averages:</strong></p>
        <ul>
            {' '.join([f'<li>{season}: {value:.2f}</li>' for season, value in seasonal_insights['seasonal_averages'].items()])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
    
    # Bubble chart
    fig_bubble = px.scatter(
        filtered_df,
        x=primary_metric,
        y=secondary_metric,
        size=tertiary_metric,
        color='country',
        hover_name='region',
        size_max=30,
        opacity=0.7,
        title=f"Relationship between {format_metric_name(primary_metric)}, {format_metric_name(secondary_metric)}, and {format_metric_name(tertiary_metric)}",
        labels={
            primary_metric: format_metric_name(primary_metric),
            secondary_metric: format_metric_name(secondary_metric),
            tertiary_metric: format_metric_name(tertiary_metric),
            'country': 'Country'
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark"
    )
    
    fig_bubble.update_layout(
        xaxis_title=format_metric_name(primary_metric),
        yaxis_title=format_metric_name(secondary_metric),
        legend_title="Country",
        font=dict(size=12),
        height=600
    )
    
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Correlation heatmap
    st.markdown('<div class="section-header">Metric Correlations</div>', unsafe_allow_html=True)
    
    # Calculate correlation matrix
    corr_metrics = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS', 'BP', 'Precipitation']
    corr_matrix = filtered_df[corr_metrics].corr()
    
    # Create correlation heatmap
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        x=corr_metrics,
        y=corr_metrics,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Between Metrics",
        template="plotly_dark"
    )
    
    fig_corr.update_layout(
        xaxis_title="Metric",
        yaxis_title="Metric",
        font=dict(size=12),
        height=500
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Get correlation insights
    correlation_insights = get_correlation_insights(filtered_df)
    
    st.markdown('<div class="section-header">Correlation Insights</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
        <p><strong>Key Correlations:</strong></p>
        <ul>
            {' '.join([f'<li>{insight}</li>' for insight in correlation_insights])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-header">Top Regions by Solar Potential</div>', unsafe_allow_html=True)
    
    # Get top regions based on the primary metric
    top_regions = get_top_regions(filtered_df, primary_metric, n=10)
    
    # Create a bar chart for top regions
    fig_top = px.bar(
        top_regions,
        x='region',
        y='mean_value',
        color='country',
        title=f"Top 10 Regions by {format_metric_name(primary_metric)}",
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
            f'Mean {format_metric_name(primary_metric)}': round(row['mean_value'], 2),
            f'Median {format_metric_name(primary_metric)}': round(region_data[primary_metric].median(), 2),
            f'Std Dev {format_metric_name(primary_metric)}': round(region_data[primary_metric].std(), 2),
            'Solar Potential Score': round(calculate_solar_potential_score(region_data), 1),
            f'Mean {format_metric_name("Tamb")}': round(region_data['Tamb'].mean(), 2),
            f'Mean {format_metric_name("RH")}': round(region_data['RH'].mean(), 2),
            f'Mean {format_metric_name("WS")}': round(region_data['WS'].mean(), 2)
        }
        
        detailed_metrics.append(region_metrics)
    
    detailed_df = pd.DataFrame(detailed_metrics)
    
    # Display the table
    st.dataframe(detailed_df, use_container_width=True)

# Insights and recommendations section
st.markdown('<div class="sub-header">Insights & Recommendations</div>', unsafe_allow_html=True)

# Generate insights based on the filtered data
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    
    # Get the country with highest average primary metric
    best_country = filtered_df.groupby('country')[primary_metric].mean().idxmax()
    best_country_value = filtered_df.groupby('country')[primary_metric].mean().max()
    
    # Get the best region overall
    best_region_row = top_regions.iloc[0]
    best_region = best_region_row['region']
    best_region_country = best_region_row['country']
    best_region_value = best_region_row['mean_value']
    
    # Get the month with highest average primary metric
    best_month_idx = monthly_data.groupby('month')[primary_metric].mean().idxmax()
    best_month = calendar.month_name[best_month_idx]
    best_month_value = monthly_data.groupby('month')[primary_metric].mean().max()
    
    st.markdown(f"""
    <div class="success-box">
        <p><span class="highlight">{best_country}</span> shows the highest average {format_metric_name(primary_metric)} at {best_country_value:.2f}.</p>
        <p>The region with the highest solar potential is <span class="highlight">{best_region}</span> in {best_region_country} with an average {format_metric_name(primary_metric)} of {best_region_value:.2f}.</p>
        <p>The best month for solar generation is <span class="highlight">{best_month}</span> with an average {format_metric_name(primary_metric)} of {best_month_value:.2f}.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Correlation insight
    ghi_tamb_corr = filtered_df[['GHI', 'Tamb']].corr().iloc[0, 1]
    ghi_rh_corr = filtered_df[['GHI', 'RH']].corr().iloc[0, 1]
    
    st.markdown(f"""
    <div class="info-box">
        <p>The correlation between GHI and temperature is <span class="highlight">{ghi_tamb_corr:.2f}</span>, indicating a {'strong positive' if ghi_tamb_corr > 0.7 else 'moderate positive' if ghi_tamb_corr > 0.3 else 'weak'} relationship.</p>
        <p>The correlation between GHI and relative humidity is <span class="highlight">{ghi_rh_corr:.2f}</span>, showing a {'strong negative' if ghi_rh_corr < -0.7 else 'moderate negative' if ghi_rh_corr < -0.3 else 'weak'} relationship.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
    
    # Get actionable recommendations
    recommendations = get_actionable_recommendations(filtered_df, primary_metric)
    
    st.markdown(f"""
    <div class="warning-box">
        <p><strong>Actionable Recommendations:</strong></p>
        <ul>
            {' '.join([f'<li>{rec}</li>' for rec in recommendations])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2023 MoonLight Energy Solutions | Solar Farm Potential Analysis Dashboard")
st.caption("Dashboard created for analyzing solar potential in West Africa. For questions or support, contact support@moonlightenergy.com")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_actual_data():
    """
    Load actual cleaned data from CSV files.
    
    Returns:
    --------
    pandas.DataFrame
        Combined dataframe containing data from all countries.
    """
    try:
        # Define file paths
        benin_path = 'data/benin_clean.csv'
        sierraleone_path = 'data/sierraleone_clean.csv'
        togo_path = 'data/togo_clean.csv'
        
        # Check if files exist
        if not (os.path.exists(benin_path) and 
                os.path.exists(sierraleone_path) and 
                os.path.exists(togo_path)):
            print("One or more data files not found. Using mock data instead.")
            return None
        
        # Load cleaned datasets for each country
        df_benin = pd.read_csv(benin_path)
        df_sierraleone = pd.read_csv(sierraleone_path)
        df_togo = pd.read_csv(togo_path)
        
        # Add a 'country' column to identify the source of each row
        df_benin['country'] = 'Benin'
        df_sierraleone['country'] = 'Sierra Leone'
        df_togo['country'] = 'Togo'
        
        # Combine all datasets into one DataFrame
        df_combined = pd.concat([df_benin, df_sierraleone, df_togo], ignore_index=True)
        
        # Ensure timestamp is in datetime format
        # Check if 'timestamp' or 'Timestamp' exists
        timestamp_col = 'timestamp' if 'timestamp' in df_combined.columns else 'Timestamp'
        df_combined.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
        
        # Convert timestamp to datetime
        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
        
        # Standardize column names (lowercase)
        df_combined.columns = [col.lower() for col in df_combined.columns]
        
        # Add region column if it doesn't exist
        if 'region' not in df_combined.columns:
            # Extract region from location or station_id if available
            if 'location' in df_combined.columns:
                df_combined['region'] = df_combined['location']
            elif 'station_id' in df_combined.columns:
                df_combined['region'] = df_combined['station_id']
            else:
                # Create dummy regions based on country
                regions = {
                    'Benin': ['Alibori', 'Atacora', 'Atlantique', 'Borgou', 'Collines', 'Donga', 'Couffo', 'Littoral'],
                    'Sierra Leone': ['Eastern', 'Northern', 'Southern', 'Western', 'North West'],
                    'Togo': ['Maritime', 'Plateaux', 'Centrale', 'Kara', 'Savanes']
                }
                
                # Assign regions based on country
                df_combined['region'] = df_combined.apply(
                    lambda row: np.random.choice(regions[row['country']]), axis=1
                )
        
        # Ensure all required columns exist
        required_cols = ['ghi', 'dni', 'dhi', 'tamb', 'rh', 'ws', 'bp', 'precipitation']
        for col in required_cols:
            if col not in df_combined.columns:
                if col == 'precipitation' and 'precip' in df_combined.columns:
                    df_combined['precipitation'] = df_combined['precip']
                elif col == 'tamb' and 'temp' in df_combined.columns:
                    df_combined['tamb'] = df_combined['temp']
                elif col == 'bp' and 'pressure' in df_combined.columns:
                    df_combined['bp'] = df_combined['pressure']
                else:
                    # Generate mock data for missing columns
                    print(f"Column {col} not found. Generating mock values.")
                    if col in ['ghi', 'dni', 'dhi']:
                        df_combined[col] = np.random.uniform(300, 800, size=len(df_combined))
                    elif col == 'tamb':
                        df_combined[col] = np.random.uniform(20, 35, size=len(df_combined))
                    elif col == 'rh':
                        df_combined[col] = np.random.uniform(30, 90, size=len(df_combined))
                    elif col == 'ws':
                        df_combined[col] = np.random.uniform(1, 8, size=len(df_combined))
                    elif col == 'bp':
                        df_combined[col] = np.random.uniform(1000, 1020, size=len(df_combined))
                    elif col == 'precipitation':
                        df_combined[col] = np.random.exponential(1, size=len(df_combined))
        
        # Standardize column names to uppercase for consistency with the rest of the code
        df_combined.rename(columns={
            'ghi': 'GHI',
            'dni': 'DNI',
            'dhi': 'DHI',
            'tamb': 'Tamb',
            'rh': 'RH',
            'ws': 'WS',
            'bp': 'BP',
            'precipitation': 'Precipitation'
        }, inplace=True)
        
        print(f"Successfully loaded actual data with {len(df_combined)} records.")
        return df_combined
        
    except Exception as e:
        print(f"Error loading actual data: {str(e)}. Using mock data instead.")
        return None

def load_data():
    """
    Load data, first trying actual data files, then falling back to mock data if needed.
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with solar metrics data
    """
    # Try to load actual data first
    actual_data = load_actual_data()
    
    # If actual data loading failed, generate mock data
    if actual_data is None:
        print("Generating mock data as fallback.")
        return generate_mock_data()
    
    return actual_data

def filter_data_by_countries(df, countries):
    """
    Filter dataframe by selected countries
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to filter
    countries : list
        List of countries to include
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    """
    return df[df['country'].isin(countries)]

def filter_data_by_date_range(df, start_date, end_date):
    """
    Filter dataframe by date range
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to filter
    start_date : datetime.date
        Start date for filtering
    end_date : datetime.date
        End date for filtering
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    """
    # Convert dates to datetime if they're not already
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Filter the dataframe
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    return df[mask]

def filter_data_by_metric_range(df, metric, min_val, max_val):
    """
    Filter dataframe by metric range
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to filter
    metric : str
        The metric to filter by
    min_val : float
        Minimum value for the metric
    max_val : float
        Maximum value for the metric
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    """
    return df[(df[metric] >= min_val) & (df[metric] <= max_val)]

def get_top_regions(df, metric, n=10):
    """
    Get top n regions based on the average value of a metric
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    metric : str
        The metric to rank by
    n : int
        Number of top regions to return
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with top regions
    """
    # Group by country and region, calculate mean
    grouped = df.groupby(['country', 'region'])[metric].mean().reset_index()
    grouped = grouped.rename(columns={metric: 'mean_value'})
    
    # Sort by mean value and get top n
    top_regions = grouped.sort_values('mean_value', ascending=False).head(n)
    
    return top_regions

def calculate_solar_potential_score(df):
    """
    Calculate a solar potential score based on multiple metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with solar metrics
        
    Returns:
    --------
    float
        Solar potential score (0-10)
    """
    # Calculate average values for key metrics
    avg_ghi = df['GHI'].mean()
    avg_dni = df['DNI'].mean()
    avg_dhi = df['DHI'].mean()
    avg_temp = df['Tamb'].mean()
    avg_rh = df['RH'].mean()
    avg_ws = df['WS'].mean()
    
    # Normalize values to 0-1 scale based on typical ranges
    # These ranges should be adjusted based on actual data in a real application
    norm_ghi = min(avg_ghi / 1000, 1)  # Assuming max GHI of 1000 W/m²
    norm_dni = min(avg_dni / 1200, 1)  # Assuming max DNI of 1200 W/m²
    norm_dhi = min(avg_dhi / 400, 1)   # Assuming max DHI of 400 W/m²
    
    # Temperature factor (optimal around 25°C, decreasing above 30°C)
    temp_factor = 1 - max(0, min((avg_temp - 25) / 20, 0.3))
    
    # Humidity factor (lower is better for solar panels)
    humidity_factor = 1 - (avg_rh / 100) * 0.3
    
    # Wind speed factor (moderate wind is good for cooling panels)
    wind_factor = min(avg_ws / 5, 1) if avg_ws <= 5 else 1 - min((avg_ws - 5) / 15, 0.3)
    
    # Calculate weighted score (0-10 scale)
    # Weights should be adjusted based on importance of each factor
    score = (
        norm_ghi * 0.4 +
        norm_dni * 0.3 +
        norm_dhi * 0.1 +
        temp_factor * 0.1 +
        humidity_factor * 0.05 +
        wind_factor * 0.05
    ) * 10
    
    return score

def format_metric_name(metric):
    """
    Format metric names for display
    
    Parameters:
    -----------
    metric : str
        The metric name to format
        
    Returns:
    --------
    str
        Formatted metric name
    """
    metric_names = {
        'GHI': 'Global Horizontal Irradiance (W/m²)',
        'DNI': 'Direct Normal Irradiance (W/m²)',
        'DHI': 'Diffuse Horizontal Irradiance (W/m²)',
        'Tamb': 'Ambient Temperature (°C)',
        'RH': 'Relative Humidity (%)',
        'WS': 'Wind Speed (m/s)',
        'BP': 'Barometric Pressure (hPa)',
        'Precipitation': 'Precipitation (mm)'
    }
    
    return metric_names.get(metric, metric)

def generate_mock_data():
    """
    Generate realistic mock data for solar metrics
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with mock solar data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define countries and regions
    countries = ['Benin', 'Sierra Leone', 'Togo']
    regions = {
        'Benin': ['Alibori', 'Atacora', 'Atlantique', 'Borgou', 'Collines', 'Donga', 'Couffo', 'Littoral'],
        'Sierra Leone': ['Eastern', 'Northern', 'Southern', 'Western', 'North West'],
        'Togo': ['Maritime', 'Plateaux', 'Centrale', 'Kara', 'Savanes']
    }
    
    # Create date range (2 years of hourly data would be too large, so we'll use daily data)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2022, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create empty list to store data
    data = []
    
    # Generate data for each country and region
    for country in countries:
        for region in regions[country]:
            # Base values for each region (with some randomness)
            # These values are adjusted to create realistic differences between regions
            if country == 'Benin':
                base_ghi = np.random.uniform(550, 650)
                base_dni = np.random.uniform(700, 800)
                base_dhi = np.random.uniform(150, 200)
                base_temp = np.random.uniform(26, 30)
                base_rh = np.random.uniform(65, 80)
                base_ws = np.random.uniform(2, 4)
                base_bp = np.random.uniform(1010, 1015)
                base_precip = np.random.uniform(0.5, 2)
            elif country == 'Sierra Leone':
                base_ghi = np.random.uniform(500, 600)
                base_dni = np.random.uniform(650, 750)
                base_dhi = np.random.uniform(180, 230)
                base_temp = np.random.uniform(25, 29)
                base_rh = np.random.uniform(75, 90)
                base_ws = np.random.uniform(1.5, 3.5)
                base_bp = np.random.uniform(1008, 1013)
                base_precip = np.random.uniform(1, 3)
            else:  # Togo
                base_ghi = np.random.uniform(525, 625)
                base_dni = np.random.uniform(675, 775)
                base_dhi = np.random.uniform(165, 215)
                base_temp = np.random.uniform(25.5, 29.5)
                base_rh = np.random.uniform(70, 85)
                base_ws = np.random.uniform(1.8, 3.8)
                base_bp = np.random.uniform(1009, 1014)
                base_precip = np.random.uniform(0.8, 2.5)
            
            for date in date_range:
                # Add seasonal variations
                day_of_year = date.dayofyear
                season_factor = np.sin((day_of_year / 365) * 2 * np.pi - np.pi/2) * 0.2 + 1
                
                # Random daily variation
                daily_variation = np.random.normal(1, 0.1)
                
                # Calculate values with seasonal and daily variations
                ghi = base_ghi * season_factor * daily_variation
                dni = base_dni * season_factor * daily_variation
                dhi = base_dhi * season_factor * daily_variation
                
                # Temperature varies with season but less directly than irradiance
                temp_season_factor = np.sin((day_of_year / 365) * 2 * np.pi - np.pi/2) * 0.15 + 1
                temp = base_temp * temp_season_factor * np.random.normal(1, 0.05)
                
                # Humidity often inversely related to temperature
                rh = base_rh * (2 - temp_season_factor) * np.random.normal(1, 0.08)
                rh = min(max(rh, 30), 100)  # Constrain between 30% and 100%
                
                # Wind speed with some seasonal variation
                ws = base_ws * np.random.normal(1, 0.2)
                
                # Barometric pressure with slight seasonal variation
                bp_season_factor = np.cos((day_of_year / 365) * 2 * np.pi) * 0.005 + 1
                bp = base_bp * bp_season_factor * np.random.normal(1, 0.002)
                
                # Precipitation with seasonal patterns (more in rainy season)
                # Rainy season varies by country
                if country == 'Benin':
                    # Rainy season: May to October
                    is_rainy_season = 5 <= date.month <= 10
                elif country == 'Sierra Leone':
                    # Rainy season: May to November
                    is_rainy_season = 5 <= date.month <= 11
                else:  # Togo
                    # Rainy season: April to October
                    is_rainy_season = 4 <= date.month <= 10
                
                precip_factor = 3 if is_rainy_season else 0.5
                precip = base_precip * precip_factor * np.random.exponential(1)
                
                # On rainy days, reduce irradiance
                if precip > 5:
                    rain_reduction = max(0.5, 1 - (precip / 50))
                    ghi *= rain_reduction
                    dni *= rain_reduction * 0.8  # DNI affected more by rain
                    dhi *= rain_reduction * 1.2  # DHI less affected
                
                # Add row to data
                data.append({
                    'timestamp': date,
                    'country': country,
                    'region': region,
                    'GHI': ghi,
                    'DNI': dni,
                    'DHI': dhi,
                    'Tamb': temp,
                    'RH': rh,
                    'WS': ws,
                    'BP': bp,
                    'Precipitation': precip
                })
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    return df

def perform_statistical_test(df, metric, countries):
    """
    Perform statistical test to compare metric across countries
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with data
    metric : str
        The metric to compare
    countries : list
        List of countries to compare
        
    Returns:
    --------
    tuple
        (test_name, statistic, p_value, interpretation)
    """
    from scipy import stats
    
    # Need at least 2 countries to compare
    if len(countries) < 2:
        return None
    
    # Get data for each country
    country_data = []
    for country in countries:
        country_df = df[df['country'] == country]
        if len(country_df) > 0:
            country_data.append(country_df[metric].values)
    
    # Need at least 2 countries with data
    if len(country_data) < 2:
        return None
    
    # Perform Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA)
    # This is more robust when we can't assume normality
    statistic, p_value = stats.kruskal(*country_data)
    
    # Interpretation
    if p_value < 0.05:
        interpretation = "There are significant differences in {} between countries (p < 0.05).".format(metric)
    else:
        interpretation = "No significant differences in {} between countries (p >= 0.05).".format(metric)
    
    return ("Kruskal-Wallis H-test", statistic, p_value, interpretation)

def get_correlation_insights(df):
    """
    Generate insights about correlations between metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with data
        
    Returns:
    --------
    list
        List of correlation insights
    """
    # Calculate correlation matrix
    corr_metrics = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS', 'BP', 'Precipitation']
    corr_matrix = df[corr_metrics].corr()
    
    # Get key correlations
    insights = []
    
    # GHI correlations
    ghi_tamb_corr = corr_matrix.loc['GHI', 'Tamb']
    ghi_rh_corr = corr_matrix.loc['GHI', 'RH']
    ghi_ws_corr = corr_matrix.loc['GHI', 'WS']
    
    # Describe correlation strength
    def describe_correlation(corr):
        if abs(corr) > 0.7:
            strength = "strong"
        elif abs(corr) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if corr > 0 else "negative"
        return f"{strength} {direction}"
    
    # Add insights
    insights.append(f"GHI and temperature have a {describe_correlation(ghi_tamb_corr)} correlation ({ghi_tamb_corr:.2f}).")
    insights.append(f"GHI and relative humidity have a {describe_correlation(ghi_rh_corr)} correlation ({ghi_rh_corr:.2f}).")
    insights.append(f"GHI and wind speed have a {describe_correlation(ghi_ws_corr)} correlation ({ghi_ws_corr:.2f}).")
    
    # DNI correlations
    dni_precip_corr = corr_matrix.loc['DNI', 'Precipitation']
    insights.append(f"DNI and precipitation have a {describe_correlation(dni_precip_corr)} correlation ({dni_precip_corr:.2f}).")
    
    return insights

def get_seasonal_insights(df, metric):
    """
    Generate insights about seasonal patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with data
    metric : str
        The metric to analyze
        
    Returns:
    --------
    dict
        Dictionary with seasonal insights
    """
    # Add month information
    df_with_month = df.copy()
    df_with_month['month'] = df_with_month['timestamp'].dt.month
    df_with_month['month_name'] = df_with_month['timestamp'].dt.month_name()
    
    # Get monthly averages
    monthly_avg = df_with_month.groupby('month')[metric].mean()
    
    # Find best and worst months
    best_month_idx = monthly_avg.idxmax()
    worst_month_idx = monthly_avg.idxmin()
    
    # Get month names
    import calendar
    best_month = calendar.month_name[best_month_idx]
    worst_month = calendar.month_name[worst_month_idx]
    
    # Calculate seasonal averages
    df_with_month['season'] = df_with_month['month'].apply(lambda m: 
        'Winter' if m in [12, 1, 2] else
        'Spring' if m in [3, 4, 5] else
        'Summer' if m in [6, 7, 8] else
        'Fall'
    )
    
    seasonal_avg = df_with_month.groupby('season')[metric].mean().to_dict()
    
    # Return insights
    return {
        'best_month': best_month,
        'best_month_value': monthly_avg[best_month_idx],
        'worst_month': worst_month,
        'worst_month_value': monthly_avg[worst_month_idx],
        'seasonal_averages': seasonal_avg
    }

def get_actionable_recommendations(df, primary_metric):
    """
    Generate actionable recommendations based on data analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with data
    primary_metric : str
        The primary metric being analyzed
        
    Returns:
    --------
    list
        List of actionable recommendations
    """
    recommendations = []
    
    # Get top regions
    top_regions = get_top_regions(df, primary_metric, n=3)
    
    # Get seasonal insights
    seasonal = get_seasonal_insights(df, primary_metric)
    
    # Calculate DNI/GHI ratio
    dni_ghi_ratio = df['DNI'].mean() / df['GHI'].mean()
    
    # Environmental factors
    avg_ws = df['WS'].mean()
    avg_rh = df['RH'].mean()
    
    # Priority locations
    top_region = top_regions.iloc[0]
    recommendations.append(f"Focus initial solar farm development in {top_region['region']}, {top_region['country']} due to consistently high {format_metric_name(primary_metric)} values.")
    
    # Seasonal planning
    recommendations.append(f"Schedule maintenance during {seasonal['worst_month']} when {primary_metric} is lowest, and ensure full operational capacity during {seasonal['best_month']} to maximize energy production.")
    
    # Technology selection
    if dni_ghi_ratio > 0.8:
        recommendations.append("The high DNI/GHI ratio suggests concentrated solar power (CSP) technology may be suitable for these regions.")
    else:
        recommendations.append("The DNI/GHI ratio indicates that photovoltaic (PV) panels would be more efficient than concentrated solar power in these regions.")
    
    # Wind integration
    if avg_ws > 4:
        recommendations.append(f"With average wind speeds of {avg_ws:.2f} m/s, consider hybrid solar-wind installations to maximize energy production throughout the day and year.")
    else:
        recommendations.append(f"The relatively low average wind speed ({avg_ws:.2f} m/s) suggests wind power may not be viable as a complementary source.")
    
    # Humidity considerations
    if avg_rh > 70:
        recommendations.append(f"High average humidity ({avg_rh:.2f}%) requires regular panel cleaning systems to maintain efficiency and prevent mold growth.")
    else:
        recommendations.append(f"The moderate humidity levels ({avg_rh:.2f}%) allow for standard maintenance protocols.")
    
    return recommendations