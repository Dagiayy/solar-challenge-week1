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
    get_module_performance_insights,
    get_actionable_recommendations
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

# Secondary metric for bubble chart
secondary_options = [m for m in available_metrics if m != primary_metric]
secondary_metric = st.sidebar.selectbox(
    "Select Secondary Metric (for Bubble Chart)",
    options=secondary_options,
    index=0
)

# Tertiary metric for bubble size
tertiary_options = [m for m in available_metrics if m not in [primary_metric, secondary_metric]]
tertiary_metric = st.sidebar.selectbox(
    "Select Tertiary Metric (for Bubble Size)",
    options=tertiary_options,
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
    for metric in available_metrics:
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Comparison Analysis", 
    "Time Series Analysis", 
    "Correlation Analysis", 
    "Regional Rankings",
    "Module Performance"
])

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
    
    # Calculate correlation matrix for key metrics
    corr_metrics = [m for m in available_metrics if m != 'Cleaning']  # Exclude binary variables
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

with tab5:
    st.markdown('<div class="section-header">Module Performance Analysis</div>', unsafe_allow_html=True)
    
    # Check if module data is available
    if 'ModA' in filtered_df.columns and 'ModB' in filtered_df.columns:
        # Module comparison
        module_data = filtered_df.groupby('country')[['ModA', 'ModB']].mean().reset_index()
        
        # Create bar chart for module comparison
        fig_modules = px.bar(
            module_data,
            x='country',
            y=['ModA', 'ModB'],
            barmode='group',
            title="Module Performance Comparison by Country",
            labels={
                'country': 'Country',
                'value': 'Irradiance (W/m²)',
                'variable': 'Module'
            },
            color_discrete_sequence=['#4CAF50', '#2196F3'],
            template="plotly_dark"
        )
        
        fig_modules.update_layout(
            xaxis_title="Country",
            yaxis_title="Irradiance (W/m²)",
            legend_title="Module",
            font=dict(size=12),
            height=400
        )
        
        st.plotly_chart(fig_modules, use_container_width=True)
        
        # Module temperature analysis if available
        if 'TModA' in filtered_df.columns and 'TModB' in filtered_df.columns:
            st.markdown('<div class="section-header">Module Temperature Analysis</div>', unsafe_allow_html=True)
            
            # Create scatter plot of module output vs temperature
            fig_temp = px.scatter(
                filtered_df,
                x='TModA',
                y='ModA',
                color='country',
                opacity=0.7,
                title="Module A Output vs Temperature",
                labels={
                    'TModA': 'Module A Temperature (°C)',
                    'ModA': 'Module A Irradiance (W/m²)',
                    'country': 'Country'
                },
                trendline="ols",
                template="plotly_dark"
            )
            
            fig_temp.update_layout(
                xaxis_title="Module A Temperature (°C)",
                yaxis_title="Module A Irradiance (W/m²)",
                legend_title="Country",
                font=dict(size=12),
                height=400
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Temperature distribution
            fig_temp_dist = px.box(
                filtered_df,
                x='country',
                y=['TModA', 'TModB', 'Tamb'],
                title="Temperature Distribution by Country",
                labels={
                    'country': 'Country',
                    'value': 'Temperature (°C)',
                    'variable': 'Measurement'
                },
                color_discrete_sequence=['#FF9800', '#F44336', '#2196F3'],
                template="plotly_dark"
            )
            
            fig_temp_dist.update_layout(
                xaxis_title="Country",
                yaxis_title="Temperature (°C)",
                legend_title="Measurement",
                font=dict(size=12),
                height=400
            )
            
            st.plotly_chart(fig_temp_dist, use_container_width=True)
        
        # Cleaning impact analysis if available
        if 'Cleaning' in filtered_df.columns:
            st.markdown('<div class="section-header">Cleaning Impact Analysis</div>', unsafe_allow_html=True)
            
            # Group by cleaning status
            cleaning_impact = filtered_df.groupby('Cleaning')[['ModA', 'ModB']].mean().reset_index()
            cleaning_impact['Cleaning'] = cleaning_impact['Cleaning'].map({0: 'No Cleaning', 1: 'After Cleaning'})
            
            # Create bar chart for cleaning impact
            fig_cleaning = px.bar(
                cleaning_impact,
                x='Cleaning',
                y=['ModA', 'ModB'],
                barmode='group',
                title="Module Performance Before and After Cleaning",
                labels={
                    'Cleaning': 'Cleaning Status',
                    'value': 'Irradiance (W/m²)',
                    'variable': 'Module'
                },
                color_discrete_sequence=['#4CAF50', '#2196F3'],
                template="plotly_dark"
            )
            
            fig_cleaning.update_layout(
                xaxis_title="Cleaning Status",
                yaxis_title="Irradiance (W/m²)",
                legend_title="Module",
                font=dict(size=12),
                height=400
            )
            
            st.plotly_chart(fig_cleaning, use_container_width=True)
        
        # Get module performance insights
        module_insights = get_module_performance_insights(filtered_df)
        
        if module_insights:
            st.markdown('<div class="section-header">Module Performance Insights</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box">
                <p><strong>Module Performance Comparison:</strong></p>
                <p>Average Module A Output: <span class="highlight">{module_insights['avg_moda']:.2f} W/m²</span></p>
                <p>Average Module B Output: <span class="highlight">{module_insights['avg_modb']:.2f} W/m²</span></p>
                <p>Difference: <span class="highlight">{module_insights['module_difference_pct']:.2f}%</span></p>
                
                {f'<p><strong>Temperature Impact:</strong></p><p>Module performance changes approximately <span class="highlight">{module_insights["temp_coefficient"]:.2f}%</span> per degree Celsius.</p>' if module_insights["temp_coefficient"] is not None else ''}
                
                {f'<p><strong>Cleaning Impact:</strong></p><p>Panel cleaning improves performance by an average of <span class="highlight">{module_insights["cleaning_improvement"]:.2f}%</span>.</p>' if module_insights["cleaning_improvement"] is not None else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Module performance data (ModA, ModB) is not available in the current dataset.")

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
    
    # Wind insights if available
    if 'WSgust' in filtered_df.columns and 'WSstdev' in filtered_df.columns:
        avg_ws = filtered_df['WS'].mean()
        avg_wsgust = filtered_df['WSgust'].mean()
        avg_wsstdev = filtered_df['WSstdev'].mean()
        
        st.markdown(f"""
        <div class="info-box">
            <p><strong>Wind Characteristics:</strong></p>
            <p>Average wind speed: <span class="highlight">{avg_ws:.2f} m/s</span></p>
            <p>Average gust speed: <span class="highlight">{avg_wsgust:.2f} m/s</span> ({(avg_wsgust/avg_ws - 1)*100:.1f}% higher than average)</p>
            <p>Wind variability (standard deviation): <span class="highlight">{avg_wsstdev:.2f} m/s</span></p>
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

