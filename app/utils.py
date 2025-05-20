import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_data():
    """
    Load actual data from CSV files.
    
    Returns:
    --------
    pandas.DataFrame
        Combined dataframe containing data from all countries.
    """
    try:
        # Define file paths
        benin_path = 'data/benin-malamville_clean_clean.csv'
        sierraleone_path = 'data/sierraleone-bumbuna_clean.csv'
        togo_path = 'data/togo-dapaong_clean.csv'
        
        # Check if files exist
        if not (os.path.exists(benin_path) and 
                os.path.exists(sierraleone_path) and 
                os.path.exists(togo_path)):
            print("One or more data files not found.")
            return pd.DataFrame()
        
        # Load datasets for each country
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
        timestamp_col = next((col for col in df_combined.columns if col.lower() == 'timestamp'), None)
        if timestamp_col:
            df_combined.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
            df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
        
        # Standardize column names based on the provided structure
        column_mapping = {
            # Map any variations to the standardized column names
            'ghi': 'GHI',
            'dni': 'DNI',
            'dhi': 'DHI',
            'moda': 'ModA',
            'modb': 'ModB',
            'tamb': 'Tamb',
            'temp': 'Tamb',
            'rh': 'RH',
            'ws': 'WS',
            'wsgust': 'WSgust',
            'wsstdev': 'WSstdev',
            'wd': 'WD',
            'wdstdev': 'WDstdev',
            'bp': 'BP',
            'pressure': 'BP',
            'cleaning': 'Cleaning',
            'precipitation': 'Precipitation',
            'precip': 'Precipitation',
            'tmoda': 'TModA',
            'tmodb': 'TModB'
        }
        
        # Standardize column names (case-insensitive)
        for col in df_combined.columns:
            col_lower = col.lower()
            if col_lower in column_mapping:
                df_combined.rename(columns={col: column_mapping[col_lower]}, inplace=True)
        
        # Add region column if it doesn't exist
        if 'region' not in df_combined.columns:
            # Extract region from location or station_id if available
            if 'location' in df_combined.columns:
                df_combined['region'] = df_combined['location']
            elif 'station_id' in df_combined.columns:
                df_combined['region'] = df_combined['station_id']
            else:
                # Use city names from file paths as regions
                df_combined['region'] = df_combined['country'].map({
                    'Benin': 'Malamville',
                    'Sierra Leone': 'Bumbuna',
                    'Togo': 'Dapaong'
                })
        
        print(f"Successfully loaded data with {len(df_combined)} records.")
        return df_combined
        
    except Exception as e:
        print(f"Error loading data: {str(e)}.")
        return pd.DataFrame()

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
    avg_ghi = df['GHI'].mean() if 'GHI' in df.columns else 0
    avg_dni = df['DNI'].mean() if 'DNI' in df.columns else 0
    avg_dhi = df['DHI'].mean() if 'DHI' in df.columns else 0
    avg_temp = df['Tamb'].mean() if 'Tamb' in df.columns else 25
    avg_rh = df['RH'].mean() if 'RH' in df.columns else 50
    avg_ws = df['WS'].mean() if 'WS' in df.columns else 3
    
    # Normalize values to 0-1 scale based on typical ranges
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
        'ModA': 'Module A Irradiance (W/m²)',
        'ModB': 'Module B Irradiance (W/m²)',
        'Tamb': 'Ambient Temperature (°C)',
        'RH': 'Relative Humidity (%)',
        'WS': 'Wind Speed (m/s)',
        'WSgust': 'Wind Gust Speed (m/s)',
        'WSstdev': 'Wind Speed Std Dev (m/s)',
        'WD': 'Wind Direction (°N)',
        'WDstdev': 'Wind Direction Std Dev',
        'BP': 'Barometric Pressure (hPa)',
        'Cleaning': 'Cleaning Status',
        'Precipitation': 'Precipitation (mm/min)',
        'TModA': 'Module A Temperature (°C)',
        'TModB': 'Module B Temperature (°C)'
    }
    
    return metric_names.get(metric, metric)

def get_correlation_insights(df, primary_metric):
    """
    Generate insights about correlations with the primary metric
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with data
    primary_metric : str
        The primary metric to analyze correlations for
        
    Returns:
    --------
    list
        List of correlation insights
    """
    # Calculate correlations with the primary metric
    corr_metrics = [col for col in df.columns if col not in ['timestamp', 'country', 'region', 'Cleaning']]
    
    insights = []
    
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
    
    # Get correlations with primary metric
    for metric in corr_metrics:
        if metric != primary_metric and metric in df.columns:
            try:
                corr = df[[primary_metric, metric]].corr().iloc[0, 1]
                insights.append(f"{primary_metric} and {metric} have a {describe_correlation(corr)} correlation ({corr:.2f}).")
            except:
                pass
    
    return insights[:5]  # Return top 5 insights