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
        benin_path ='../data/benin-malamville_clean_clean.csv'
        sierraleone_path = '../data/sierraleone-bumbuna_clean.csv'
        togo_path = '../data/togo-dapaong_clean.csv'
        
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
        required_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH', 'WS', 
                         'WSgust', 'WSstdev', 'WD', 'WDstdev', 'BP', 'Cleaning', 
                         'Precipitation', 'TModA', 'TModB']
        
        for col in required_cols:
            if col not in df_combined.columns:
                # Generate mock data for missing columns
                print(f"Column {col} not found. Generating mock values.")
                if col in ['GHI', 'DNI', 'DHI']:
                    df_combined[col] = np.random.uniform(300, 800, size=len(df_combined))
                elif col in ['ModA', 'ModB']:
                    df_combined[col] = np.random.uniform(300, 800, size=len(df_combined))
                elif col == 'Tamb':
                    df_combined[col] = np.random.uniform(20, 35, size=len(df_combined))
                elif col == 'RH':
                    df_combined[col] = np.random.uniform(30, 90, size=len(df_combined))
                elif col == 'WS':
                    df_combined[col] = np.random.uniform(1, 8, size=len(df_combined))
                elif col == 'WSgust':
                    df_combined[col] = df_combined['WS'] * np.random.uniform(1.2, 2.0, size=len(df_combined))
                elif col == 'WSstdev':
                    df_combined[col] = df_combined['WS'] * np.random.uniform(0.1, 0.3, size=len(df_combined))
                elif col == 'WD':
                    df_combined[col] = np.random.uniform(0, 360, size=len(df_combined))
                elif col == 'WDstdev':
                    df_combined[col] = np.random.uniform(5, 30, size=len(df_combined))
                elif col == 'BP':
                    df_combined[col] = np.random.uniform(1000, 1020, size=len(df_combined))
                elif col == 'Cleaning':
                    df_combined[col] = np.random.choice([0, 1], size=len(df_combined), p=[0.95, 0.05])
                elif col == 'Precipitation':
                    df_combined[col] = np.random.exponential(0.1, size=len(df_combined))
                elif col in ['TModA', 'TModB']:
                    df_combined[col] = df_combined['Tamb'] + np.random.uniform(5, 15, size=len(df_combined))
        
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
    
    # Include module data if available
    if 'ModA' in df.columns and 'ModB' in df.columns:
        avg_moda = df['ModA'].mean()
        avg_modb = df['ModB'].mean()
        # Use average of modules
        avg_mod = (avg_moda + avg_modb) / 2
    else:
        avg_mod = None
    
    # Normalize values to 0-1 scale based on typical ranges
    norm_ghi = min(avg_ghi / 1000, 1)  # Assuming max GHI of 1000 W/m²
    norm_dni = min(avg_dni / 1200, 1)  # Assuming max DNI of 1200 W/m²
    norm_dhi = min(avg_dhi / 400, 1)   # Assuming max DHI of 400 W/m²
    
    # Include module data in normalization if available
    if avg_mod is not None:
        norm_mod = min(avg_mod / 1000, 1)
    else:
        norm_mod = 0
    
    # Temperature factor (optimal around 25°C, decreasing above 30°C)
    temp_factor = 1 - max(0, min((avg_temp - 25) / 20, 0.3))
    
    # Humidity factor (lower is better for solar panels)
    humidity_factor = 1 - (avg_rh / 100) * 0.3
    
    # Wind speed factor (moderate wind is good for cooling panels)
    wind_factor = min(avg_ws / 5, 1) if avg_ws <= 5 else 1 - min((avg_ws - 5) / 15, 0.3)
    
    # Calculate weighted score (0-10 scale)
    # Weights should be adjusted based on importance of each factor
    score = (
        norm_ghi * 0.35 +
        norm_dni * 0.25 +
        norm_dhi * 0.1 +
        norm_mod * 0.1 +
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
                base_moda = np.random.uniform(500, 600)
                base_modb = np.random.uniform(520, 620)
                base_temp = np.random.uniform(26, 30)
                base_rh = np.random.uniform(65, 80)
                base_ws = np.random.uniform(2, 4)
                base_bp = np.random.uniform(1010, 1015)
                base_precip = np.random.uniform(0.5, 2)
            elif country == 'Sierra Leone':
                base_ghi = np.random.uniform(500, 600)
                base_dni = np.random.uniform(650, 750)
                base_dhi = np.random.uniform(180, 230)
                base_moda = np.random.uniform(450, 550)
                base_modb = np.random.uniform(470, 570)
                base_temp = np.random.uniform(25, 29)
                base_rh = np.random.uniform(75, 90)
                base_ws = np.random.uniform(1.5, 3.5)
                base_bp = np.random.uniform(1008, 1013)
                base_precip = np.random.uniform(1, 3)
            else:  # Togo
                base_ghi = np.random.uniform(525, 625)
                base_dni = np.random.uniform(675, 775)
                base_dhi = np.random.uniform(165, 215)
                base_moda = np.random.uniform(475, 575)
                base_modb = np.random.uniform(495, 595)
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
                moda = base_moda * season_factor * daily_variation
                modb = base_modb * season_factor * daily_variation
                
                # Temperature varies with season but less directly than irradiance
                temp_season_factor = np.sin((day_of_year / 365) * 2 * np.pi - np.pi/2) * 0.15 + 1
                temp = base_temp * temp_season_factor * np.random.normal(1, 0.05)
                
                # Module temperatures are typically higher than ambient
                tmoda = temp + np.random.uniform(10, 15)
                tmodb = temp + np.random.uniform(10, 15)
                
                # Humidity often inversely related to temperature
                rh = base_rh * (2 - temp_season_factor) * np.random.normal(1, 0.08)
                rh = min(max(rh, 30), 100)  # Constrain between 30% and 100%
                
                # Wind speed with some seasonal variation
                ws = base_ws * np.random.normal(1, 0.2)
                wsgust = ws * np.random.uniform(1.2, 2.0)
                wsstdev = ws * np.random.uniform(0.1, 0.3)
                
                # Wind direction
                wd = np.random.uniform(0, 360)
                wdstdev = np.random.uniform(5, 30)
                
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
                    moda *= rain_reduction
                    modb *= rain_reduction
                
                # Cleaning status (1 = cleaning occurred, 0 = no cleaning)
                # More likely to clean after rainy days or every 30 days
                cleaning = 1 if (precip > 10 and np.random.random() < 0.3) or (date.day % 30 == 0) else 0
                
                # Add row to data
                data.append({
                    'timestamp': date,
                    'country': country,
                    'region': region,
                    'GHI': ghi,
                    'DNI': dni,
                    'DHI': dhi,
                    'ModA': moda,
                    'ModB': modb,
                    'Tamb': temp,
                    'RH': rh,
                    'WS': ws,
                    'WSgust': wsgust,
                    'WSstdev': wsstdev,
                    'WD': wd,
                    'WDstdev': wdstdev,
                    'BP': bp,
                    'Cleaning': cleaning,
                    'Precipitation': precip,
                    'TModA': tmoda,
                    'TModB': tmodb
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
        interpretation = f"There are significant differences in {format_metric_name(metric)} between countries (p < 0.05)."
    else:
        interpretation = f"No significant differences in {format_metric_name(metric)} between countries (p >= 0.05)."
    
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
    # Calculate correlation matrix for key metrics
    corr_metrics = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS', 'BP', 'Precipitation']
    
    # Add module metrics if available
    if 'ModA' in df.columns:
        corr_metrics.append('ModA')
    if 'ModB' in df.columns:
        corr_metrics.append('ModB')
    if 'TModA' in df.columns:
        corr_metrics.append('TModA')
    if 'TModB' in df.columns:
        corr_metrics.append('TModB')
    
    corr_matrix = df[corr_metrics].corr()
    
    # Get key correlations
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
    
    # GHI correlations
    ghi_tamb_corr = corr_matrix.loc['GHI', 'Tamb']
    ghi_rh_corr = corr_matrix.loc['GHI', 'RH']
    ghi_ws_corr = corr_matrix.loc['GHI', 'WS']
    
    # Add insights
    insights.append(f"GHI and temperature have a {describe_correlation(ghi_tamb_corr)} correlation ({ghi_tamb_corr:.2f}).")
    insights.append(f"GHI and relative humidity have a {describe_correlation(ghi_rh_corr)} correlation ({ghi_rh_corr:.2f}).")
    insights.append(f"GHI and wind speed have a {describe_correlation(ghi_ws_corr)} correlation ({ghi_ws_corr:.2f}).")
    
    # DNI correlations
    dni_precip_corr = corr_matrix.loc['DNI', 'Precipitation']
    insights.append(f"DNI and precipitation have a {describe_correlation(dni_precip_corr)} correlation ({dni_precip_corr:.2f}).")
    
    # Module correlations if available
    if 'ModA' in corr_metrics and 'TModA' in corr_metrics:
        moda_tmoda_corr = corr_matrix.loc['ModA', 'TModA']
        insights.append(f"Module A irradiance and temperature have a {describe_correlation(moda_tmoda_corr)} correlation ({moda_tmoda_corr:.2f}).")
    
    # Wind correlations if available
    if 'WS' in corr_metrics and 'WSgust' in df.columns:
        ws_wsgust_corr = df[['WS', 'WSgust']].corr().iloc[0, 1]
        insights.append(f"Wind speed and gust speed have a {describe_correlation(ws_wsgust_corr)} correlation ({ws_wsgust_corr:.2f}).")
    
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

def get_module_performance_insights(df):
    """
    Generate insights about module performance
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with data
        
    Returns:
    --------
    dict
        Dictionary with module performance insights
    """
    # Check if module data is available
    if 'ModA' not in df.columns or 'ModB' not in df.columns:
        return None
    
    # Calculate average module performance
    avg_moda = df['ModA'].mean()
    avg_modb = df['ModB'].mean()
    
    # Calculate module temperature impact if available
    if 'TModA' in df.columns and 'TModB' in df.columns:
        # Group by temperature bins and calculate average module output
        df['TModA_bin'] = pd.cut(df['TModA'], bins=range(20, 70, 5))
        df['TModB_bin'] = pd.cut(df['TModB'], bins=range(20, 70, 5))
        
        moda_by_temp = df.groupby('TModA_bin')['ModA'].mean().dropna()
        modb_by_temp = df.groupby('TModB_bin')['ModB'].mean().dropna()
        
        # Calculate temperature coefficient (% change per degree C)
        if len(moda_by_temp) > 1:
            temp_range = [(b.mid) for b in moda_by_temp.index]
            if len(temp_range) > 1:
                temp_diff = temp_range[-1] - temp_range[0]
                if temp_diff > 0:
                    moda_values = moda_by_temp.values
                    moda_diff_pct = (moda_values[-1] - moda_values[0]) / moda_values[0] * 100
                    moda_temp_coef = moda_diff_pct / temp_diff
                else:
                    moda_temp_coef = 0
            else:
                moda_temp_coef = 0
        else:
            moda_temp_coef = 0
        
        # Calculate cleaning impact if available
        if 'Cleaning' in df.columns:
            # Compare performance before and after cleaning
            cleaning_days = df[df['Cleaning'] == 1]['timestamp'].dt.date.unique()
            
            if len(cleaning_days) > 0:
                before_after_diff = []
                
                for clean_day in cleaning_days:
                    clean_day_dt = pd.to_datetime(clean_day)
                    day_before = clean_day_dt - pd.Timedelta(days=1)
                    day_after = clean_day_dt + pd.Timedelta(days=1)
                    
                    # Get data for day before and day of cleaning
                    before_df = df[(df['timestamp'].dt.date == day_before.date())]
                    after_df = df[(df['timestamp'].dt.date == day_after.date())]
                    
                    if not before_df.empty and not after_df.empty:
                        before_moda = before_df['ModA'].mean()
                        after_moda = after_df['ModA'].mean()
                        
                        # Calculate percentage improvement
                        if before_moda > 0:
                            pct_improvement = (after_moda - before_moda) / before_moda * 100
                            before_after_diff.append(pct_improvement)
                
                if before_after_diff:
                    avg_cleaning_improvement = np.mean(before_after_diff)
                else:
                    avg_cleaning_improvement = 0
            else:
                avg_cleaning_improvement = 0
        else:
            avg_cleaning_improvement = None
    else:
        moda_temp_coef = None
        avg_cleaning_improvement = None
    
    # Return insights
    return {
        'avg_moda': avg_moda,
        'avg_modb': avg_modb,
        'module_difference_pct': (avg_moda - avg_modb) / avg_modb * 100 if avg_modb > 0 else 0,
        'temp_coefficient': moda_temp_coef,
        'cleaning_improvement': avg_cleaning_improvement
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
    
    # Get module insights if available
    module_insights = get_module_performance_insights(df)
    
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
    
    # Module recommendations if available
    if module_insights:
        if abs(module_insights['module_difference_pct']) > 5:
            better_module = "A" if module_insights['avg_moda'] > module_insights['avg_modb'] else "B"
            recommendations.append(f"Module {better_module} outperforms the other by {abs(module_insights['module_difference_pct']):.1f}%. Consider standardizing on this module type for future installations.")
        
        if module_insights['temp_coefficient'] is not None and module_insights['temp_coefficient'] < -0.3:
            recommendations.append(f"Module performance decreases by approximately {abs(module_insights['temp_coefficient']):.2f}% per degree Celsius. Consider enhanced cooling solutions or temperature-resistant modules.")
        
        if module_insights['cleaning_improvement'] is not None and module_insights['cleaning_improvement'] > 3:
            recommendations.append(f"Panel cleaning improves performance by an average of {module_insights['cleaning_improvement']:.1f}%. Implement a regular cleaning schedule, especially during dry seasons.")
    
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