import pandas as pd
import pymannkendall as mk

# Load the dataset
df = pd.read_csv("E:/Nava/Urmia Lake Research/Urmia_Lake_Climate_Data_month.csv", parse_dates=['date'])

# Define the columns of interest
columns_of_interest = ['temp_celsius', 'precip_mm', 'pressure_pa', 'soil_moisture', 'LWE']

# Define periods
periods = {
    "2003–2013": (df['date'].dt.year >= 2003) & (df['date'].dt.year <= 2013),
    "2014–2023": (df['date'].dt.year >= 2014) & (df['date'].dt.year <= 2023),
    "2003–2023": (df['date'].dt.year >= 2003) & (df['date'].dt.year <= 2023),
}

# Function to extract MK results nicely
def extract_mk_results(series):
    result = mk.original_test(series)
    trend = result.trend
    slope = result.slope
    p_value = result.p
    s_stat = result.s
    return trend, slope, p_value, s_stat

# Create the table
for col in columns_of_interest:
    print(f"\n==== {col} ====")
    table = {"Parameter": [], "Trend": [], "Slope": [], "P-value": [], "Test Statistic (S)": []}
    
    for period_name, mask in periods.items():
        data_period = df.loc[mask, col]
        trend, slope, p_value, s_stat = extract_mk_results(data_period)
        table["Parameter"].append(period_name)
        table["Trend"].append(trend.capitalize())
        table["Slope"].append(round(slope, 4))
        table["P-value"].append(round(p_value, 5))
        table["Test Statistic (S)"].append(round(s_stat, 1))
    
    # Display as DataFrame
    display(pd.DataFrame(table).set_index("Parameter"))
