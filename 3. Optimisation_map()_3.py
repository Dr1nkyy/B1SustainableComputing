# Future hourly forecast takes adavantage of map function to calculate hourly average deviation for each hour of the day.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import csv

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to create features on dataframe for easier indexing

def create_features(df):
    
    df['hour']      = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    
    return df

#  Function to process data from csv file

def ProcessData_CSVModule(path):
    processed_data = []
    
    # Open the CSV file and read it line by line
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # Append the processed row to the list
                processed_data.append({
                    'DATETIME': row['DATETIME'],
                    'CARBON_INTENSITY': float(row['CARBON_INTENSITY']) if row['CARBON_INTENSITY'] else None
                })
            except (KeyError,ValueError):
                # Skip rows with invalid data
                continue
    
    df = pd.DataFrame(processed_data)
    # Set 'DATETIME' as index and process further
    df = df.set_index('DATETIME')                 # Set DATETIME as index
    df.index = pd.to_datetime(df.index)           # Convert to datetime
    df = df.resample('h').mean()                  # Resample to hourly intervals
    df.index = df.index.tz_localize(None)         # Remove timezone info
    df = create_features(df)                      # Create additional features
    
    return df

# Define data used in model

def ShorterData(df, start = '2000-01-01', end='2024-01-01'):
    df = df.loc[(df.index < end) & (df.index >= start)] # Filter data to only include data from start to 2023
    return df

# Original function to predict daily average carbon intensity for 2024 by fitting a linear regression to corresponding days of the from previous years in the data set.
def futuredailyforecast1(df, model = LinearRegression()):
    df = df.resample('D').mean()
    daily_dates_2024 = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    futuredf = pd.DataFrame(index=daily_dates_2024)
    futuredf = create_features(futuredf)
    
    for i in range(1,367):
        
        X = df[df['dayofyear']==i].index.dayofyear.values.reshape(-1,1)
        y = df[df['dayofyear']==i]['CARBON_INTENSITY']
        X_future = futuredf[futuredf['dayofyear']==i].index.dayofyear.values.reshape(-1,1)
        
        model.fit(X,y)
        futuredf.loc[futuredf['dayofyear'] == i, 'Prediction'] = model.predict(X_future)    
    return futuredf

# Function predicts daily average carbpn intensity for 2024 by fitting a linear regression to 
# corresponding days of the from previous years in the data set.
# Optimised
def futuredailyforecast2(df, model=LinearRegression()):
    df_daily = df.resample('D').mean()
    daily_dates_2024 = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Create features for future dates
    futuredf = pd.DataFrame(index=daily_dates_2024)
    futuredf = create_features(futuredf)

    # Precompute 'dayofyear' groups
    predictions = []
    for day in range(1, 367):
        historical = df_daily[df_daily['dayofyear'] == day]
        future = futuredf[futuredf['dayofyear'] == day]
        if not historical.empty and not future.empty:
            X = historical.index.dayofyear.values.reshape(-1, 1)
            y = historical['CARBON_INTENSITY']
            X_future = future.index.dayofyear.values.reshape(-1, 1)
            
            model.fit(X, y)
            predictions.append(pd.Series(model.predict(X_future), index=future.index))
    return pd.concat(predictions).to_frame(name='Prediction')

# Function means fractional deviation from daily average for each hour of the day in df
# and returns array of mean hourly deviations.
# Old
def HourlyAverageDeviation(df):
    df['DailyAverage'] = df['CARBON_INTENSITY'].resample('D').mean().reindex(df.index, method='ffill')
    df['Deviation'] = (df['CARBON_INTENSITY'] - df['DailyAverage'])/df['DailyAverage']
    df['HourlyAverageDeviation'] = df['Deviation'].groupby(df['hour']).transform('mean')
    return df['HourlyAverageDeviation'].unique()


# New: Optimised hourly average deviation function
def hourly_average_deviation(df):
    df_daily_avg = df['CARBON_INTENSITY'].resample('D').transform('mean')
    deviation = (df['CARBON_INTENSITY'] - df_daily_avg) / df_daily_avg
    hourly_deviation = deviation.groupby(df.index.hour).mean()

    
    # Output is now a series with hour as index so map can be used in futurehourlyforecast
    return hourly_deviation


# Function predicts hourly carbon intensity for 2024 by multiplying daily average carbon intensity by
# mean deviation for each hour of the day.

# Optimised: Use hourly_average_deviation function
def futurehourlyforecast(df):
    hourly_devs = hourly_average_deviation(df)
    daily_pred = futuredailyforecast1(df)
    
    hourly_dates_2024 = pd.date_range(start='2024-01-01 00:00', end='2024-12-31 23:00', freq='H')
    futuredfhourly = pd.DataFrame(index=hourly_dates_2024)
    futuredfhourly = create_features(futuredfhourly)
    
    futuredfhourly['HourlyAverageDeviation'] = futuredfhourly['hour'].map(hourly_devs)
    futuredfhourly['DailyAverage_Predicted'] = daily_pred['Prediction'].reindex(futuredfhourly.index, method='ffill')
    futuredfhourly['Prediction'] = futuredfhourly['DailyAverage_Predicted'] * (1 + futuredfhourly['HourlyAverageDeviation'])
    return futuredfhourly

# Function to combine Predicitions and Actual into singular dataframe, pass through wanted function
def PredvsActual(PathfromCurrentDir, StartDate, func):
    
    dff = ProcessData_CSVModule(PathfromCurrentDir)       # Gives Processed Data from Original Data.
    df = ShorterData(dff, start = StartDate)    # Indexes Data to the end of 2023.
    
    future = func(df) 
    actual_2024 = dff.loc[(dff.index >= '2024-01-01') & (dff.index < '2025-01-01'), ['CARBON_INTENSITY']]   # Actual data from 2024
    PredvsActual = future[['Prediction']] # Predicted data for 2024
    PredvsActual = PredvsActual.join(actual_2024, how='left') # Join the two dataframes
    PredvsActual.index.name = 'DATETIME'
    PredvsActual = PredvsActual[['CARBON_INTENSITY', 'Prediction']] # Swap columns for tidiness
    return PredvsActual

# Decorator function to record the runtime of another function.

def record_runtime(func):
    """
    Decorator function to record the runtime of another function.
    
    Parameters:
    func (function): The function whose runtime is to be recorded.
    
    Returns:
    function: The wrapped function with runtime recording.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        # print(f"Runtime of {func.__name__}: {runtime:.4f} seconds")
        return result, runtime
    return wrapper

@record_runtime
def HourlyPredictNA(PathfromCurrentDir, StartDate):
    Hourly = PredvsActual(PathfromCurrentDir, StartDate, futurehourlyforecast)
    mse = compute_mse(Hourly)
    
    # Write to CSV line by line
    with open('Step3/Files/HourlyPredictOpti11.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['DATETIME', 'CARBON_INTENSITY', 'Prediction'])
        # Write data line by line
        for index, row in Hourly.iterrows():
            writer.writerow([index, row['CARBON_INTENSITY'], row['Prediction']])
            
    return Hourly, mse

# Function to plot the comparison between Predictions and Actual Data for chosen DataFrame.

def PlotComparison(Comparisondf):
    ax = Comparisondf[['CARBON_INTENSITY']].plot(figsize=(15,5))
    Comparisondf[['Prediction']].plot(ax=ax,style='.')
    plt.legend(['True Data', 'Prediction'])
    ax.set_title('Raw Data and Prediction')
    plt.show()

# Function to calculate score of each method.
# Score is defined as the root mean squared error between the predicted and actual carbon intensity.

def compute_mse(Comparisondf):
    mse = np.sqrt(mean_squared_error(Comparisondf['CARBON_INTENSITY'], Comparisondf['Prediction']))
    return mse

###########################################################

path = '../../IntensityData0924.csv'
StartDate = '2020-01-01'

# Compute Forecasts

(HourlyResults, HourlyRT) = HourlyPredictNA(path, StartDate)

# Create DataFrame to store results

results_df = pd.DataFrame(columns=['Frequency', 'MSE', 'Runtime'])

# Append results to the DataFrame
results_df = pd.concat([results_df, pd.DataFrame([{'Frequency': 'Hourly', 'MSE': HourlyResults[1], 'Runtime': HourlyRT}])], ignore_index=True)
results_df.to_csv('Step3/Files/ResultsOpti11.csv')

# PlotComparison(HourlyResults[0])

print(results_df)