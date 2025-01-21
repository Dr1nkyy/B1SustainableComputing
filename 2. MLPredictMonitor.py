"""
This file provides functions for predicting carbon intensiry for 2024 using XGBoost ML. 
The functions are as follows:

"""


# Relevant Imports needed for the ML.py file
import threading
import requests


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from numpy import trapz as integrate

# Functions for processing data and creating features.

# Processing Data from csv file
def ProcessData(PathfromCurretnDir):
    df = pd.read_csv(PathfromCurretnDir)    # Read csv file
    df = df.set_index('DATETIME')           # Set DATETIME as index.
    df.index = pd.to_datetime(df.index)     # Convert index from object to datetime for easier manipulation using pandas.
    df = df.resample('h').mean()            # Data now given in hourly intervals.
    df.index = df.index.tz_localize(None)   # Ensure index timezone is removed.
    df = create_features(df)
    df = add_lags(df)
    return df

def ShorterData(df, start = '2000-01-01', end='2024-01-01'):
    df = df.loc[(df.index < end) & (df.index >= start)] # Filter data to only include data from start to 2023
    return df
    

# Function for creating features from datetime index.

def create_features(df):
    
    """
    Creates time series features from datetime index.
    Datetime notaion given by:
    https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.dayofweek.html

    """
    
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

# Function for adding lags to the data.

def add_lags(df):
    target_map = df['CARBON_INTENSITY'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map) #52 week offset, same day of the week
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map) #104 week offset, same day of the week
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map) #156 week offset, same day of the week
    return df

# Function to create dataframe with future dates for prediction.
def create_future_df(df,startfuture = '2024-01-01', endfuture = '2025-01-01'):
    future = pd.date_range(startfuture,endfuture, freq='1h')
    future_df = pd.DataFrame(index=future)
    future_df['isFuture'] = True
    df['isFuture'] = False
    df_and_future = pd.concat([df, future_df])
    df_and_future = create_features(df_and_future)
    df_and_future = add_lags(df_and_future)

    future_with_features = df_and_future.query('isFuture').copy()
    
    return future_with_features

# Function to train model.

def TrainModel(df):
    Feature = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'lag1', 'lag2', 'lag3']
    Target = 'CARBON_INTENSITY'
    
    X_all = df[Feature]
    y_all = df[Target]

    reg = xgb.XGBRegressor( base_score=1, 
                            booster='gbtree',
                            n_estimators=2000,
                            early_stopping_rounds=100,
                            objective='reg:squarederror',
                            max_depth=5,
                            learning_rate=0.005) 
    reg.fit(X_all, y_all,
            eval_set=[(X_all, y_all)],
            verbose=100)
    return reg

# Function to combine Predicitions and Actual into singular dataframe

def combine_pred_actual(dff, future_with_features):
    actual_2024 = dff.loc[dff.index >= '2024-01-01', ['CARBON_INTENSITY']]   # Actual data from 2024
    PredvsActual = future_with_features.loc[future_with_features.index >= '2024-01-01', ['Prediction']] # Predicted data for 2024
    PredvsActual = PredvsActual.join(actual_2024, how='left') # Join the two dataframes
    PredvsActual.index.name = 'DATETIME'
    PredvsActual = PredvsActual[['CARBON_INTENSITY', 'Prediction']] # Swap columns for tidiness
    return PredvsActual

# Function to compare Predictions vs Actual Data
def PredvsActual(PathfromCurrentDir, StartData):
    dff = ProcessData(PathfromCurrentDir)       # Gives Processed Data from Original Data.
    df = ShorterData(dff, start = StartData)    # Indexes Data to the end of 2023.
    Feature = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'lag1', 'lag2', 'lag3']
    
    # Create future data
    future_with_features = create_future_df(df)

    # Predict the future
    reg = TrainModel(df)
    future_with_features['Prediction'] = reg.predict(future_with_features[Feature])

    # Combine Predicitions and Actual into singular dataframe
    PredvsActual = combine_pred_actual(dff, future_with_features)

    return PredvsActual

# Function to plot the comparison between Predictions and Actual Data for chosen DataFrame
def PlotComparison(Comparisondf):
    ax = Comparisondf[['CARBON_INTENSITY']].plot(figsize=(15,5))
    Comparisondf[['Prediction']].plot(ax=ax,style='.')
    plt.legend(['True Data', 'Prediction'])
    ax.set_title('Raw Data and Prediction')
    plt.show()
    
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
        return result, runtime
    return wrapper

# Function to calculate score of each method.
# Score is defined as the root mean squared error between the predicted and actual carbon intensity.

def compute_mse(Comparisondf):
    mse = np.sqrt(mean_squared_error(Comparisondf['CARBON_INTENSITY'], Comparisondf['Prediction']))
    return mse


# Function to get Hourly Forcast of Carbon Intensity for 2024
@record_runtime
def HourlyPredictML(PathfromCurrentDir, StartDate):
    Hourly = PredvsActual(PathfromCurrentDir, StartDate)
    mse = compute_mse(Hourly)
    Hourly.to_csv('Step2/ML_HourlyPredict2024.csv') # Save to csv
    return Hourly, mse

# Function to get Hourly Forcast of Carbon Intensity for 2024
@record_runtime
def DailyPredictML(PathfromCurrentDir, StartDate):
    Daily = PredvsActual(PathfromCurrentDir, StartDate)
    Daily = Daily.resample('D').mean()
    mse = compute_mse(Daily)
    Daily.to_csv('Step2/ML_DailyPredict2024.csv') # Save to csv
    return Daily, mse

# Function to get Monthly Forcast of Carbon Intensity for 2024
@record_runtime
def MonthlyPredictML(PathfromCurrentDir, StartDate):
    Monthly = PredvsActual(PathfromCurrentDir, StartDate)
    Monthly = Monthly.resample('M').mean()
    mse = compute_mse(Monthly)
    Monthly.to_csv('Step2/ML_MonthlyPredict2024.csv') # Save to csv
    return Monthly, mse

# Function to get Yearly Forcast of Carbon Intensity for 2024
@record_runtime
def AnnualPredictML(PathfromCurrentDir, StartDate):
    Annual = PredvsActual(PathfromCurrentDir, StartDate)
    Annual = Annual.resample('Y').mean()
    mse = compute_mse(Annual)
    Annual.to_csv('Step2/ML_AnnualPredict2024.csv') # Save to csv
    return Annual, mse

# Function to add mean column to results data.

def add_mean_row(df):
    # Calculate mean values for numerical columns
    mean_values = df.select_dtypes(include='number').mean()

    # Create a new row with mean values
    mean_row = pd.DataFrame(mean_values).transpose()

    mean_row.index = ['Mean']

    # Append the new row to the DataFrame
    results_df = pd.concat([df, mean_row], ignore_index=False)

    # Remove the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in results_df.columns:
        results_df = results_df.drop(columns=['Unnamed: 0'])
        
    return results_df

# Function to give current carbon intensity at a given postcode

def currentcarbonintensity(Postcode):
    headers = {
    'Accept': 'application/json'
    }
    
    r = requests.get(f'https://api.carbonintensity.org.uk/regional/postcode/{Postcode}', params={}, headers = headers)
    
    CIcurrent = r.json()
    intensity_value = CIcurrent['data'][0]['data'][0]['intensity']['forecast']
    
    return intensity_value

# Run the code

def run_code(path, StartDate):

    # Compute Forecasts
    (HourlyResults, HourlyRT) = HourlyPredictML(path, StartDate)
    # (DailyResults, DailyRT) = DailyPredictML(path, StartDate)
    # (MonthlyResults, MonthlyRT) = MonthlyPredictML(path, StartDate)
    # (AnnualResults, AnnualRT)= AnnualPredictML(path, StartDate)

    
    return HourlyResults, HourlyRT
###########################################################

# Function for polling power of PC every 100ms

def poll_power_consumption(path, StartDate, interval=0.1):
    headers = {
        'Accept': 'application/json'
    }
    
    Powers = []
    stop_event = threading.Event()
    
    def poll():
        while not stop_event.is_set():
            try:
                # Make a request to the API to get CPU and GPU power
                response = requests.get('http://10.0.130.85:8085/data.json', headers=headers)
                data = response.json()
            
                # Navigate through the JSON structure to retrieve the CPU Package Power value
                cpu_package_power = data['Children'][0]['Children'][1]['Children'][1]['Children'][0]['Value']
                gpu_power = data['Children'][0]['Children'][3]['Children'][0]['Children'][0]['Value']
            
                # Convert the string to a float
                cpu_package_power = float(cpu_package_power[:-1])
                gpu_power = float(gpu_power[:-1])
                total_power = cpu_package_power + gpu_power
            
                Powers.append(total_power)
            
            
            except Exception as e:
                print(f"Error: {e}")
        
            # Sleep for the specified interval
            time.sleep(interval)
        
    # Create a thread to run the poll_power_consumption function
    polling_thread = threading.Thread(target=poll)
    polling_thread.start()

    # Run forecast code
    HourlyResults, HourlyRT = run_code(path, StartDate)

    stop_event.set()
    polling_thread.join()

    return Powers , HourlyResults, HourlyRT

def run_monitoring(path, startDate, Postcode, interval=0.1):
    Powers , HourlyResults, HourlyRT = poll_power_consumption(path, startDate)
    
    Energy = integrate(Powers, dx=interval)
    CI = currentcarbonintensity(Postcode)
    CI_J = CI / (1000 * 3600)
    Carbon = Energy * CI_J * 1000000

    results_df = pd.DataFrame(columns=['Frequency', 'MSE', 'Runtime', 'Energy', 'Carbon'])

    # Append results to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([{'Frequency': 'Hourly', 
                                                       'MSE': HourlyResults[1], 
                                                       'Runtime': HourlyRT,
                                                       'Energy': Energy, 
                                                       'Carbon': Carbon}])], 
                                                      ignore_index=True)

    return results_df

def run_multiple_times(path, startDate,Postcode, num_runs=5):
    results_df = pd.DataFrame(columns=['Frequency', 'MSE', 'Runtime', 'Energy', 'Carbon'])
    
    for _ in range(num_runs):
        print(f"Run {_}")
        run_results = run_monitoring(path, startDate,Postcode)
        results_df = pd.concat([results_df, run_results], ignore_index=True)
    
    #Append a row to results with mean values

    results_df = add_mean_row(results_df)
    
    # Write to CSV
    results_df.to_csv('Step2/ScriptsMonitor/MLAnnualResults09.csv')

    # Print the results
    print(results_df)
    
    return results_df

###########################################################

path = '../../IntensityData0924.csv'
StartDate = '2009-01-01'
Postcode = 'OX1'
num_runs = 20

run_multiple_times(path, StartDate, Postcode, num_runs)







