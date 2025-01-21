# Optimised Script
# Optimisation 1: Use of Vectorised Operations in futuredailyforcast.
# Optimisation 2: Use of NumPy arrays for computations in HourlyAverageDeviation.
# Optimisation 3: Reading and Writing CSV in bulk.
import threading
import requests


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import trapz as integrate

# Function to create features on dataframe for easier indexing

def create_features(df):
    
    df['hour']      = df.index.hour
    df['dayofyear'] = df.index.dayofyear
    
    return df


# OPTIMISED: Reading CSV in chunks
# Function to process data from csv file

def ProcessData_Batched(PathfromCurrentDir, chunksize=80000):
    '''
    Load data in chunks.
    This ensures only a manageable amount of data is loaded into memory at a time, 
    improving cache utilisation.
    '''
    chunks = []
    for chunk in pd.read_csv(PathfromCurrentDir, chunksize=chunksize):
        chunk['DATETIME'] = pd.to_datetime(chunk['DATETIME'])
        chunk = chunk.set_index('DATETIME')
        chunk = chunk.resample('h').mean()
        chunk.index = chunk.index.tz_localize(None)  # Ensure index timezone is removed.
        chunk = create_features(chunk)
        chunks.append(chunk)
    return pd.concat(chunks)

# Define data used in model

def ShorterData(df, start = '2000-01-01', end='2024-01-01'):
    df = df.loc[(df.index < end) & (df.index >= start)] # Filter data to only include data from start to 2023
    return df

# Function predicts daily average carbpn intensity for 2024 by fitting a linear regression to 
# corresponding days of from previous years in the data set.


def futuredailyforecast(df, model = LinearRegression()):
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



# Function means fractional deviation from daily average for each hour of the day in df
# and returns array of mean hourly deviations.

# Optimised hourly average deviation function
def hourly_average_deviation(df):
    df_daily_avg = df['CARBON_INTENSITY'].resample('D').transform('mean')
    deviation = (df['CARBON_INTENSITY'] - df_daily_avg) / df_daily_avg
    hourly_deviation = deviation.groupby(df.index.hour).mean()
    return hourly_deviation

# Function predicts hourly carbon intensity for 2024 by multiplying daily average carbon intensity by
# mean deviation for each hour of the day.

# Optimised
def futurehourlyforecast(df):
    hourly_devs = hourly_average_deviation(df)
    daily_pred = futuredailyforecast(df)
    
    hourly_dates_2024 = pd.date_range(start='2024-01-01 00:00', end='2024-12-31 23:00', freq='h')
    futuredfhourly = pd.DataFrame(index=hourly_dates_2024)
    futuredfhourly = create_features(futuredfhourly)
    
    futuredfhourly['HourlyAverageDeviation'] = futuredfhourly['hour'].map(hourly_devs)
    futuredfhourly['DailyAverage_Predicted'] = daily_pred['Prediction'].reindex(futuredfhourly.index, method='ffill')
    futuredfhourly['Prediction'] = futuredfhourly['DailyAverage_Predicted'] * (1 + futuredfhourly['HourlyAverageDeviation'])
    return futuredfhourly

# Function to combine Predicitions and Actual into singular dataframe, pass through wanted function
def PredvsActual(PathfromCurrentDir, StartDate, func):
    
    dff = ProcessData_Batched(PathfromCurrentDir)       # Gives Processed Data from Original Data.
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

# OPTIMISED: CSV file in written in bulk.

# Function to predict hourly carbon intensity for 2024 and calculate the mean squared error of the prediction.
@record_runtime
def HourlyPredictNA_Opti(PathfromCurrentDir, StartDate):
    Hourly = PredvsActual(PathfromCurrentDir, StartDate, futurehourlyforecast)
    mse = compute_mse(Hourly)
    Hourly.to_csv('Step3/NA_HourlyPredict2024_Opti.csv')
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

def run_code(path, startDate):

    # Compute Forecasts
    (HourlyResults,HourlyRT) = HourlyPredictNA_Opti(path, startDate)
    
    return HourlyResults,HourlyRT
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
                response = requests.get('http://10.19.150.63:8085/data.json', headers=headers)
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
    print(CI)
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
        run_results = run_monitoring(path, startDate,Postcode)
        results_df = pd.concat([results_df, run_results], ignore_index=True)
    
    #Append a row to results with mean values

    results_df = add_mean_row(results_df)
    
    # Write to CSV
    results_df.to_csv('Step3/ResultsOpti.csv')

    # Print the results
    print(results_df)
    
    return results_df

###########################################################

path = '../../IntensityData0924.csv'
StartDate = '2020-01-01'
Postcode = 'OX1'
num_runs = 20

run_multiple_times(path, StartDate, Postcode, num_runs)