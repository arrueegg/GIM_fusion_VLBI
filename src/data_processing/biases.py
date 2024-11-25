import numpy as np
import pandas as pd
import os
import json
import datetime
from math import radians, sin, cos, sqrt, atan2
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.colors import PowerNorm


import warnings
warnings.filterwarnings("ignore")


def df_from_html(file):
    try:
        # Read HTML tables from the file
        dfs = pd.read_html(file)
        if dfs:
            # Select the first table from the list of tables
            df = dfs[0]
            # Remove rows where the second column is empty
            df = df[df.iloc[:, 1].notna()]
            # Reset the index of the DataFrame
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            print("No tables found in the HTML file.")
    except FileNotFoundError:
        print(f"The file '{file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_all_sessions(years):
    # Create a list of file paths for the master files
    masterfiles = [f"/scratch2/arrueegg/WP1/PyIono_Output/sessions/master{year}.html" for year in years]

    # Append additional file paths for VGOS master files
    years = np.arange(2017, 2020)
    [masterfiles.append(f"/scratch2/arrueegg/WP1/PyIono_Output/sessions/masterVGOS{year}.html") for year in years]

    # Create an empty DataFrame to store all the session data
    all_sessions = pd.DataFrame()

    # Iterate through each file path and read the data into a DataFrame
    for file in masterfiles:
        df = df_from_html(file)
        all_sessions = pd.concat([all_sessions, df], ignore_index=True)

    # Convert the 'Start' column to datetime and extract only the date
    all_sessions['Start'] = pd.to_datetime(all_sessions['Start']).dt.date

    return all_sessions


def load_txt(path):
    """
    Load a text file of session results and extract the 'vtecs' table into a DataFrame.

    Parameters:
        path (str): The path to the directory containing the text file.

    Returns:
        DataFrame: DataFrame for the 'vtecs' table.
    """

    # Read the text file
    with open(path, 'r') as file:
        data = file.read()

    # Split the text into tables based on empty lines
    tables = data.split('\n\n')

    # Define column names for the 'vtecs' table
    vtecs_columns = ['station', 'date', 'epoch', 'vgos_vtec', 'v_vtec_sigma', 'gims_vtec', 'madr_vtec']

    # Extract only the first table (vtecs)
    vtecs_table = tables[0]
    
    # Use StringIO to simulate a file for pandas
    table_data = StringIO(vtecs_table)
    
    # Read the table as a DataFrame with predefined column names
    df_vtecs = pd.read_csv(table_data, delim_whitespace=True, skipinitialspace=True, names=vtecs_columns, skiprows=1)

    return df_vtecs

def get_station_coords(df):
    # Load the station coordinates JSON file
    with open('/scratch2/arrueegg/WP1/PyIono_Output/Results/station_coords.json') as f:
        station_coords = json.load(f)

    station_coords = pd.DataFrame.from_dict(station_coords, orient='index').reset_index()
    station_coords.rename(columns={'index': 'station', 'Latitude': 'lat', 'Longitude': 'lon', 'Altitude': 'alt'}, inplace=True)

    df = df.merge(station_coords, on='station', how='left')

    return df

def get_data(res_path, years):

    data = pd.DataFrame()

    for tech in ["VLBI", "VGOS"]:
        for year in years:
            path = os.path.join(res_path, tech, str(year))
            if os.path.isdir(path):
                session_paths = [os.path.join(path, s) for s in os.listdir(path)]

                for session_path in session_paths:
                    txt_path = os.path.join(session_path, f'{session_path.split("/")[-1]}.txt')
                    if os.path.isfile(txt_path):
                        data = pd.concat([data, load_txt(txt_path)], ignore_index=True)

    columns = ['station', 'date', 'epoch', 'vgos_vtec', 'gims_vtec']
    data = data[columns]

    data = get_station_coords(data)

    return data

def plot_median_biases(plot_path, df):
    # Check if required columns are in the DataFrame
    required_columns = ['vgos_vtec', 'gims_vtec', 'lat', 'lon', 'station']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")
    
    # Drop rows with NaN values in the required columns
    df = df.dropna(subset=required_columns)
    
    # Calculating the bias between vgos_vtec and gims_vtec
    df['bias'] = df['vgos_vtec'] - df['gims_vtec']

    # Grouping by station to calculate the median bias and count observations per station
    station_bias = df.groupby('station').agg(
        median_bias=('bias', 'median'),
        observation_count=('bias', 'size'),
        latitude=('lat', 'first'),  # Assuming latitude is constant per station
        longitude=('lon', 'first')  # Assuming longitude is constant per station
    ).reset_index()

    # Sorting by the number of observations for the barplot
    station_bias_sorted = station_bias.sort_values(by='observation_count', ascending=False)

    # Color-coding the bars based on the number of observations
    observation_counts = station_bias_sorted['observation_count']
    colors = plt.cm.viridis(observation_counts / max(observation_counts))  # Normalizing to get color gradient


    # Plotting the barplot with stations sorted by observation count
    plt.figure(figsize=(12, 6))
    bars = plt.bar(station_bias_sorted['station'], station_bias_sorted['median_bias'], color=colors)
    plt.xlabel('Station')
    plt.ylabel('Median Bias (vgos_vtec - gims_vtec)')
    plt.title('Median Bias per Station Sorted by Number of Observations')
    plt.xticks(rotation=90)

    # Adding color bar with the correct mappable for the color scale, ensuring association with the current axis
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(observation_counts), vmax=max(observation_counts)))
    sm.set_array([])  # Dummy array for the color bar
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Number of Observations')  # Associating with current axis

    plt.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/median_bias_per_station_sorted.png")
    #plt.show()


    # Scatter plot of biases dependent on latitude, color-coded by number of observations
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(station_bias['latitude'], station_bias['median_bias'], 
                        c=station_bias['observation_count'], cmap='viridis', alpha=0.7)
    plt.xlabel('Latitude')
    plt.ylabel('Median Bias (vgos_vtec - gims_vtec)')
    plt.title('Biases per Station Dependent on Latitude')
    plt.colorbar(scatter, label='Number of Observations')

    # Fit linear, quadratic, and cubic trends weighted by the number of observations
    station_bias = station_bias.sort_values(by='latitude')
    x = station_bias['latitude']
    y = station_bias['median_bias']
    weights = station_bias['observation_count']

    # Generate a smooth x-axis for plotting the polynomial fits
    x_smooth = np.linspace(x.min(), x.max(), 500)  # 500 points for higher resolution

    # Linear fit
    linear_fit = np.polyfit(x, y, 1, w=weights)
    plt.plot(x_smooth, np.polyval(linear_fit, x_smooth), label='Linear fit')

    # Quadratic fit
    quadratic_fit = np.polyfit(x, y, 2, w=weights)
    plt.plot(x_smooth, np.polyval(quadratic_fit, x_smooth), label='Quadratic fit')

    # Cubic fit
    cubic_fit = np.polyfit(x, y, 3, w=weights)
    plt.plot(x_smooth, np.polyval(cubic_fit, x_smooth), label='Cubic fit')

    # Quartic fit
    quartic_fit = np.polyfit(x, y, 4, w=weights)
    plt.plot(x_smooth, np.polyval(quartic_fit, x_smooth), label='Quartic fit')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_path}/biases_per_station_dependent_on_latitude.png")
    # plt.show()

    # Scatter plot of biases dependent on longitude, color-coded by number of observations
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(station_bias['longitude'], station_bias['median_bias'], 
                        c=station_bias['observation_count'], cmap='viridis', alpha=0.7)
    plt.xlabel('Longitude')
    plt.ylabel('Median Bias (vgos_vtec - gims_vtec)')
    plt.title('Biases per Station Dependent on Longitude')
    plt.colorbar(scatter, label='Number of Observations')

    station_bias = station_bias.sort_values(by='longitude')
    x = station_bias['longitude']
    y = station_bias['median_bias']
    weights = station_bias['observation_count']

    # Generate a smooth x-axis for plotting the polynomial fits
    x_smooth = np.linspace(x.min(), x.max(), 500)

    # Linear fit
    linear_fit = np.polyfit(x, y, 1, w=weights)
    plt.plot(x_smooth, np.polyval(linear_fit, x_smooth), label='Linear fit')

    # Quadratic fit
    quadratic_fit = np.polyfit(x, y, 2, w=weights)
    plt.plot(x_smooth, np.polyval(quadratic_fit, x_smooth), label='Quadratic fit')

    # Cubic fit
    cubic_fit = np.polyfit(x, y, 3, w=weights)
    plt.plot(x_smooth, np.polyval(cubic_fit, x_smooth), label='Cubic fit')

    # Quartic fit
    quartic_fit = np.polyfit(x, y, 4, w=weights)
    plt.plot(x_smooth, np.polyval(quartic_fit, x_smooth), label='Quartic fit')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_path}/biases_per_station_dependent_on_longitude.png")
    # plt.show()


def plot_bias_timeseries(plot_path, df):
    # Check if required columns are in the DataFrame
    required_columns = ['vgos_vtec', 'gims_vtec', 'lat', 'lon', 'station', 'date', 'epoch']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")
    
    # Drop rows with NaN values in the required columns
    df = df.dropna(subset=required_columns)
    
    # Calculating the bias between vgos_vtec and gims_vtec
    df['bias'] = df['vgos_vtec'] - df['gims_vtec']
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["epoch"], format="%Y/%m/%d %H:%M:%S")
    df = df.sort_values(by='datetime', ascending=True)

    # Create the directory if it does not exist
    os.makedirs(plot_path, exist_ok=True)

    # Plotting timeseries of biases for each station
    for station in df['station'].unique():
        station_data = df[df['station'] == station]
        
        plt.figure(figsize=(10, 6))
        plt.plot(station_data['datetime'], station_data['bias'], label=station)
        plt.ylim(-40, 30)
        plt.xlabel('Time')
        plt.ylabel('Bias')
        plt.title(f'Bias Timeseries for Station {station}')
        plt.legend()
        plt.tight_layout()
        
        # Save each station's plot separately
        plt.savefig(f"{plot_path}/bias_timeseries_{station}.png")
        plt.close()

def get_num_sessions(df_sess_sta):
    grouped = df_sess_sta.groupby('vlbi_station')
    print(f'There are {len(grouped)} stations in the dataset.')
    print(f'There are {len(df_sess_sta)} sessions in the dataset.')
    for station in grouped:
        print(f'There are {len(station[1])} sessions for station {station[0]}.')

def main():

    res_path = '/scratch2/arrueegg/WP1/PyIono_Output/Results/' #Cont17  colocated
    plot_path = '/scratch2/arrueegg/WP2/GIM_fusion_VLBI/src/data_processing/plots/'
    years = np.arange(2013, 2025)
    plot = 1

    data = get_data(res_path=res_path, years=years)
    plot_bias_timeseries(plot_path, data)
    plot_median_biases(plot_path, data)

    data['bias'] = data['vgos_vtec'] - data['gims_vtec']
    print(f"Overall Bias: {data['bias'].median()}")

if __name__ == "__main__":
    main()