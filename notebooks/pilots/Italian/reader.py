import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import time
import schedule
import re
import ruptures as rpt

# File paths
source_file = '../../../data/pilots/CFRP production from Cetma Composites/800.xlsx'
destination_file = '../../../data/pilots/CFRP production from Cetma Composites/simulated_800.csv'

# Function to read source Excel data
def read_source_data(source_file):
    try:
        return pd.read_excel(source_file)
    except Exception as e:
        print(f"Error reading source file: {e}")
        return None

# Function to initialize destination CSV file
def initialize_destination_file(columns, destination_file):
    pd.DataFrame(columns=columns).to_csv(destination_file, index=False)

# Function to append data in chunks to destination CSV
def append_data_in_chunks(source_data, destination_file, start_idx, chunk_size=30):
    end_idx = start_idx + chunk_size
    if start_idx < len(source_data):
        chunk = source_data.iloc[start_idx:end_idx]
        chunk.to_csv(destination_file, mode='a', header=False, index=False)
        return end_idx
    else:
        return None

# Function to process the CSV file and update stats_df
def process_file():
    global stats_df, calculated_metrics
    try:
        # Read the CSV file
        try:
            df = pd.read_csv(destination_file)
        except pd.errors.EmptyDataError:
            print("CSV file is empty. Skipping processing.")
            return
        
        # Ensure the file contains the required columns
        if 'Time_sec' not in df.columns or 'PIECE TEMP [°C]' not in df.columns:
            print("CSV file does not contain the required columns.")
            return

        # Add smoothed temperature column
        alpha = 0.1  # Smoothing parameter (0 < alpha < 1)
        df['Smoothed PIECE TEMP [°C]'] = df['PIECE TEMP [°C]'].ewm(alpha=alpha, adjust=False).mean()
        data = df['Smoothed PIECE TEMP [°C]'].values
        time_vals = df['Time_sec'].values

        # Apply change point detection
        model = "rbf"  # Using RBF model for change point detection
        algo = rpt.Pelt(model=model, min_size=200, jump=10).fit(data)
        change_points = algo.predict(pen=50)

        # Filter change points to only include significant increases
        filtered_points = []
        for cp in change_points:
            if cp == 0 or cp >= len(data):
                continue
            if np.mean(data[cp:cp+10]) > np.mean(data[cp-10:cp]):
                filtered_points.append(cp)

        # Convert filtered points to actual time values
        filtered_times = time_vals[filtered_points]
        
        # Define points based on the dataset number
        dataset_match = re.search(r'simulated_(\d+)\.csv', destination_file)
        if dataset_match:
            dataset = int(dataset_match.group(1))
        else:
            raise ValueError("Dataset number not found in the file path.")

        if dataset in [800, 806, 808, 880, 882, 883, 884, 887, 890]:
            points = np.array([[400, 1000], [5000, 5800], [9600, 10400]])
            constant_points = np.array([[2000, 3000], [7000, 8000], [11200, 12200]])
            inf_points = np.array([1600, 3900, 6400, 9000, 11000])
        else:
            points = np.array([[400, 1000], [4200, 6000], [9600, 10600]])
            constant_points = np.array([[2000, 3500], [7000, 8500], [11200, 12200]])
            inf_points = np.array([1750, 3900, 6500, 9100, 11000])

        # Update stats_df with calculated metrics
        idx = 0  # Since we are processing one file, idx is 0
        stats_df.loc[idx, 'Dataset'] = dataset
        stats_df.loc[idx, 'Min value'] = np.min(data)
        stats_df.loc[idx, 'Max value'] = np.max(data)

        # Calculate metrics if sufficient data is available
        for i, (start, end) in enumerate(points):
            if len(data) > end // 10:
                slope = (data[end // 10] - data[start // 10]) / (end - start)
                stats_df.loc[idx, f'Slope {i + 1}'] = slope
                calculated_metrics.add(f'Slope {i + 1}')
                print(f"Slope {i + 1} calculated: {slope}")

        for i, (start, end) in enumerate(constant_points):
            if len(data) > end // 10:
                avg = np.mean(data[start // 10:end // 10])
                stats_df.loc[idx, f'Const {i + 1}'] = avg
                calculated_metrics.add(f'Const {i + 1}')
                print(f"Const {i + 1} calculated: {avg}")

        for i, inflection_point in enumerate(inf_points):
            if len(data) > inflection_point // 10:
                inflection = data[inflection_point // 10]
                stats_df.loc[idx, f'Inf {i + 1}'] = inflection
                calculated_metrics.add(f'Inf {i + 1}')
                print(f"Inf {i + 1} calculated: {inflection}")

        
        if dataset == 808:
            if pd.notna(stats_df.loc[idx, 'Slope 2']):
                if stats_df.loc[idx, 'Slope 2'] > 0.0173:
                    print("Slope 2 larger than usual")
                    if stats_df.loc[idx, 'Const 1'] > 73:
                        print("Slope 2 larger than usual and Const 1 larger than usual")

        if dataset == 839:
            if pd.notna(stats_df.loc[idx, 'Const 1']):
                if stats_df.loc[idx, 'Const 1'] > 71.8 :
                    print("Constant 1 larger than usual")
                    if stats_df.loc[idx, 'Inf 2'] > 72.7:
                        print("Inf 2 larger than usual and Const 1 larger than usual")
        if dataset == 852:
            if pd.notna(stats_df.loc[idx, 'Slope 1']):
                if stats_df.loc[idx, 'Slope 1'] > 0.041:
                    print("Slope 1 larger than usual")
        

        # Plot the graph with all the data up to this point
        print("LENGTH OF DATA ", len(data))
        if len(data) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(time_vals, data, label='Smoothed PIECE TEMP [°C]', color='blue')
            
            # Highlight sloped sections
            for i, (start, end) in enumerate(points):
                if len(data) > end // 10:
                    plt.plot(time_vals[start:end], data[start // 10:end // 10], 'o-', color='orange', label='Sloped section' if i == 0 else "")
            
            # Highlight constant sections
            for i, (start, end) in enumerate(constant_points):
                if len(data) > end // 10:
                    plt.plot(time_vals[start:end], data[start // 10:end // 10], 'o-', color='green', label='Constant section' if i == 0 else "")
            
            # Highlight inflection points
            for i, inflection_point in enumerate(inf_points):
                if len(data) > inflection_point // 10:
                    plt.axvline(x=time_vals[inflection_point], color='red', linestyle='--', label='Inflection point' if i == 0 else "")

            plt.xlabel('Time (sec)')
            plt.ylabel('PIECE TEMP [°C]')
            plt.title(f'Dataset: {dataset}')
            plt.legend()
            plt.tight_layout()  # Ensures labels are not cut off
            plt.show()


    except Exception as e:
        
        print(f"An error occurred: {e}")

def main():
    global stats_df, calculated_metrics, start_idx
    source_data = read_source_data(source_file)
    if source_data is None:
        return

    # Initialize an empty DataFrame and save it to the destination CSV file if not already initialized
    try:
        pd.read_csv(destination_file)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        initialize_destination_file(source_data.columns, destination_file)

    # Append data in chunks
    start_idx = append_data_in_chunks(source_data, destination_file, start_idx)

if __name__ == '__main__':
    start_idx = 0
    stats_df = pd.DataFrame(columns=['Dataset', 'Min value', 'Max value', 'Slope 1', 'Slope 2', 'Slope 3', 
                                     'Const 1', 'Const 2', 'Const 3', 'Inf 1', 'Inf 2', 'Inf 3', 'Inf 4', 'Inf 5'])
    calculated_metrics = set()

    # Schedule the main processing loop every minute
    schedule.every(0.1).minutes.do(main)
    schedule.every(0.2).minutes.do(process_file)

    # Execute the scheduled jobs
    while True:
        schedule.run_pending()
        time.sleep(1)
