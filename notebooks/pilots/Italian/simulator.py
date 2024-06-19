import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import openpyxl
import time

# File paths
source_file = '../../../data/pilots/CFRP production from Cetma Composites/800.xlsx'
destination_file = '../../../data/pilots/CFRP production from Cetma Composites/simulated_800.csv'

def read_source_data(source_file):
    # Read the entire source Excel file
    return pd.read_excel(source_file)

def initialize_destination_file(columns, destination_file):
    # Initialize an empty DataFrame and save it to the destination CSV file
    pd.DataFrame(columns=columns).to_csv(destination_file, index=False)

def append_data_in_chunks(source_data, destination_file, start_idx, chunk_size=30):
    end_idx = start_idx + chunk_size
    if start_idx < len(source_data):
        chunk = source_data.iloc[start_idx:end_idx]
        chunk.to_csv(destination_file, mode='a', header=False, index=False)
        return end_idx
    else:
        return None

def main():
    # Read the entire source Excel file
    source_data = read_source_data(source_file)

    # Initialize an empty DataFrame and save it to the destination CSV file
    initialize_destination_file(source_data.columns, destination_file)

    # Start index for reading data
    start_idx = 0

    # Append data in chunks every 5 seconds
    time_passed = 0
    while start_idx is not None:
        start_idx = append_data_in_chunks(source_data, destination_file, start_idx)
        if start_idx is not None:
            time.sleep(5)
            time_passed += 5
            print('Time passed: ', time_passed, 'sec')

if __name__ == '__main__':
    main()
