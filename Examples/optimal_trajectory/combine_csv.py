import pandas as pd
import os



def combine(PATH, output_filename):
    """
    Combines all CSV files in a specified directory into a single CSV file with transposed data.

    This function reads all CSV files in the given directory, transposes the data by setting the 
    'Metric' column as the index and transposing the DataFrame, and then combines the transposed
    DataFrames into one consolidated DataFrame. The resulting DataFrame is saved as a new CSV file.

    Args:
        PATH (str): The path to the directory containing the CSV files to combine.
        output_filename (str): The name of the output CSV file where the combined data will be saved.

    Returns:
        None: The combined CSV file is saved directly to the specified directory.

    Example:
        ```
        # Directory containing the CSV files
        PATH = r"..\\Experiments\\Reactor2"
        
        # Name of the combined output file
        output_filename = "Reactor2_combined.csv"

        # Combine CSV files and save
        combine(PATH, output_filename)
        ```
    """
    
    # combine all csv files into one
    csv_files = [f for f in os.listdir(PATH) if f.endswith('.csv')]
    transposed_dfs = []

    for csv_file in csv_files:

        file_path = os.path.join(PATH, csv_file)
        df = pd.read_csv(file_path)
        df_transposed = df.set_index('Metric').transpose() # transpose data
        
        # update df
        transposed_dfs.append(df_transposed)

    combined_df = pd.concat(transposed_dfs, ignore_index=True)

    # save csv file
    output_file = os.path.join(PATH, output_filename)
    combined_df.to_csv(output_file, index=False)

    print(f"File saved to: {output_file}")
    