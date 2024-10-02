import pandas as pd
import os

def combine(PATH, output_filename):
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

