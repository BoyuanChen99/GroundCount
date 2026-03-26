"""
This script is optional, only if you want to create a counting-only version of the PhD dataset. It should be applied after running phd.py in the same dir. 
"""



import pandas as pd
import os


def main():
    ### Step 0: Define paths
    data_dir = "/data_coco2014"
    input_dir = os.path.join(data_dir, "phd")
    output_dir = os.path.join(data_dir, "phd_counting")
    os.makedirs(output_dir, exist_ok=True)

    ### Step 1: Loop through each .json file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            ### Step 2: Load the .csv file into a DataFrame. Only keep rows where 'task' is 'counting'
            df = pd.read_csv(input_file)
            df_counting = df[df['task'] == 'counting']

            ### Step 3: Save the filtered DataFrame to the output directory
            df_counting.to_csv(output_file, index=False)
            
            ### Step 4: Print status, with number of rows processed
            print(f"Processed {len(df_counting)} rows for file {filename}.")





if __name__ == "__main__":
    main()
