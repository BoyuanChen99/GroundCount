import json
import os
from tqdm import tqdm

def main():
    ### Step 0: Define input and output file paths
    split = "train"
    data_dir = "../../../data/coco/annotations"
    input_file_name = f"count_{split}2017.json"
    output_file_name = f"count_{split}2017_hpc.json"

    ### Step 1: Load input data
    input_file = os.path.join(data_dir, input_file_name)
    input_data = json.load(open(input_file, "r"))

    ### Step 2: Define the current path and the target HPC path
    current_base_path = "/home/Desktop/data/"
    target_hpc_base_path = "/data_coco2017/"

    ### Step 3: Process each image path and change to HPC path
    for i in tqdm(range(len(input_data["image_paths"])), desc="Processing paths"):
        image_path = input_data["image_paths"][i]
        if image_path.startswith(current_base_path):
            relative_path = os.path.relpath(image_path, current_base_path)
            new_image_path = os.path.join(target_hpc_base_path, relative_path)
            input_data["image_paths"][i] = new_image_path

    ### Step 4: Save the modified data to the output file
    output_file = os.path.join(data_dir, output_file_name)
    with open(output_file, "w") as f:
        json.dump(input_data, f, indent=4)




if __name__ == "__main__":
    main()
