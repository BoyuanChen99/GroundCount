"""
Run it after inference to visually inspect where the model got things right or wrong on the PhD benchmark.
"""

import argparse
import os
from PIL import Image
from gui import ImageViewer
import pandas as pd
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from utils import process_response


def parse_args():
    argparser = argparse.ArgumentParser(description="Run inference with the model.")
    ### VLM and dataset
    argparser.add_argument("--model_name", type=str, default="Ovis2.5-2B_p3", help="Name of the model to use for inference.")
    argparser.add_argument("--benchmark", type=str, default="phd", help="The benchmark of the experiment. eg: pope, chair, etc.")
    ### Hoper parameters
    argparser.add_argument("--temperature", type=float, default="0.0", help="The temperature used for inference.")
    argparser.add_argument("--subset", type=str, default="base", help="Subset of the POPE dataset to use. The three options are: coco, aokvqa, and gqa.")
    ### Fixed directories
    argparser.add_argument("--data_folder", type=str, default="../../../data", help="Path to the data folder containing POPE dataset.")
    argparser.add_argument("--subfolder", type=str, default="train2014", help="Only relevant for coco subset.")
    argparser.add_argument("--output_folder", type=str, default="../../results", help="Folder to save the results.")
    ### Save images on clicking "next"
    argparser.add_argument("--save_mode", type=bool, default=True, help="Whether to save the images when clicking 'next'.")
    return argparser.parse_args()



def main(args):
    ### Step 0.1: Define, and read the output file
    if args.temperature <= 0:
        temperature = "greedy"
    else:
        temperature = str(args.temperature)
        if temperature.endswith(".0"):
            temperature = "temp"+temperature[:-2]
    # Load output file
    output_file_csv = os.path.join(args.output_folder, args.benchmark, args.model_name, temperature, f"{args.subset}_{args.model_name}.csv")
    with open(output_file_csv, "r") as f:
        df = pd.read_csv(f)
        answers = df.to_dict(orient="records")
        results = df.to_dict(orient="records")
    
    ### Step 0.2: Get the necessary column names right
    col_prompt = "question"
    col_question_id = "idx_question"
    col_response = "response"
    col_image = "image"
    col_label = "label"
    col_processed = "processed_response"
    col_task = "task"

    ### Step 0.3: Calculate correctness rate
    correct = sum(1 for r in results if r[col_processed].strip().lower() == r[col_label].strip().lower())
    total = len(results)
    correctness_rate = correct / total if total > 0 else 0
    print(f"Correctness rate: {correctness_rate:.2%}")

    ### Step 1: Loop through the images in the GUI
    images_info = []
    for result in results:
        question_id = result[col_question_id]
        prompt = result[col_prompt]
        model_output = result[col_response].lower().strip()
        model_id = result.get("model_id", args.model_name)
        image_file = result[col_image]
        answer = next(
            (a[col_label] for a in answers 
            if a[col_question_id] == question_id and a[col_prompt] == prompt and a[col_image] == image_file),
            "NOT FOUND"
        )

        ### Step 1.1: Process model response on-site
        model_output = process_response(model_output, args.benchmark)

        ### Step 1.2: Get the image
        if "train" in image_file:
            image_dir = os.path.join(args.data_folder, "coco", "train2014")
        elif "val" in image_file:
            image_dir = os.path.join(args.data_folder, "coco", "val2014")
        elif "ccs" in args.subset:
            image_dir = os.path.join(args.data_folder, "coco", "CCS_images")
        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        images_info.append({
            "question_id": question_id,
            "prompt": prompt,
            "model_output": model_output,
            "model_id": model_id,
            "image_file": image_file,
            "answer": answer,
            "image": image,
            "original_response": result[col_response],
            "task": result.get(col_task, "N/A"),
        })

    if images_info:
        app = ImageViewer(images_info, title=f"Phd {args.subset} {args.model_name}, t={args.temperature}", save_mode=True)
        app.mainloop()
    else:
        print("No images to display.")



if __name__ == "__main__":
    args = parse_args()
    main(args)
