import pandas as pd
import os
import sys
import argparse
from tqdm import tqdm
import torch
import time
import spacy

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from vlms import init_vlm
from utils import load_dataframe, concatenate_response, disable_huggingface_warnings, load_yolo_model, extract_object_name


args = argparse.ArgumentParser(description="Run VLM on a dataset for thorough evaluation. Outputs are saved in  'results' dir.")
# LLM
args.add_argument("--model", type=str, default="allenai/Molmo2-4B", help="Model name")
# Prompt and image paths
args.add_argument("--data_dir", type=str, default="../../../data", help="The dataset to run test on")
args.add_argument("--output_dir", type=str, default="../../results/", help="The dataset to run test on")
args.add_argument("--dataset", type=str, default="phd_counting", help="The dataset to run test on")
args.add_argument("--subset", type=str, default="base", help="pope: {gqa, aokvqa, coco}; phd: {base, icc, iac(sec), ccs}")
args.add_argument("--subsplit", type=str, default=None, help="Only used for POPE dataset. Choices are: {popular, adversarial, random}")
# LLM inference parameters
args.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. When set to 0.0 the model will do greedy decoding and ignore the other parameters. ")
args.add_argument("--max_new_tokens", type=int, default=1024, help="The maximum number of tokens to generate (does not include thinking).")
FLAGS = args.parse_args()




def main(args):
    ### Step 0: Initialize the vlm
    disable_huggingface_warnings()
    vlm = init_vlm(args.model, dtype=torch.float32, device=None)
    model_name_short = args.model.split("/")[-1]
    print(f"{args.model} VLM loaded.")
    nlp = spacy.load("en_core_web_sm")

    ### Step 1: Load input path&dataframe, and prepare output path&dataframe
    dataset = args.dataset
    print(f"Processing dataset: {dataset}")
    dataset_dir = args.data_dir
    if not args.subset:
        all_subsets = ["base", "icc", "iac", "ccs"]
    else:
        all_subsets = [args.subset]
    
    # Step 2.1: Loop by subsets
    for subset in all_subsets:
        df, col_prompt, col_image, image_dir = load_dataframe(dataset, dataset_dir, subset=subset, subsplit=args.subsplit)
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_data_dir = os.path.join(output_dir, dataset)
        os.makedirs(output_data_dir, exist_ok=True)
        output_model_dir = os.path.join(output_data_dir, f"{model_name_short}_pointing")
        os.makedirs(output_model_dir, exist_ok=True)
        if "phd" in args.dataset:
            output_file = os.path.join(output_model_dir, f"{subset}.csv")
        else:
            output_file = os.path.join(output_model_dir, f"{subset}_{args.subsplit}.csv")
        if os.path.exists(output_file):
            df_output = pd.read_csv(output_file)
        else:
            df_output = pd.DataFrame(columns=df.columns.tolist() + ["response"])

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if len(df_output) > 0:
                existing_pairs = set(zip(df_output["image_idx"], df_output["question_idx"]))
                if (row["image_idx"], row["question_idx"]) in existing_pairs:
                    print(f"Skipping idx {idx} as it is already processed.")
                    continue
            image_path = os.path.join(image_dir, row[col_image])
            prompt_dir = "../../prompts"
            object = extract_object_name(row[col_prompt], nlp)
            if "phd" in args.dataset:
                if row["task"] == "counting" or row["task"] == "positional":
                    prompt_file = "phd_counting.txt"
                else:
                    prompt_file = "phd_general.txt"
                prompt = f"Point to all the {object} in this image."
            else:
                prompt = open(os.path.join(prompt_dir, args.prompt_file), "r").read().strip()
                prompt = prompt.replace("{original_prompt}", row[col_prompt])
            if "phd" in args.dataset:
                if not "ccs" in subset and not os.path.exists(image_path):
                    image_path = image_path.replace("train", "val")
                    row[col_image] = row[col_image].replace("train", "val")

            response, num_generated_tokens, elapsed_time = vlm.infer_pointing_with_stats(
                    prompt=prompt, 
                    image=image_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            elapsed_time = round(elapsed_time, 2)
            df_output = concatenate_response(
                response, 
                row, 
                df_output, 
                col_image, 
                num_generated_tokens=num_generated_tokens,
                elapsed_time=elapsed_time,
            )
            with open(output_file, "w") as f:
                df_output.to_csv(f, index=False)



if __name__ == "__main__":
    main(FLAGS)
    