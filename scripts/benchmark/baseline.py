import pandas as pd
import os
import sys
import argparse
from tqdm import tqdm
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from vlms import init_vlm
from utils import load_dataframe, concatenate_response, disable_huggingface_warnings


args = argparse.ArgumentParser(description="Run VLM on a dataset for thorough evaluation. Outputs are saved in  'results' dir.")

# VLM Model: OpenGVLab/InternVL3_5-1B, YannQi/R-4B, Qwen/Qwen3-VL-2B-Thinking, AIDC-AI/Ovis2.5-2B
args.add_argument("--model", type=str, default="AIDC-AI/Ovis2.5-2B", help="Model name")
args.add_argument("--checkpoint", type=str, default=None, help="The full path of the checkpoint to load the model weights from")

# Dataset and subgroup
args.add_argument("--data_dir", type=str, default="../../../data", help="The dataset to run test on")
args.add_argument("--output_dir", type=str, default="../../results/phd_counting", help="The dataset to run test on")
args.add_argument("--dataset", type=str, default="phd_counting", help="The dataset to run test on")
args.add_argument("--subset", type=str, default="base", help="pope: {gqa, aokvqa, coco}; phd: {base, icc, iac(sec), ccs}")
args.add_argument("--subsplit", type=str, default=None, help="Only used for POPE dataset. Choices are: {popular, adversarial, random}")

# Hyperparameters
args.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. When set to 0.0 the model will do greedy decoding and ignore the other parameters. ")
args.add_argument("--max_new_tokens", type=int, default=1024, help="The maximum number of tokens to generate (does not include thinking).")
args.add_argument("--thinking_budget", type=int, default=1024, help="The maximum number of tokens can be used for thinking.")
FLAGS = args.parse_args()


def main(args):
    ### Step 0: Initialize the vlm
    disable_huggingface_warnings()
    vlm = init_vlm(args.model, dtype=torch.float32, device=None)
    model_name_short = args.model.split("/")[-1]
    print(f"{args.model} VLM loaded.")


    ### Step 1: Load the checkpoint. If it's a folder, get the first checkpoint file in ".pt" format
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        if os.path.isdir(checkpoint_path):
            ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
            if len(ckpt_files) == 0:
                raise ValueError(f"No .pt checkpoint files found in directory {checkpoint_path}")
            ckpt_files.sort()
            checkpoint_path = os.path.join(checkpoint_path, ckpt_files[0])
        print(f"Loading checkpoint from {checkpoint_path}...")
        vlm.load_checkpoint(checkpoint_path)
        print(f"Checkpoint loaded.")
        if "p3" in args.checkpoint.lower():
            model_name_short += f"_p3"
        elif "p2" in args.checkpoint.lower():
            model_name_short += f"_p2"
        else:
            model_name_short += f"_p1"


    ### Step 2: Load input path&dataframe, and prepare output path&dataframe
    dataset = args.dataset
    dataset_dir = args.data_dir
    
    # Step 2.1: Loop by subsets
    if not args.subset:
        all_subsets = ["base", "icc", "iac", "ccs"]
    else:
        all_subsets = [args.subset]
    for subset in all_subsets:
        df, col_prompt, col_image, image_dir = load_dataframe(dataset, dataset_dir, subset=subset, subsplit=args.subsplit)

        # Step 2.2: Set output file path
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_data_dir = os.path.join(output_dir, dataset)
        os.makedirs(output_data_dir, exist_ok=True)
        output_model_dir = os.path.join(output_data_dir, model_name_short)
        os.makedirs(output_model_dir, exist_ok=True)
        if "phd" in args.dataset:
            output_file = os.path.join(output_model_dir, f"{subset}.csv")
        else:
            output_file = os.path.join(output_model_dir, f"{subset}_{args.subsplit}.csv")

        # Step 2.3: Initialize output df
        if os.path.exists(output_file):
            df_output = pd.read_csv(output_file)
        else:
            df_output = pd.DataFrame(columns=df.columns.tolist() + ["response"])


        ### Step 3: Loop through the dataset and do inference
        for idx, row in tqdm(df.iterrows(), total=len(df)):

            ### Step 3.1: Continue if df_output already has a row matching the row's col_prompt and col_image
            if len(df_output) > 0:
                existing_pairs = set(zip(df_output["image_idx"], df_output["question_idx"]))
                if (row["image_idx"], row["question_idx"]) in existing_pairs:
                    print(f"Skipping idx {idx} as it is already processed.")
                    continue

            ### Step 3.2: Prepare the image path and the prompt
            image_path = os.path.join(image_dir, row[col_image])
            prompt_dir = "../../prompts"
            if "phd" in args.dataset:
                if row["task"] == "counting" or row["task"] == "positional":
                    prompt_file = "phd_counting.txt"
                else:
                    prompt_file = "phd_general.txt"
                prompt = open(os.path.join(prompt_dir, prompt_file), "r").read().strip()
                if type(row["context"]) is str:
                    prompt = prompt.replace("{context}", row["context"])
                else:
                    prompt = prompt.replace("{context}", "N/A")
                prompt = prompt.replace("{question}", row[col_prompt])
            else:
                prompt = open(os.path.join(prompt_dir, args.prompt_file), "r").read().strip()
                prompt = prompt.replace("{original_prompt}", row[col_prompt])

            ### Step 3.2.1: Special case for PhD, as it is using both train and val of coco...
            if "phd" in args.dataset:
                if not "ccs" in subset and not os.path.exists(image_path):
                    image_path = image_path.replace("train", "val")
                    row[col_image] = row[col_image].replace("train", "val")

            ### Step 3.3: Infer and process the response
            response, num_generated_tokens, elapsed_time = vlm.infer_with_stats(
                    prompt=prompt, 
                    image=image_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    enable_thinking=True,
                    thinking_budget=args.thinking_budget,
                )
            elapsed_time = round(elapsed_time, 2)

            ### Step 3.4: Concatenate the current row to the end of df_output
            df_output = concatenate_response(
                response, 
                row, 
                df_output, 
                col_image,
                num_generated_tokens=num_generated_tokens,
                elapsed_time=elapsed_time
            )

            ### Step 3.5: Write to the output path
            with open(output_file, "w") as f:
                df_output.to_csv(f, index=False)

    


if __name__ == "__main__":
    main(FLAGS)