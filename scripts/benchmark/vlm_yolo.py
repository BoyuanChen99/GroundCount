import pandas as pd
import os
import sys
import argparse
from tqdm import tqdm
import torch
import time

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from vlms import init_vlm
from utils import load_dataframe, concatenate_response, disable_huggingface_warnings, get_yolo_string, load_yolo_model


args = argparse.ArgumentParser(description="Run VLM on a dataset for thorough evaluation. Outputs are saved in  'results' dir.")
# VLM
args.add_argument("--model", type=str, default="AIDC-AI/Ovis2.5-2B", help="Model name")
args.add_argument("--checkpoint", type=str, default=None, help="The full path of the checkpoint to load the model weights from")
# YOLO info
args.add_argument("--yolo_model", type=str, default="yolov13x", help="Model name")
args.add_argument("--provide_yolo_info", type=bool, default=True, help="Whether to provide YOLO detection info in the prompt")
args.add_argument("--conf_threshold", type=float, default=0.5, help="The confidence threshold for YOLO detections")
args.add_argument("--no_confidence", default=False, action="store_true", help="Whether to provide YOLO detection confidence scores in the prompt")
args.add_argument("--no_position", default=False, action="store_true", help="Whether to provide YOLO detection position information in the prompt")
# Prompt and image paths
args.add_argument("--data_dir", type=str, default="../../../data", help="The dataset to run test on")
args.add_argument("--output_dir", type=str, default="../../results/", help="The dataset to run test on")
args.add_argument("--dataset", type=str, default="phd_counting", help="The dataset to run test on")
args.add_argument("--subset", type=str, default="base", help="pope: {gqa, aokvqa, coco}; phd: {base, icc, iac(sec), ccs}")
args.add_argument("--subsplit", type=str, default=None, help="Only used for POPE dataset. Choices are: {popular, adversarial, random}")
# VLM inference parameters
args.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. When set to 0.0 the model will do greedy decoding and ignore the other parameters. ")
args.add_argument("--max_new_tokens", type=int, default=1024, help="The maximum number of tokens to generate (does not include thinking).")
args.add_argument("--thinking_budget", type=int, default=1024, help="The maximum number of tokens can be used for thinking.")
args.add_argument("--enable_thinking", type=bool, default=True, help="Allow thinking or not.")
FLAGS = args.parse_args()




def main(args):
    ### Step 0: Initialize the vlm
    disable_huggingface_warnings()
    vlm = init_vlm(args.model, dtype=torch.float32, device=None)
    model_name_short = args.model.split("/")[-1]
    print(f"{args.model} VLM loaded.")
    yolo_model = None
    if args.provide_yolo_info:
        yolo_model = load_yolo_model(args.yolo_model)
        print(f"YOLO model {args.yolo_model} loaded for prompt augmentation.")

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
        if args.provide_yolo_info:
            suffix_parts = ["yolo"]
            if args.no_confidence:
                suffix_parts.append("noconf")
            if args.no_position:
                suffix_parts.append("nopos")
            if args.conf_threshold != 0.5:
                suffix_parts.append(f"ct{args.conf_threshold}")
            output_model_dir = os.path.join(output_data_dir, f"{model_name_short}_{'_'.join(suffix_parts)}")
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
            if "phd" in args.dataset:
                if not "ccs" in subset and not os.path.exists(image_path):
                    image_path = image_path.replace("train", "val")
                    row[col_image] = row[col_image].replace("train", "val")
            
            # Get YOLO detections and plug into prompt
            if args.provide_yolo_info and yolo_model is not None:
                yolo_start = time.perf_counter()
                yolo_string = get_yolo_string(
                                yolo_model, 
                                image_path,
                                conf_threshold=args.conf_threshold,
                                no_confidence=args.no_confidence,
                                no_position=args.no_position,
                            )
                yolo_time = time.perf_counter() - yolo_start
                prompt = prompt.replace("{yolo_detections}", yolo_string)
            else:
                prompt = prompt.replace("{yolo_detections}", "N/A")
                yolo_time = 0.0

            response, num_generated_tokens, elapsed_time = vlm.infer_with_stats(
                    prompt=prompt, 
                    image=image_path,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking,
                    thinking_budget=args.thinking_budget,
                )
            elapsed_time += yolo_time
            elapsed_time = round(elapsed_time, 2)
            df_output = concatenate_response(
                response, 
                row, 
                df_output, 
                col_image, 
                num_generated_tokens=num_generated_tokens,
                elapsed_time=elapsed_time,
                yolo_time=round(yolo_time, 2),
                yolo_output=yolo_string.replace("\n", "\\n"),
            )
            with open(output_file, "w") as f:
                df_output.to_csv(f, index=False)




if __name__ == "__main__":
    main(FLAGS)
    