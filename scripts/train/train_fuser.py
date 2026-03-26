"""
Train the Fusion VLM. 
"""

import json
import os
import sys
import argparse
import warnings
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic._internal._generate_schema')
warnings.filterwarnings('ignore', message='.*repr.*attribute.*Field.*')
warnings.filterwarnings('ignore', message='.*frozen.*attribute.*Field.*')

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.abspath(os.path.join(root_dir, 'src')))
from fusers import init_fuser
from all_fusers.base import FusedConfig
from utils import disable_huggingface_warnings, float_to_e_str, process_scienceqa, combine_datasets, process_thinkmode


args = argparse.ArgumentParser(description="Run VLM on a dataset for thorough evaluation. Outputs are saved in  'results' dir.")

# VLM Model: OpenGVLab/InternVL3-8B, YannQi/R-4B, Qwen/Qwen2.5-VL-3B-Instruct, AIDC-AI/Ovis2.5-2B
args.add_argument("--model", type=str, default="AIDC-AI/Ovis2.5-2B", help="Model name")
args.add_argument("--thinkmode", type=bool, default=False, help="Whether to enable thinkmode")
args.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint to load the model weights from")

# Dataset paths
args.add_argument("--checkpoint_dir", type=str, default="../../checkpoints", help="The path to the training dataset")
args.add_argument("--train_data_dir", type=str, default="../../../data", help="The path to the training dataset")
args.add_argument("--val_data_dir", type=str, default="../../../data", help="The path to the validation dataset")
args.add_argument("--template_path", type=str, default="../../prompts/phd_counting.txt", help="Path to the prompt template file")
args.add_argument("--shuffle", type=bool, default=False, help="Whether to shuffle the train dataset")
args.add_argument("--vizwiz_dir", type=str, default=None, help="Could mingle from other datasets for training.")
args.add_argument("--vqaonline_dir", type=str, default=None, help="Could mingle from other datasets for training.")

# Prompting with extra yolo info
args.add_argument("--provide_yolo_info", action="store_true", help="Whether to provide extra YOLO information in the prompts")

# Mixed dataset anatomy
args.add_argument("--num_vizwiz", type=int, default=1, help="Number of VizWiz samples to use")
args.add_argument("--num_scienceqa", type=int, default=1, help="Number of ScienceQA samples to use")
args.add_argument("--num_cococount", type=int, default=1, help="Number of COCO-Count samples to use")
args.add_argument("--num_vqaonline", type=int, default=1, help="Number of VQAOnline samples to use")
args.add_argument("--num_repetitions", type=int, default=3, help="Number of times to repeat the mixed dataset per epoch")

# Hyperparameters
args.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
args.add_argument("--batch_size", type=int, default=1, help="Training batch size")
args.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
args.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
args.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
args.add_argument("--warmup_type", type=str, default="cosine", help="Warmup type for learning rate scheduler")

# LR multipliers for each module of the fused model
args.add_argument("--language_only_after_steps", type=int, default=None, help="After this number of steps, only train the language decoder. Freeze all other modules.")
args.add_argument("--lr_multiplier_fusion", type=float, default=10.0, help="Learning rate multiplier for the fusion block")
args.add_argument("--lr_multiplier_vlm", type=float, default=0.1, help="Learning rate multiplier for the VLM backbone")
args.add_argument("--lr_multiplier_language", type=float, default=10.0, help="Learning rate multiplier for the language decoder")
args.add_argument("--lr_multiplier_yolo", type=float, default=100.0, help="Learning rate multiplier for the YOLO detector")
args.add_argument("--language_lr_decay", type=float, default=0.95, help="Layer-wise learning rate decay for the language decoder")

# Other training settings
args.add_argument("--save_tag", type=str, default="ovis2.5_fuse", help="Tag for saving the trained model checkpoints")
args.add_argument("--num_steps_per_log", type=int, default=100, help="Number of steps between printing losses")
args.add_argument("--num_steps_per_val", type=int, default=1000, help="Number of steps between evaluations")
args.add_argument("--num_steps_per_save", type=int, default=1000, help="Number of steps between saves")
args.add_argument("--val_size", type=int, default=0, help="Number of questions in the eval set to evaluate on during training")

# Hyperparameters in CFG
args.add_argument("--alignment_loss_weight", type=float, default=0.3, help="Weight for the alignment loss in the CFG")
args.add_argument("--alignment_loss_type", type=str, default="cosine", help="Type of alignment loss in the CFG")
args.add_argument("--alignment_temperature", type=float, default=0.07, help="Temperature for the alignment loss in the CFG")
args.add_argument("--no_train_vlm", type=bool, default=False, help="Whether to train the VLM or not")
args.add_argument("--no_train_yolo", type=bool, default=False, help="Whether to train the YOLO or not")
args.add_argument("--no_train_fusion", type=bool, default=False, help="Whether to train the fusion block or not")
args.add_argument("--no_train_language", type=bool, default=False, help="Whether to train the language decoder or not")
args.add_argument("--vlm_num_layers_to_train", type=int, default=None, help="Number of VLM layers to train")
args.add_argument("--language_num_layers_to_train", type=int, default=None, help="Number of language decoder layers to train")
FLAGS = args.parse_args()


def main(args):
    ### Step 0: Initialize the vlm and checkpoint dir
    disable_huggingface_warnings()
    print(f"Initializing the FUSER {args.model}...")
    cfg = FusedConfig(
        vlm_name=args.model,
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        train_vlm=not args.no_train_vlm,
        train_yolo=not args.no_train_yolo,
        train_fusion=not args.no_train_fusion,
        train_language=not args.no_train_language,
        dtype=torch.float32,
        alignment_loss_type=args.alignment_loss_type,
        alignment_loss_weight=args.alignment_loss_weight,
        alignment_temperature=args.alignment_temperature,
        vlm_num_layers_to_train=args.vlm_num_layers_to_train,
        language_num_layers_to_train=args.language_num_layers_to_train,
        language_lr_decay=args.language_lr_decay,
    )
    vlm = init_fuser(cfg)
    print(f"{args.model} VLM loaded.")
    model_name_short = args.model.split("/")[-1]
    vlm.print_anatomy()
    # Make checkpoint dir if not exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)


    ### Step 1: Load 4 different train datasets: VizWiz, ScienceQA, COCO-Count, VQAOnline
    # # Step 1.1: Load VizWiz from Multimodal-Fatima/VizWiz
    # print(f"Loading VizWiz dataset...")
    # vizwiz_path = os.path.join(args.vizwiz_dir, "processed.json")
    # data_vizwiz = json.load(open(vizwiz_path, 'r'))
    # print(f"Loaded {len(data_vizwiz['image_paths'])} training samples from VizWiz.")

    # # Step 1.2: Load ScienceQA from https://huggingface.co/datasets/derek-thomas/ScienceQA
    # print(f"Loading ScienceQA dataset...")
    # dataset_scienceqa = load_dataset("derek-thomas/ScienceQA")
    # train_data = dataset_scienceqa['train']
    # validation_data = dataset_scienceqa['validation']
    # test_data = dataset_scienceqa['test']
    # merged_dataset = concatenate_datasets([train_data, validation_data, test_data])
    # data_scienceqa = process_scienceqa(merged_dataset)
    # print(f"Loaded {len(data_scienceqa['image_paths'])} training samples from ScienceQA.")

    # Step 1.3: Load COCO-Count
    print(f"Loading COCO-Count dataset...")
    dataset = "coco/annotations"
    dataset_dir = args.train_data_dir
    cococount_path = os.path.join(dataset_dir, dataset, "count_train2017.json")
    with open(cococount_path, 'r') as f:
        data_cococount = json.load(f)
    print(f"Loaded {len(data_cococount['image_paths'])} training samples from {cococount_path}.")

    # # Step 1.4: Load VQAOnline
    # print(f"Loading VQAOnline dataset...")
    # vqaonline_path = os.path.join(args.vqaonline_dir, "processed.json")
    # data_vqaonline = json.load(open(vqaonline_path, 'r'))
    # print(f"Loaded {len(data_vqaonline['image_paths'])} training samples from VQAOnline.")

    # Step 1.5: Combine datasets (function implemented in utils)
    train_data = combine_datasets(
            datasets = [
                # data_vizwiz, 
                # data_scienceqa, 
                data_cococount, 
                # data_vqaonline,
            ], 
            numbers = [
                # args.num_vizwiz, 
                # args.num_scienceqa, 
                args.num_cococount, 
                # args.num_vqaonline
            ], 
            epochs = args.num_repetitions
        )
    
    # Step 1.6: Process for think mode
    if args.thinkmode:
        print("Processing training data for thinkmode...")
        train_data = process_thinkmode(train_data, model=args.model)
        print(f"Thinkmode processing completed. The first entry question is:\n{train_data['question'][0]}\n\nThe first entry answer is:\n{train_data['answer'][0]}\n\n")

    # Step 1.7: Remove YOLO info if not providing
    if not args.provide_yolo_info:
        print("Removing YOLO information from training data prompts...")
        for i in range(len(train_data['question'])):
            question = train_data['question'][i]
            # Assume YOLO info is enclosed in between "### YOLO Detections:\n" and "\n\n### Description" 
            start_tag = "### YOLO Detections:\n"
            end_tag = "\n\n### Description"
            start_idx = question.find(start_tag)
            end_idx = question.find(end_tag)
            if start_idx != -1 and end_idx != -1:
                # Replace the segment with "YOLO info not available."
                question_cleaned = question[:start_idx] + "YOLO info not available." + question[end_idx + len(end_tag):]
                train_data['question'][i] = question_cleaned.strip()
        print("YOLO information removed from training data prompts.")
    

    ### Step 2: Load val dataset: wrong baseline questions PhD "counting" and "positional" split
    baseline_results_dir = os.path.join(root_dir, "results", "phd", "baseline", "old", model_name_short)
    baseline_result_base = os.path.join(baseline_results_dir, "base.csv")
    df_baseline = pd.read_csv(baseline_result_base)
    df_baseline['correct'] = df_baseline['correct'].astype(int)
    df_baseline = df_baseline[df_baseline['correct'] == 0]
    df_baseline = df_baseline[df_baseline['task'].isin(['counting', 'positional'])]
    df_baseline = df_baseline[['image', 'question', 'label']]
    df_baseline = df_baseline.rename(columns={'image': 'image_paths'})
    df_baseline = df_baseline.iloc[12:].reset_index(drop=True) # Hard code this so it starts with the "giraffe" question
    # Process questions with template in a for loop
    with open(args.template_path, 'r') as f:
        template = f.read()
        for idx, row in df_baseline.iterrows():
            question_filled = template.replace("{question}", row['question'])
            df_baseline.at[idx, 'question'] = question_filled
    # Add data_dir to image paths
    df_baseline['image_paths'] = df_baseline['image_paths'].apply(lambda x: os.path.join(args.val_data_dir, 'coco', 'train2014' if 'train' in x else 'val2014', x))
    val_data = df_baseline.to_dict(orient='list')
    print(f"Loaded {len(val_data['image_paths'])} validation samples from {baseline_result_base}.")


    ### Step 3: Define run_name from args
    lr_in_e = float_to_e_str(args.learning_rate)
    if args.warmup_steps > 0:
        run_name = f"bs{args.batch_size}_lr{lr_in_e}_warmup{args.warmup_steps}"
    else:
        run_name = f"bs{args.batch_size}_lr{lr_in_e}"


    ### Step 4: Train
    print(f"Training begins...")
    vlm.train_loop(
        train_data = train_data,
        val_data = val_data,
        enable_thinking=args.thinkmode,
        val_size = args.val_size,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        val_every = args.num_steps_per_val,
        log_every = args.num_steps_per_log,
        save_every = args.num_steps_per_save,
        shuffle_train = args.shuffle,
        save_dir = args.checkpoint_dir,
        checkpoint_title = f"{args.save_tag}_{run_name}",
        lr_multipliers = {
            'fusion': args.lr_multiplier_fusion,
            'vit': args.lr_multiplier_vlm,
            'language': args.lr_multiplier_language,
            'yolo': args.lr_multiplier_yolo,
        },
        language_only_after_steps=args.language_only_after_steps,
    )




if __name__ == "__main__":
    main(FLAGS)