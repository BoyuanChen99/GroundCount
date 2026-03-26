"""
This script loads all categories and saves the mapping from singular form to plural form.
"""
import json
import os


def main():
    ### Step 0: Initialize input and output files
    split = "val2017"
    coco_dir = "../../../data/coco"
    annotations_dir = os.path.join(coco_dir, "annotations")
    instances_file = os.path.join(annotations_dir, f"instances_{split}.json")
    output_dir = "../../results/coco"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"categories.json")

    ### Step 1: Load instances file and extract relevant information
    with open(instances_file, "r") as f:
        instances_data = json.load(f)
    all_categories = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
    
    ### Step 2: Process the categories to create singular to plural mapping
    singular_to_plural = {}
    for cat in all_categories.values():
        singular_form = cat
        if cat == "person":
            plural_form = "people"
        elif cat == "bus":
            plural_form = "buses"
        elif cat == "bench":
            plural_form = "benches"
        elif cat == "sheep":
            plural_form = "sheep"
        elif cat == "skis": 
            plural_form = "skis"
        elif cat == "wine glass":
            plural_form = "wine glasses"
        elif cat == "knife":
            plural_form = "knives"
        elif cat == "couch":
            plural_form = "couches"
        elif cat == "scissors":
            plural_form = "scissors"
        elif cat == "toothbrush":
            plural_form = "toothbrushes"
        else:
            plural_form = cat + 's'  # Simple pluralization by adding 's'
        singular_to_plural[singular_form] = plural_form

    ### Step 3: Save the sorted annotations to the output file
    with open(output_file, "w") as f:
        json.dump(singular_to_plural, f, indent=4)


if __name__ == "__main__":
    main()