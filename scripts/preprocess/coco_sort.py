"""
Sorts instances_[split].json to a sorted annotations-only dataset.
"""
import json
import os


def main():
    ### Step 0: Initialize input and output files
    split = "val2014"
    coco_dir = "../../../data/coco"
    annotations_dir = os.path.join(coco_dir, "annotations")
    instances_file = os.path.join(annotations_dir, f"instances_{split}.json")
    output_dir = "../../results/coco"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"sorted_{split}.json")

    ### Step 1: Load instances file and extract relevant information
    with open(instances_file, "r") as f:
        instances_data = json.load(f)
    all_categories = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
    all_annotations = instances_data["annotations"]
    print(f"Total annotations: {len(all_annotations)}")

    ### Step 2: Sort all_annotations by image_id(index of the image) and then by id(global entry). This will take a while.
    print(f"Sorting...")
    all_annotations.sort(key=lambda x: (x["image_id"], x["id"]))

    ### Step 3: Process the desired entries in order. "index" is wrong and we will fix it in the next step. 
    for i, ann in enumerate(all_annotations):
        # Step 3.1: Replace "category_id" with "category" and re-sequence keys
        category_name = all_categories[ann["category_id"]]
        # Step 3.2: Process bbox to get the center. [x,y,w,h] -> [center_x, center_y]
        bbox = ann.get("bbox", [])
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        # Step 3.3: Rebuild the dict to enforce key order and drop unwanted fields
        ordered_ann = {
            "image_id": ann["image_id"],
            "id": ann["id"],
            "category": category_name,
            "center": [center_x, center_y],
            "bbox": ann.get("bbox", []),  # keep bbox; default to [] if missing
        }
        all_annotations[i] = ordered_ann
    
    ### Step 4: Merge all entries sharing the same image_id into one entry with a list of annotations
    merged_annotations = []
    current_image_id = None
    current_entry = None
    for ann in all_annotations:
        # Step 4.1: Remove "image_id" from subsequent annotations to avoid redundancy
        ann_copy = ann.copy()
        ann_copy.pop("image_id", None)
        if ann["image_id"] != current_image_id:
            if current_entry is not None:
                merged_annotations.append(current_entry)
            current_image_id = ann["image_id"]
            current_entry = {
                "image_id": ann["image_id"],
                "annotations": [ann_copy]
            }
        else:
            current_entry["annotations"].append(ann_copy)

    ### Step 5: Sanity check
    split_dir = os.path.join(coco_dir, split)
    num_image_files = len(os.listdir(split_dir))
    print(f"Ground-truth num files: {num_image_files}, Merged entries: {len(merged_annotations)}")

    ### Step 6: Sort inside each entry's annotations, first by "category", second by "center_x", and add the "index" field to each sub-element
    for i, entry in enumerate(merged_annotations):
        entry_copy = entry["annotations"]
        entry_copy.sort(key=lambda x: (x["category"], x["center"][0], x["center"][1]))
        object_counts = {}
        for j, ann in enumerate(entry_copy):
            category = ann["category"]
            if category not in object_counts:
                object_counts[category] = 0
            entry_copy[j]["index"] = object_counts[category]+1
            object_counts[category] += 1
            # Re-order ann
            ordered_ann = {
                "category": ann["category"],
                "index": entry_copy[j]["index"],
                "center": ann["center"],
                "bbox": ann["bbox"],
                "id": ann["id"]
            }
            entry_copy[j] = ordered_ann
        merged_annotations[i]["annotations"] = entry_copy

    ### Step 7: Save the sorted annotations to the output file
    with open(output_file, "w") as f:
        json.dump(merged_annotations, f, indent=4)


if __name__ == "__main__":
    main()