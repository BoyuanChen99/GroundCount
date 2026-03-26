"""
Process COCO annotations to generate counting Q&A pairs.
Merges category processing, sorting, and Q&A generation into a single pipeline.
"""
import json
import os
import argparse
import random
from PIL import Image


def get_plural_form(category):
    """
    Get the plural form of a category name.
    """
    if category == "person":
        return "people"
    elif category == "bus":
        return "buses"
    elif category == "bench":
        return "benches"
    elif category == "sheep":
        return "sheep"
    elif category == "skis":
        return "skis"
    elif category == "wine glass":
        return "wine glasses"
    elif category == "knife":
        return "knives"
    elif category == "couch":
        return "couches"
    elif category == "scissors":
        return "scissors"
    elif category == "toothbrush":
        return "toothbrushes"
    else:
        return category + 's'


def get_position(center, img_width, img_height):
    """
    Determine the position of an object based on its center coordinates.
    Returns a string like "upper-left", "middle-center", "lower-right", etc.
    """
    x, y = center
    # Determine vertical position
    if y < img_height / 3:
        vertical = "upper"
    elif y < 2 * img_height / 3:
        vertical = "middle"
    else:
        vertical = "lower"
    # Determine horizontal position
    if x < img_width / 3:
        horizontal = "left"
    elif x < 2 * img_width / 3:
        horizontal = "center"
    else:
        horizontal = "right"
    # Always combine vertical and horizontal
    return f"{vertical}-{horizontal}"


def get_position_sort_key(position):
    """
    Get a sort key for position strings to ensure left-to-right, top-to-bottom ordering.
    Returns (col, row) where smaller values come first.
    """
    parts = position.split('-')
    vertical = parts[0]
    horizontal = parts[1]
    # Map vertical to row number (0=upper, 1=middle, 2=lower)
    row_map = {"upper": 0, "middle": 1, "lower": 2}
    row = row_map.get(vertical, 1)
    # Map horizontal to column number (0=left, 1=center, 2=right)
    col_map = {"left": 0, "center": 1, "right": 2}
    col = col_map.get(horizontal, 1)
    # Return (col, row) to sort left-to-right first, then top-to-bottom
    return (col, row)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process COCO annotations to counting Q&A textual pairs.")
    parser.add_argument("--split", type=str, default="train2017", help="Dataset split (e.g., val2017, train2017)")
    parser.add_argument("--coco_dir", type=str, default="../../../data/coco", help="Path to COCO dataset directory")
    args = parser.parse_args()
    split = args.split
    coco_dir = args.coco_dir
    
    ### Step 0: Initialize input and output files
    annotations_dir = os.path.join(coco_dir, "annotations")
    instances_file = os.path.join(annotations_dir, f"instances_{split}.json")
    output_file = os.path.join(annotations_dir, f"count_{split}.json")
    if not os.path.exists(instances_file):
        print(f"Error: Instances file {instances_file} not found.")
        return
    print(f"Processing {instances_file}...")
    # Read the phd counting template
    phd_counting_template = open("../../prompts/phd_counting.txt").read()
    
    ### Step 1: Load instances file and extract relevant information
    print("Loading instances file...")
    with open(instances_file, "r") as f:
        instances_data = json.load(f)
    all_categories = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
    all_annotations = instances_data["annotations"]
    print(f"Total annotations: {len(all_annotations)}")
    # Create singular to plural mapping
    singular_to_plural = {cat_name: get_plural_form(cat_name) for cat_name in all_categories.values()}
    
    ### Step 2: Sort all_annotations by image_id and then by id
    print("Sorting annotations...")
    all_annotations.sort(key=lambda x: (x["image_id"], x["id"]))
    
    ### Step 3: Process and transform annotations
    print("Processing annotations...")
    for i, ann in enumerate(all_annotations):
        # Replace "category_id" with "category" and process bbox
        category_name = all_categories[ann["category_id"]]
        bbox = ann.get("bbox", [])
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        # Rebuild the dict to enforce key order and drop unwanted fields
        ordered_ann = {
            "image_id": ann["image_id"],
            "id": ann["id"],
            "category": category_name,
            "center": [center_x, center_y],
            "bbox": ann.get("bbox", []),
        }
        all_annotations[i] = ordered_ann
    
    ### Step 4: Merge all entries sharing the same image_id
    print("Merging annotations by image_id...")
    merged_annotations = []
    current_image_id = None
    current_entry = None
    for ann in all_annotations:
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
    # Don't forget the last entry
    if current_entry is not None:
        merged_annotations.append(current_entry)
    print(f"Merged into {len(merged_annotations)} image entries")
    
    ### Step 5: Pre-process to get all categories and their counts
    print("Pre-processing categories and counts...")
    category_info_list = []
    split_dir = os.path.join(coco_dir, split)
    
    for entry in merged_annotations:
        image_id = entry["image_id"]
        image_path = os.path.join(split_dir, f"{str(image_id).zfill(12)}.jpg")
        image_path = os.path.abspath(image_path)
        
        # Load image to get actual dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {e}")
            continue
        
        # Compute positions for all annotations
        for ann in entry["annotations"]:
            center = ann["center"]
            ann["position"] = get_position(center, img_width, img_height)
        
        # Group annotations by category
        category_annotations = {}
        for ann in entry["annotations"]:
            category = ann["category"]
            if category not in category_annotations:
                category_annotations[category] = []
            category_annotations[category].append(ann)
        
        # Store info for each category
        for category, anns in category_annotations.items():
            category_info_list.append({
                'entry': entry,
                'category': category,
                'anns': anns,
                'count': len(anns),
                'image_path': image_path,
                'img_width': img_width,
                'img_height': img_height
            })
    
    total_pairs = len(category_info_list)
    print(f"Total Q&A pairs: {total_pairs}")
    
    # Separate into count=1 and count>1
    count_one_pairs = [info for info in category_info_list if info['count'] == 1]
    count_multi_pairs = [info for info in category_info_list if info['count'] > 1]
    
    print(f"Pairs with count=1: {len(count_one_pairs)}")
    print(f"Pairs with count>1: {len(count_multi_pairs)}")
    
    # Shuffle both lists
    random.shuffle(count_one_pairs)
    random.shuffle(count_multi_pairs)
    
    # Calculate global targets
    num_yes = total_pairs // 2
    num_no_plus = total_pairs // 4
    num_no_minus = total_pairs - num_yes - num_no_plus
    
    print(f"Target YES: {num_yes}")
    print(f"Target NO_PLUS: {num_no_plus}")
    print(f"Target NO_MINUS: {num_no_minus}")
    
    # Step 1: Assign NO_MINUS to count>1 pairs only
    for i in range(min(num_no_minus, len(count_multi_pairs))):
        count_multi_pairs[i]['q_type'] = 'NO_MINUS'
    
    # Step 2: Collect remaining unassigned pairs
    remaining_pairs = []
    for info in count_one_pairs:
        if 'q_type' not in info:
            remaining_pairs.append(info)
    for info in count_multi_pairs:
        if 'q_type' not in info:
            remaining_pairs.append(info)
    
    random.shuffle(remaining_pairs)
    
    # Step 3: Assign NO_PLUS to remaining pairs
    for i in range(num_no_plus):
        if i < len(remaining_pairs):
            remaining_pairs[i]['q_type'] = 'NO_PLUS'
    
    # Step 4: Assign YES to all remaining unassigned pairs
    for info in remaining_pairs:
        if 'q_type' not in info:
            info['q_type'] = 'YES'
    
    # Combine all pairs
    all_pairs = count_one_pairs + count_multi_pairs
    random.shuffle(all_pairs)
    
    ### Step 6: Generate Q&A pairs
    print("Generating Q&A pairs...")
    output_data = []
    
    # Statistics tracking
    stats_yes = 0
    stats_no_plus_one = 0
    stats_no_minus_one = 0
    
    for pair_info in all_pairs:
        entry = pair_info['entry']
        category = pair_info['category']
        anns = pair_info['anns']
        count = pair_info['count']
        image_path = pair_info['image_path']
        q_type = pair_info['q_type']
        
        category_plural = singular_to_plural.get(category, category + "s")
        
        # Sort by position (left-to-right, top-to-bottom) and assign indices
        anns.sort(key=lambda x: (get_position_sort_key(x["position"]), x["center"][0], x["center"][1]))
        for idx, ann in enumerate(anns):
            ann["index"] = idx + 1
            ordered_ann = {
                "category": ann["category"],
                "index": ann["index"],
                "center": ann["center"],
                "position": ann["position"],
                "bbox": ann["bbox"],
                "id": ann["id"]
            }
            anns[idx] = ordered_ann
        
        # Determine asked count based on type
        if q_type == 'YES':
            asked_count = count
            stats_yes += 1
        elif q_type == 'NO_PLUS':
            asked_count = count + 1
            stats_no_plus_one += 1
        elif q_type == 'NO_MINUS':
            asked_count = count - 1
            stats_no_minus_one += 1
        
        # Generate yes/no question
        if asked_count == 1:
            yes_no_question = f"Is there {asked_count} {category} in the image?"
        else:
            yes_no_question = f"Are there {asked_count} {category_plural} in the image?"
        
        # Wrap with template
        full_question = phd_counting_template.replace("{context}", "There is no extra context for this question.").replace("{question}", yes_no_question)
        
        # Generate the each-area counting part
        if count == 1:
            each_area_counting = f"{category} {anns[0]['index']} {anns[0]['position']}"
        else:
            position_parts = []
            i = 0
            while i < len(anns):
                pos = anns[i]["position"]
                indices_at_position = [anns[i]["index"]]
                
                j = i + 1
                while j < len(anns) and anns[j]["position"] == pos:
                    indices_at_position.append(anns[j]["index"])
                    j += 1
                
                if len(indices_at_position) == 1:
                    position_parts.append(f"{category} {indices_at_position[0]} {pos}")
                else:
                    indices_str = ", ".join(str(k) for k in indices_at_position)
                    position_parts.append(f"{category_plural} {indices_str} {pos}")
                
                i = j
            
            each_area_counting = "; ".join(position_parts)
        
        # Generate affirmative total summary
        if count < asked_count:
            if count == 1:
                total_summary = f"there is only {count} {category} in the image"
            else:
                total_summary = f"there are only {count} {category_plural} in the image"
        else:
            if count == 1:
                total_summary = f"there is {count} {category} in the image"
            else:
                total_summary = f"there are {count} {category_plural} in the image"
        
        # Determine if same or different
        is_correct = (asked_count == count)
        same_or_different = "the same as" if is_correct else "different from"
        answer_yes_no = "YES" if is_correct else "NO"
        
        # Asked item description
        if asked_count == 1:
            asked_item = f"{asked_count} {category}"
        else:
            asked_item = f"{asked_count} {category_plural}"
        
        # Construct the final answer
        answer = f"I see {each_area_counting}. So in total, {total_summary}. This is {same_or_different} what the question asked - {asked_item}. So the answer is {answer_yes_no}.\nMy final answer is: {answer_yes_no}"
        
        # Add to output
        output_data.append({
            "image_path": image_path,
            "question": full_question,
            "answer": answer
        })
    
    ### Step 7: Convert to final format
    final_output = {
        "image_paths": [entry["image_path"] for entry in output_data],
        "question": [entry["question"] for entry in output_data],
        "answer": [entry["answer"] for entry in output_data]
    }
    
    ### Step 8: Save output
    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=4)
    
    print(f"\nGenerated {len(output_data)} Q&A pairs and saved to {output_file}")
    print(f"\n=== Statistics ===")
    print(f"Total entries: {len(output_data)}")
    print(f"YES answers: {stats_yes} ({100*stats_yes/len(output_data):.2f}%)")
    print(f"NO answers (asked +1): {stats_no_plus_one} ({100*stats_no_plus_one/len(output_data):.2f}%)")
    print(f"NO answers (asked -1): {stats_no_minus_one} ({100*stats_no_minus_one/len(output_data):.2f}%)")
    print(f"Total NO answers: {stats_no_plus_one + stats_no_minus_one} ({100*(stats_no_plus_one + stats_no_minus_one)/len(output_data):.2f}%)")


if __name__ == "__main__":
    main()