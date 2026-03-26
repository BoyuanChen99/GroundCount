import os
import json
import argparse
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm


def get_position(center, img_width, img_height):
    x, y = center
    vertical = "upper" if y < img_height / 3 else "middle" if y < 2 * img_height / 3 else "lower"
    horizontal = "left" if x < img_width / 3 else "center" if x < 2 * img_width / 3 else "right"
    return f"{vertical}-{horizontal}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--data_dir", type=str, default="../../../data/coco/annotations")
    args = parser.parse_args()

    ### Step 0: Prepare YOLO
    yolo_link = "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13x.pt"
    path_HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/huggingface_cache"))
    weights_path = os.path.join(path_HF_HOME, "yolov13x.pt")
    if not os.path.exists(weights_path):
        os.system(f"wget {yolo_link} -O {weights_path}")

    ### Step 1: Set input and output paths
    split = args.split
    data_dir = args.data_dir
    question_template_path = f"../../prompts/phd_counting.txt"
    question_template = open(question_template_path, "r").read()
    input_file = os.path.join(data_dir, f"count_{split}2017_nobbox.json")
    output_file = os.path.join(data_dir, f"count_{split}2017_yolo.json")
    input_data = json.load(open(input_file, "r"))

    ### Step 1.5: Load existing output or initialize
    if os.path.exists(output_file):
        output_data = json.load(open(output_file, "r"))
        processed_paths = set(output_data["image_paths"])
    else:
        output_data = {"image_paths": [], "question": [], "answer": []}
        processed_paths = set()

    ### Step 2: Loop through and run YOLO inference
    model = YOLO(weights_path)
    step_count = 0
    for i in tqdm(range(len(input_data["image_paths"])), desc="Processing images"):
        image_path = input_data["image_paths"][i]
        if image_path in processed_paths:
            continue
        answer = input_data["answer"][i]
        output_data["image_paths"].append(image_path)
        output_data["answer"].append(answer)
        results = model(image_path, conf=0.5, verbose=False)
        yolo_info = results[0].boxes.data.cpu().numpy().tolist()
        img_height, img_width = results[0].orig_shape
        names = results[0].names

        ### Step 3: Process YOLO info - compute centers and sort
        objects = []
        for box in yolo_info:
            x1, y1, x2, y2, conf, cls_id = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            label = names[int(cls_id)]
            pos = get_position((cx, cy), img_width, img_height)
            objects.append({"label": label, "pos": pos, "cx": cx, "cy": cy, "conf": conf})
        objects.sort(key=lambda o: (o["cx"], o["cy"]))

        ### Step 4: Build string with indexed labels
        class_counter = defaultdict(int)
        descriptions = []
        for obj in objects:
            class_counter[obj["label"]] += 1
            idx = class_counter[obj["label"]]
            descriptions.append(f"{obj['label']} {idx} {obj['pos']}: {obj['conf']:.2f}")
        yolo_string = "\n".join(descriptions) if descriptions else "no objects detected."

        ### Step 5: Create question with YOLO info
        original_question_full = input_data["question"][i]
        original_question = original_question_full.split("### Here is the question:")[-1].strip()
        original_description = original_question_full.split("### Here is the question:")[0].split("### Here is the description:")[1].strip()
        question = question_template.replace("{yolo_detections}", yolo_string)
        question = question.replace("{context}", original_description)
        question = question.replace("{question}", original_question)
        output_data["question"].append(question)

        ### Step 6: Write every 100 steps
        step_count += 1
        if step_count % 100 == 0:
            json.dump(output_data, open(output_file, "w"), indent=4)

    # Final save
    json.dump(output_data, open(output_file, "w"), indent=4)




if __name__ == "__main__":
    main()