import pandas as pd
import os
import shutil
import subprocess
import json
import tempfile
import random
from pathlib import Path
from transformers import logging as hf_logging
from collections import defaultdict
from ultralytics import YOLO


def extract_object_name(question, nlp):
    doc = nlp(question)
    # Find noun chunks, exclude "image"
    for chunk in doc.noun_chunks:
        if chunk.root.text.lower() not in ('image', 'picture', 'photo'):
            # Skip chunks that are just numbers/determiners
            if chunk.root.pos_ in ('NOUN', 'PROPN'):
                return chunk.root.lemma_  # lemma for singular form
    # Fallback: find any noun
    for token in doc:
        if token.pos_ == 'NOUN' and token.text.lower() not in ('image', 'picture', 'photo'):
            return token.lemma_
    return None


def get_position(center, img_width, img_height):
    x, y = center
    vertical = "upper" if y < img_height / 3 else "middle" if y < 2 * img_height / 3 else "lower"
    horizontal = "left" if x < img_width / 3 else "center" if x < 2 * img_width / 3 else "right"
    return f"{vertical}-{horizontal}"

def get_yolo_string(
    yolo_model, 
    image_path, 
    conf_threshold=0.5,
    no_confidence=False,
    no_position=False,
):
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)
    yolo_info = results[0].boxes.data.cpu().numpy().tolist()
    img_height, img_width = results[0].orig_shape
    names = results[0].names
    objects = []
    for box in yolo_info:
        x1, y1, x2, y2, conf, cls_id = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        label = names[int(cls_id)]
        pos = get_position((cx, cy), img_width, img_height)
        objects.append({"label": label, "pos": pos, "cx": cx, "cy": cy, "conf": conf})
    objects.sort(key=lambda o: (o["cx"], o["cy"]))
    class_counter = defaultdict(int)
    descriptions = []
    for obj in objects:
        class_counter[obj["label"]] += 1
        idx = class_counter[obj["label"]]
        parts = [f"{obj['label']} {idx}"]
        if not no_position:
            parts.append(obj['pos'])
        desc = " ".join(parts)
        if not no_confidence:
            desc += f": {obj['conf']:.2f}"
        descriptions.append(desc)        
    return "\n".join(descriptions) if descriptions else "no objects detected."

def load_yolo_model(yolo_model_name):
    yolo_link = f"https://github.com/iMoonLab/yolov13/releases/download/yolov13/{yolo_model_name}.pt"
    path_HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/huggingface_cache"))
    weights_path = os.path.join(path_HF_HOME, f"{yolo_model_name}.pt")
    if not os.path.exists(weights_path):
        os.system(f"wget {yolo_link} -O {weights_path}")
    return YOLO(weights_path)



def float_to_e_str(x: float) -> str:
    s = f"{x:.15e}"               # e.g. "2.000000000000000e-05"
    mant, exp = s.split('e')
    mant = mant.rstrip('0').rstrip('.') or '0'   # "2"
    exp = str(int(exp))            # "-5" (drops leading zeros and +)
    return f"{mant}e{exp}"

def process_thinkmode(training_set, model="ovis"):
    THINK_TAGS = {
        "ovis": ("<think>", "</think>"),
        "qwen": ("<think>", "</think>"),
        "deepseek": ("<think>", "</think>"),
        "glm": ("<think>", "</think>"),
        "internvl": ("<|reasoning|>", "<|/reasoning|>"),
    }
    model_lower = model.lower()
    think_tag, end_think_tag = ("<think>", "</think>")  # default
    for key, tags in THINK_TAGS.items():
        if key in model_lower:
            think_tag, end_think_tag = tags
            break
    def add_think_tags(answer):
        if not isinstance(answer, str):
            return answer
        idx = answer.find("\n")
        if idx == -1:
            return f"{think_tag}{answer}{end_think_tag}\n"
        return f"{think_tag}{answer[:idx]}{end_think_tag}\n{answer[idx+1:]}"
    training_set["answer"] = [add_think_tags(a) for a in training_set["answer"]]
    return training_set

def disable_huggingface_warnings():
    hf_logging.set_verbosity_error()


def download_with_wget(url: str, out_path: str, rate_limit: str = "200k") -> bool:
    """
    Download `url` to `out_path` using wget with retries and timeouts.
    Returns True on success, False otherwise.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already present and non-empty
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] Already exists: {out_path}")
        return True

    # Ensure wget exists
    if shutil.which("wget") is None:
        print("[FAIL] `wget` not found on PATH.")
        return False

    # Download to a temp file first for atomic move on success
    with tempfile.NamedTemporaryFile(
        dir=str(out_path.parent), prefix=out_path.name + ".", suffix=".part", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "wget",
        "--tries=20",
        "--waitretry=5",
        "--retry-on-http-error=429,500,502,503,504",
        "--timeout=30",
        f"--limit-rate={rate_limit}",
        "-O", str(tmp_path),
        url,
    ]

    print(f"[START] Downloading {url} -> {out_path}")
    try:
        # Capture output so failures print nicely; keep return code semantics simple
        result = subprocess.run(
            cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(f"[DONE] wget finished with exit code {result.returncode}")
        if result.returncode != 0:
            # Show a compact error message
            err = (result.stderr or "").strip().splitlines()[-1:]  # last line if any
            if err:
                print(f"[FAIL] wget error: {err[0]}")
            else:
                print("[FAIL] wget failed with no stderr.")
            return False

        # Verify file exists and is non-empty
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            print("[FAIL] Download resulted in empty file.")
            return False

        # Atomic move into place
        os.replace(tmp_path, out_path)
        size_kb = out_path.stat().st_size / 1024
        print(f"[OK] Saved {out_path} ({size_kb:.1f} KB)")
        return True

    except FileNotFoundError:
        print("[FAIL] `wget` executable not found during run.")
        return False
    except Exception as e:
        print(f"[FAIL] Exception during download: {e}")
        return False
    finally:
        # Clean up temp file if it’s still around
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def load_dataframe(dataset_name, data_dir="../../../data", subset=None, subsplit=None):
    dataset_name = dataset_name.lower()
    if "haloquest" in dataset_name and "eval" in dataset_name:
        dataset_dir = os.path.join(data_dir, "haloquest")
        file_path = os.path.join(dataset_dir, f"haloquest-eval.csv")
        col_prompt = "question"
        col_url = "url"
        col_image = "image_name"
        image_dir = os.path.join(dataset_dir, "images")
        df = pd.read_csv(file_path)
        image_dir = os.path.join(dataset_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        for f in os.listdir(image_dir):
            f_path = os.path.join(image_dir, f)
            if os.path.isfile(f_path) and os.path.getsize(f_path) < 1024:
                os.remove(f_path)
        for idx, row in df.iterrows():
            image_name = row[col_image]
            image_url = row[col_url]
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                download_with_wget(image_url, image_path)
        cols_to_remove = [c for c in ["Index", "index"] if c in df.columns]
        df = df.drop(columns=cols_to_remove)
    elif "chair" in dataset_name:
        json_file = os.path.join(data_dir, "chair/chair_1994.json")
        image_dir = os.path.join(data_dir, "coco/val2014")
        # Create an empty dataframe
        df = pd.DataFrame()
        with open(json_file, "r") as f:
            for line in f:
                json_data = json.loads(line)
                if df.empty:
                    df = pd.json_normalize(json_data)
                else:
                    df = pd.concat([df, pd.json_normalize(json_data)], ignore_index=True)
        col_prompt = "text"
        col_image = "image"
    elif "pope" in dataset_name:
        questions = [json.loads(q) for q in open(f"{data_dir}/pope/{subset}/{subset}_pope_{subsplit}.json", "r")]
        df = pd.json_normalize(questions)
        col_prompt = "text"
        col_image = "image"
        image_dir = os.path.join(data_dir, "coco", "val2014")
    elif "phd_counting" in dataset_name:
        data_file = os.path.join(data_dir, "phd_counting", f"{subset}.csv")
        df = pd.read_csv(data_file)
        col_prompt = "question"
        col_image = "image"
        if "ccs" in subset:
            image_dir = os.path.join(data_dir, "coco", "CCS_images")
        else:
            image_dir = os.path.join(data_dir, "coco", "train2014")
    elif "phd" in dataset_name:
        data_file = os.path.join(data_dir, "phd", f"{subset}.csv")
        df = pd.read_csv(data_file)
        col_prompt = "question"
        col_image = "image"
        if "ccs" in subset:
            image_dir = os.path.join(data_dir, "coco", "CCS_images")
        else:
            image_dir = os.path.join(data_dir, "coco", "train2014")
    return df, col_prompt, col_image, image_dir




def concatenate_response(
    response: str,
    row: pd.Series, 
    df_output: pd.DataFrame, 
    col_image: str,
    num_generated_tokens: int = None,
    elapsed_time: float = None,
    **kwargs
):
    for c in ("idx_image", "idx_question"):
        if c not in df_output.columns:
            df_output[c] = pd.Series(dtype="Int64")
    NEVER_COLS = {"Index", "index", "Unnamed: 0"}
    row = row.drop(labels=[c for c in NEVER_COLS if c in row.index])
    df_output = df_output.drop(columns=[c for c in NEVER_COLS if c in df_output.columns], errors="ignore")
    row = row.copy()
    row["response"] = response.replace("\n", "\\n")
    row["time"] = elapsed_time
    row["num_tokens"] = num_generated_tokens
    for key, value in kwargs.items():
        row[key] = value
    if len(df_output) == 0:
        row["idx_image"] = 0
        row["idx_question"] = 0
    else:
        last = df_output.iloc[-1]
        same_image = (
            (col_image in row.index)
            and (col_image in df_output.columns)
            and pd.notna(last.get(col_image))
            and row[col_image] == last[col_image]
        )
        last_idx_image = -1 if pd.isna(last.get("idx_image")) else int(last["idx_image"])
        last_idx_question = -1 if pd.isna(last.get("idx_question")) else int(last["idx_question"])
        if same_image:
            row["idx_image"] = last_idx_image
            row["idx_question"] = last_idx_question + 1
        else:
            row["idx_image"] = last_idx_image + 1
            row["idx_question"] = 0
    
    df_output = pd.concat([df_output, row.to_frame().T], ignore_index=True)
    df_output = df_output.drop(columns=[c for c in NEVER_COLS if c in df_output.columns], errors="ignore")
    
    # Reorder: target_sequence first, then other columns (now runs for ALL cases)
    target_sequence = ["idx_image", "idx_question", "task", "correct", "time", "num_tokens", "question", "image", "context", "response", "label", "hitem", "subject", "gt"]
    ordered_cols = [col for col in target_sequence if col in df_output.columns]
    other_cols = [col for col in df_output.columns if col not in target_sequence]
    df_output = df_output[ordered_cols + other_cols]
    return df_output



def process_scienceqa(dataset):
    image_paths = []
    questions = []
    answers = []
    for item in dataset:
        image_paths.append(item['image'])
        question_text = item['question'] + '\n'
        choices = item['choices']
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            question_text += f"{letter}. {choice}\n"
        questions.append(question_text.strip())
        answer_idx = item['answer']
        answer_letter = chr(65 + answer_idx)
        answers.append(f"{answer_letter}. {choices[answer_idx]}")
    return {
        'image_paths': image_paths,
        'question': questions,
        'answer': answers
    }



def combine_datasets(datasets, numbers, epochs):
    combined = {
        'image_paths': [],
        'question': [],
        'answer': []
    }
    # Track starting position for each dataset
    current_positions = [0 for _ in datasets]
    
    for _ in range(epochs):
        for dataset_idx, (dataset, num_samples) in enumerate(zip(datasets, numbers)):
            num_samples = int(num_samples)
            total_size = len(dataset['image_paths'])
            
            # Get consecutive indices starting from current position
            start_idx = current_positions[dataset_idx]
            end_idx = min(start_idx + num_samples, total_size)
            
            # If we reach the end, wrap around
            if end_idx - start_idx < num_samples:
                # Take remaining from current position
                indices = list(range(start_idx, total_size))
                # Wrap around to beginning
                remaining = num_samples - len(indices)
                indices.extend(list(range(0, remaining)))
                current_positions[dataset_idx] = remaining
            else:
                indices = list(range(start_idx, end_idx))
                current_positions[dataset_idx] = end_idx % total_size
            
            # Filter out None image_paths
            valid_indices = [i for i in indices if dataset['image_paths'][i] is not None]
            
            # Append to combined dataset
            combined['image_paths'].extend([dataset['image_paths'][i] for i in valid_indices])
            combined['question'].extend([dataset['question'][i] for i in valid_indices])
            combined['answer'].extend([dataset['answer'][i] for i in valid_indices])
    
    return combined