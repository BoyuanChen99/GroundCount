import pandas as pd
import os
import re


# Number word to int mapping
NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20
}


def parse_number_from_question(question: str) -> int:
    """Extract the count number from question like 'Are there two dogs in the image?'"""
    question_lower = question.lower()
    # Try word match first
    for word, num in NUM_WORDS.items():
        if word in question_lower:
            return num
    # Try digit match
    match = re.search(r'\b(\d+)\b', question)
    if match:
        return int(match.group(1))
    return -1


def count_coordinates(response: str) -> int:
    """Count coordinate pairs separated by ';'"""
    if not response or pd.isna(response):
        return 0
    response = str(response).strip()
    if not response:
        return 0
    return len(response.split(';'))


def main():
    ### Step 0: Initialize the file paths
    model = "Qwen3-VL-2B-Thinking"
    results_dir = f"../../results/phd/baseline/{model}/"
    counting_only = True
    coordinate_mode = "pointing" in results_dir  # Set True for VLMs that output coordinates instead of YES/NO

    ### Step 0.1: Set the column names
    col_label = "label"
    col_response = "processed_response"
    col_task = "task"

    ### Step 1: Loop through the result files and judge in the following sequence: [base, iac, icc, ccs]
    sequence = ["base", "iac", "icc", "ccs"]
    files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    files_sorted = sorted(files, key=lambda x: next((i for i, s in enumerate(sequence) if s in x.lower()), len(sequence)))
    print(f"Results for {model}:")
    for file in files_sorted:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(results_dir, file))

            if coordinate_mode:
                # Special mode: judge by comparing parsed number from question vs coordinate count
                correctness = []
                for idx, row in df.iterrows():
                    expected_num = parse_number_from_question(str(row["question"]))
                    actual_num = count_coordinates(row["response"])
                    label_is_yes = str(row[col_label]).lower() == "yes"
                    # If label is YES, expected_num should match actual_num
                    # If label is NO, expected_num should NOT match actual_num
                    if label_is_yes:
                        correct = 1 if expected_num == actual_num else 0
                    else:
                        correct = 1 if expected_num != actual_num else 0
                    correctness.append(correct)
                df["correct"] = correctness
                # Set processed_response to coordinate count for reference
                df[col_response] = df["response"].apply(count_coordinates)
            else:
                # Step 1.1: Change the "UNSURE" entries in "processed_response" column to actual results by re-examining the "response" column
                for idx, row in df.iterrows():
                    original_response = str(row["response"])
                    if "YES" in original_response[-10:].upper():
                        df.at[idx, col_response] = "YES"
                    elif "NO" in original_response[-10:].upper() or " no " in original_response.lower() or " not " in original_response.lower() or ".no" in original_response.lower() or ".not" in original_response.lower():
                        df.at[idx, col_response] = "NO"
                    else:
                        df.at[idx, col_response] = "YES"
                
                # Step 1.2: Create a column called "correct" to the right. It is 1 if "processed_response".lower()=="label".lower() else 0
                correctness = []
                for _, row in df.iterrows():
                    if str(row[col_response]).lower() == str(row[col_label]).lower():
                        correctness.append(1)
                    else:
                        correctness.append(0)
                df["correct"] = correctness

            # Remove idx_image and idx_question columns if exist
            if "idx_image" in df.columns:
                df = df.drop(columns=["idx_image"])
            if "idx_question" in df.columns:
                df = df.drop(columns=["idx_question"])

            # Step 1.2.1: Filter to counting tasks only if counting_only is True
            if counting_only:
                df_stats = df[df[col_task] == "counting"].copy()
            else:
                df_stats = df

            # Step 1.3: Get the correctness rate per different value in "task" column
            task_sequence = ["object", "attribute", "positional", "counting", "sentiment"]
            task_groups = df_stats.groupby(col_task)
            correctness_rates = {}
            for task in task_sequence:
                if task in task_groups.groups:
                    correctness_rates[task] = task_groups.get_group(task)["correct"].mean()
            
            # Step 1.4: Print the correctness rates
            print(f"\n{file}:")
            for task, rate in correctness_rates.items():
                print(f"{task}: {100*rate:.1f}%")
            
            # Step 1.5: Print average time
            if "time" in df_stats.columns:
                # Get mean in float
                df_stats["time"] = df_stats["time"].astype(float)
                avg_time = df_stats["time"].mean()
                # Keep 1 decimal place
                print(f"avg time: {avg_time:.1f}s")

            # Step 1.6: Re-order the columns in sequence
            target_sequence = ["image_idx", "question_idx", col_task, "correct", "time", "num_tokens", "question", "context", "image", "response", col_response, col_label, "hitem", "subject", "gt"]
            df = df[[col for col in target_sequence if col in df.columns]]

            # Step 1.7: Save the updated dataframe back to the csv file
            df.to_csv(os.path.join(results_dir, file), index=False)
    
    

if __name__ == "__main__":
    main()