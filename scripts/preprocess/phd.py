"""
Reshapes question pairs from JSON into standardized CSV files for multiple context settings.
"""



import json
import pandas as pd


def main():
    all_subsets = ["base", "icc", "iac", "ccs"]
    for subset in all_subsets:
        ### Step 0: Initialize input and output files
        input_file = f"../../../data/phd/data.json"
        output_file = f"../../../data/phd/{subset}.csv"

        ### Step 1: Load the json file as a normal list or dict
        with open(input_file, "r") as f:
            data = json.load(f)

        ### Step 2: Initialize a dataframe with hand-crafted column titles
        df = pd.DataFrame(columns=["Index", "image_idx", "question_idx", "task", "question", "label", "context", "image", "hitem", "subject", "gt"])

        ### Step 3: Loop through each element in data, and append to df
        i = 0
        for _, item in enumerate(data):
            # Context
            if "base" in subset:
                if "context" not in item.keys():
                    continue
                context = ""
            elif "icc" in subset:
                if "context" not in item.keys():
                    continue
                elif "icc" not in item.get("context", ""):
                    continue
                else:
                    context = item.get("context").get("icc")
            elif "iac" in subset:
                if "context" not in item.keys():
                    continue
                elif "iac" not in item.get("context", ""):
                    continue
                else:
                    context = item.get("context").get("iac")
            elif "ccs" in subset:
                if "ccs_description" not in item.keys():
                    continue
                else:
                    context = item.get("ccs_description")

            # Index
            if len(df) == 0:
                image_idx = 1
                question_idx = 1
            else:
                if item["image_id"] in df["image"].values:
                    image_idx = df.loc[df["image"] == item["image_id"], "image_idx"].values[0]
                    question_idx = df.iloc[-1]["question_idx"] + 1
                else:
                    image_idx = df["image_idx"].max() + 1
                    question_idx = 1

            # Append the row
            rows = []
            image_id = item.get("image_id") if not "ccs" in subset else item.get("image_id")
            if "ccs" in subset:
                image_id = image_id + ".png"
            else:
                image_id = "COCO_train2014_" + image_id + ".jpg"
            rows.append({
                "Index": 2*i+1,
                "image_idx": image_idx,
                "question_idx": question_idx,
                "task": item.get("task"),
                "question": item.get("yes_question"),
                "label": "yes",
                "context": context,
                "image": image_id,
                "hitem": item.get("hitem"),
                "subject": item.get("subject"),
                "gt": item.get("gt"),
            })
            rows.append({
                "Index": 2*i+2,
                "image_idx": image_idx,
                "question_idx": question_idx + 1,
                "task": item.get("task"),
                "question": item.get("no_question"),
                "label": "no",
                "context": context,
                "image": image_id,
                "hitem": item.get("hitem"),
                "subject": item.get("subject"),
                "gt": item.get("gt"),
            })
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            i += 1


        ### Step 4: Write df to CSV
        df.to_csv(output_file, index=False)
        print(f"Finished processing subset: {subset}")


if __name__ == "__main__":
    main()