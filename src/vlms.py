# src/vlms.py

# Torch and transformers
import torch

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*device_map keys do not match.*")  # Ovis2.5
warnings.filterwarnings("ignore", message=".*slow image processor.*") # Image processor for all models
warnings.filterwarnings("ignore", category=FutureWarning)

# SOTA models for experiments, in release order
from all_vlms.molmo2 import Molmo2
from all_vlms.qwen3vl import Qwen3
from all_vlms.internvl3d5 import InternVL3d5
from all_vlms.r4b import R
from all_vlms.ovis2d5 import Ovis2d5

# Older models
from all_vlms.qwen2d5vl import Qwen2d5
from all_vlms.internvl3 import InternVL3
from all_vlms.ovis2 import Ovis2



### ========== Model Initializer ========== ###
def init_vlm(model_name, dtype=torch.float32, device=None):
    if "molmo2" in model_name.lower():
        return Molmo2(model=model_name, dtype=dtype, device=device)
    elif "internvl3_5" in model_name.lower():
        return InternVL3d5(model=model_name, dtype=dtype, device=device)
    elif "internvl3" in model_name.lower():
        return InternVL3(model=model_name, dtype=dtype, device=device)
    elif "r-4b" in model_name.lower():
        return R(model=model_name, dtype=dtype, device=device)
    elif "ovis2.5" in model_name.lower():
        return Ovis2d5(model=model_name, dtype=dtype, device=device)
    elif "ovis2" in model_name.lower():
        return Ovis2(model=model_name, dtype=dtype, device=device)
    elif "qwen3" in model_name.lower():
        return Qwen3(model=model_name, dtype=dtype, device=device)
    elif "qwen2.5" in model_name.lower():
        return Qwen2d5(model=model_name, dtype=dtype, device=device)
    raise ValueError(f"Unknown model name: {model_name}")



### ========== Train tests ==========
def main():
    ### Step 0: Initialize VLM
    vlm = Ovis2d5(model="AIDC-AI/Ovis2.5-2B", dtype=torch.float32)

    ### Step 1: Test inference
    outs = vlm.infer(
        image_path="/home/bc3194/Desktop/data/coco/val2017/000000039769.jpg",
        prompt="Describe the image in detail.",
        enable_thinking=False,
    )
    print(outs)
    input("Infer just done. Proceed to training?")

    # 2) Prepare data with just one sample for demonstration
    big_dict = {
        "image_paths": [
            "/home/bc3194/Desktop/data/coco/train2017/000000000009.jpg",
            ["/home/bc3194/Desktop/data/coco/train2017/000000000009.jpg", "/home/bc3194/Desktop/data/coco/train2017/000000000025.jpg"],
        ],
        "question": [
            "What is in the picture? ",
            "List the main objects visible in the second image. ",
        ],
        "answer": [
            "A lunch tray with bread, broccoli, and other food items.",
            "Two people riding bicycles on a city street.",
        ],
    }

    # 3) Train
    vlm.train_loop(
        data=big_dict,
        num_epochs=2,
        batch_size=1,
        lr=5e-5,
        save_every_epoch=False,
        save_tag="ovis2.5",
        save_fn=lambda tag: vlm.model.save_pretrained(f"../checkpoints/{tag}"),
    )
    


if __name__ == "__main__":
    main()