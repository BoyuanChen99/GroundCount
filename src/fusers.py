import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image

# Fusion configs
from all_fusers.base import FusedConfig

# Models
from all_fusers.ovis2d5_yolov13x import FusedOvis2d5Yolo
from all_fusers.r4b_yolov13x import FusedR4BYolo



def init_fuser(cfg: FusedConfig) -> FusedOvis2d5Yolo:
    """Initialize the fused model with YOLO"""
    if cfg.vlm_name.lower().find("ovis2.5") != -1:
        return FusedOvis2d5Yolo(cfg)
    elif cfg.vlm_name.lower().find("r4b") != -1:
        return FusedR4BYolo(cfg)
    else:
        raise ValueError(f"Fuser for VLM {cfg.vlm_name} is not implemented.")



if __name__ == "__main__":
    cfg = FusedConfig(
        vlm_name="AIDC-AI/Ovis2.5-2B"
    )
    model = init_fuser(cfg)
    print("\n[Success] Spatially-aligned YOLO fusion model initialized!")
    # Test inference
    image_path = "../../data/coco/train2014/COCO_train2014_000000576290.jpg"
    image = Image.open(image_path).convert("RGB")
    prompt = "Are there 4 giraffes in this image?"
    # Test with thinking enabled (uses config defaults)
    outputs = model.infer([image], [image_path], prompt)
    print(f"\nInference output (thinking={cfg.enable_thinking}): {outputs[0]}")
    # Test with thinking explicitly disabled
    outputs_no_think = model.infer([image], [image_path], prompt, enable_thinking=False)
    print(f"\nInference output (thinking=False): {outputs_no_think[0]}")
