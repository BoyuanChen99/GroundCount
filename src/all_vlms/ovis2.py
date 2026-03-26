from .base import VLM, default_response

import math
import os
import requests
import time
from pathlib import Path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split

# Torch and transformers
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# Training
import wandb

# Custom imports
from vqa_loader import QAImagePathDataset, _qa_collate

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*device_map keys do not match.*")  # Ovis2.5
warnings.filterwarnings("ignore", message=".*slow image processor.*") # Image processor for all models
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*'repr' attribute.*Field().*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*Field().*")




# ===== Ovis2-2/8/16B =====
# This code is adapted from the official team's code on huggingface: https://huggingface.co/AIDC-AI/Ovis2-8B
class Ovis2(VLM):
    def __init__(self, model="AIDC-AI/Ovis2-2B", dtype=torch.float32):
        self.path = model
        self.model = AutoModelForCausalLM.from_pretrained(
                        self.path,
                        dtype=dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        device_map="auto"
                    ).eval()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.device = self.model.device
    
    def infer(self, prompt, image_path=None, video=None, max_new_tokens=1024, temperature=0.0, max_partition=12):
        response = default_response
        # 1-image 1-round
        if image_path is not None and video is None:
            image = Image.open(image_path).convert('RGB')
            images = [image]
            query = f'<image>\n{prompt}'
            prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
            pixel_values = [pixel_values]
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    top_p=None,
                    top_k=None,
                    temperature=temperature if temperature > 0 else None,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return response


