from .base import VLM

# Basic packages
import time

# Torch and transformers
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*device_map keys do not match.*")  # Ovis2.5
warnings.filterwarnings("ignore", message=".*slow image processor.*") # Image processor for all models
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*'repr' attribute.*Field().*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*Field().*")

# Log in to Huggingface Hub
from huggingface_hub import login




# ====== Qwen3-VL-Thinking ====== #
# Code from https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking
class Qwen3(VLM):
    def __init__(self, model="Qwen/Qwen3-VL-2B-Thinking", dtype=torch.float32, device=None):
        if not device:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model,
                        dtype=dtype,
                        device_map="auto"
                    ).eval()
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model,
                        dtype=dtype,
                        device_map={"": device}
                    ).eval()
        self.path = model
        self.processor = AutoProcessor.from_pretrained(model)
        self.device = self.model.device
        self.dtype = dtype

    def process_input(self, prompt, image=None, add_generation_prompt=True):
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=add_generation_prompt
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        return inputs

    def infer_with_stats(
        self, 
        prompt, 
        image=None,
        max_new_tokens=1024,
        enable_thinking=True,
        thinking_budget=1024,
        temperature=0.0,
        add_generation_prompt=True
    ):
        start_time = time.perf_counter()
        inputs = self.process_input(
            prompt, 
            image,
            add_generation_prompt=add_generation_prompt
        )
        gen_kwargs = {
            "max_new_tokens": max_new_tokens+(thinking_budget if enable_thinking else 0),
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed_time = time.perf_counter() - start_time
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        num_generated_tokens = len(generated_ids_trimmed[0])
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return output_text[0], num_generated_tokens, elapsed_time

    def infer(
        self, 
        prompt, 
        image=None,
        max_new_tokens=1024, 
        temperature=0.0,
        add_generation_prompt=True,
        enable_thinking=True,
        thinking_budget=1024
    ):
        output_text, _, _ = self.infer_with_stats(
            prompt, 
            image, 
            max_new_tokens, 
            temperature,
            add_generation_prompt,
            enable_thinking,
            thinking_budget
        )
        return output_text
