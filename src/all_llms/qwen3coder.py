# src/all_llms/qwen3coder.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Optional

from .base import LLM


class Qwen3Coder(LLM):
    def __init__(
        self, 
        model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        quantization: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if quantization == "4bit":
            print(f"Loading model {model_name} with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        elif quantization == "8bit":
            print(f"Loading model {model_name} with 8-bit quantization...")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            print(f"Loading model {model_name} in bfloat16 without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto"
            )
    

    def infer(
        self,
        prompt: str,
        messages: Optional[list] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 65536,
        temperature: float = 0.7,
        top_p: float = 0.99,
        top_k: int = 100,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs
    ):
        if prompt is None and messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        if not messages:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_beams=num_beams,
            **kwargs
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)