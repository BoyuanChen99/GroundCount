# src/all_llms/qwen3.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
from typing import Optional, Tuple, Union

from .base import LLM


class Qwen3(LLM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        quantization: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.think_end_token_id = 151668  # </think>
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
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

    def _parse_thinking(self, output_ids: list) -> Tuple[str, str]:
        try:
            index = len(output_ids) - output_ids[::-1].index(self.think_end_token_id)
        except ValueError:
            index = 0
        thinking = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return thinking, content

    def infer(
        self,
        prompt: str = None,
        messages: Optional[list] = None,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = True,
        return_thinking: bool = False,
        max_new_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.99,
        top_k: int = 100,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs
    ) -> str:
        response, num_generated_tokens, elapsed_time = self.infer_with_stats(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            return_thinking=return_thinking,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_beams=num_beams,
            **kwargs
        )
        return response

    def infer_with_stats(
        self,
        prompt: str = None,
        messages: Optional[list] = None,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = True,
        return_thinking: bool = False,
        max_new_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.99,
        top_k: int = 100,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs
    ) -> Tuple[Union[str, Tuple[str, str]], int, float]:
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
            enable_thinking=enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        start_time = time.perf_counter()
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
        elapsed_time = time.perf_counter() - start_time
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        num_generated_tokens = len(output_ids)
        
        if enable_thinking:
            thinking, content = self._parse_thinking(output_ids)
            if return_thinking:
                return (thinking, content), num_generated_tokens, elapsed_time
            return content, num_generated_tokens, elapsed_time
        else:
            return self.tokenizer.decode(output_ids, skip_special_tokens=True), num_generated_tokens, elapsed_time
        