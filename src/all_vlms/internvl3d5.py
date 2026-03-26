from .base import VLM, default_response

# Basic packages
import math
import numpy as np
import re
import os
import requests
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import gc

# Torch and transformers
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

# Training
import wandb

# Custom imports
from vqa_loader import QAImagePathDataset, _qa_collate

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*device_map keys do not match.*")  # Ovis2.5
warnings.filterwarnings("ignore", message=".*slow image processor.*") # Image processor for all models
warnings.filterwarnings("ignore", category=FutureWarning)


R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()



# ===== InternVL3.5-1/2/4/8/14/20/30/38/241B =====
# From huggingface official code: https://huggingface.co/OpenGVLab/InternVL3_5-2B
class InternVL3d5(VLM):
    ### ========== Initialization ========== ###
    def __init__(self, model="OpenGVLab/InternVL3_5-1B", dtype=torch.float32, load_in_8bit=False, use_flash_attn=True, device=None):
        # Note that the image dimension must be fixed. The original code uses Imagenet settings by default. 
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        if "1B" in model.upper():
            model = "OpenGVLab/InternVL3_5-1B"
        elif "2B" in model.upper():
            model = "OpenGVLab/InternVL3_5-2B"
        elif "4B" in model.upper():
            model = "OpenGVLab/InternVL3_5-4B"
        elif "8B" in model.upper():
            model = "OpenGVLab/InternVL3_5-8B"
        elif "14B" in model.upper():
            model = "OpenGVLab/InternVL3_5-14B"
        elif "20B" in model.upper():
            model = "OpenGVLab/InternVL3.5-20B-A4B"
        elif "30B" in model.upper():
            model = "OpenGVLab/InternVL3.5-30B-A3B"
        elif "38B" in model.upper():
            model = "OpenGVLab/InternVL3_5-38B"
        elif "241B" in model.upper():
            model = "OpenGVLab/InternVL3.5-241B-A28B"
        self.path = model
        # Detect how many gpus are available. If there are more than 1, we will split the model evenly.
        if device:
            self.model = AutoModel.from_pretrained(
                            self.path,
                            dtype=dtype,
                            load_in_8bit=load_in_8bit,
                            use_flash_attn=use_flash_attn,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        ).eval().to(device)
        else:
            self.device_map = self.split_model(self.path)
            self.model = AutoModel.from_pretrained(
                            self.path,
                            dtype=dtype,
                            load_in_8bit=load_in_8bit,
                            use_flash_attn=use_flash_attn,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map=self.device_map
                        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
                        self.path, 
                        trust_remote_code=True, 
                        use_fast=False
                    )
        self.device = self.model.device
        self.dtype = dtype

    ### ========== Helper functions ========== ###
    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    ### A helper function used for any AutoModel that runs on multiple gpus. The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors. The returned output is a dictionary that maps layer names to device IDs.
    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        return device_map

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        # Find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    ### ========== Video-related functions ========== ###
    # Video multi-round conversation
    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def process_input(self, prompt, image=None, video=None, max_num=12, multi_rounds=False):
        # 1-image 1-round
        if video is None and isinstance(image, (str, Image.Image)) and not multi_rounds:
            if isinstance(image, str):
                pixel_values = self.load_image(image, max_num=max_num).to(self.dtype).to(self.device)
            else:
                pixel_values = self.load_image_from_pil(image, max_num=max_num).to(self.dtype).to(self.device)
            question = f"<image>\n{prompt}"
            inputs = (pixel_values, question)
            return inputs
        elif video is None and isinstance(image, list) and multi_rounds:
            pixel_values_raw = []
            for img in image:
                if isinstance(img, str):
                    pv = self.load_image(img, max_num=12).to(self.dtype).to(self.device)
                else:
                    pv = self.load_image_from_pil(img, max_num=12).to(self.dtype).to(self.device)
                pixel_values_raw.append(pv)
            pixel_values = torch.cat(pixel_values_raw, dim=0)
            question = f"<image>\n{prompt}"
            inputs = (pixel_values, question)
            return inputs

    def infer_with_stats(
        self, 
        prompt, 
        image=None, 
        video=None, 
        max_new_tokens=1024, 
        temperature=0.0, 
        multi_rounds=False,
        enable_thinking=True,
        thinking_budget=1024
    ):
        start_time = time.perf_counter()
        if enable_thinking:
            self.model.system_message = R1_SYSTEM_PROMPT
            max_new_tokens += thinking_budget
        pixel_values, question = self.process_input(prompt, image, video, multi_rounds=multi_rounds)
        if temperature > 0:
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        else:
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        elapsed_time = time.perf_counter() - start_time
        num_generated_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
        return response, num_generated_tokens, elapsed_time

    def infer(
        self, 
        prompt, 
        image=None, 
        video=None, 
        max_new_tokens=1024, 
        temperature=0.0, 
        multi_rounds=False,
        enable_thinking=True,
        thinking_budget=1024
    ):
        response, _, _ = self.infer_with_stats(
            prompt=prompt,
            image=image,
            video=video,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            multi_rounds=multi_rounds,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget
        )
        return response

    def train_loop(
        self,
        data,
        num_epochs=1,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        grad_clip_norm=1.0,
        save_every_epoch=False,
        save_tag="internvl3d5-trained",
        save_dir="../../checkpoints",
        save_fn=None,
        val_data=None,
        eval_every_steps=200,
        max_eval_batches=None,
        save_on_best_val=True,
        seed_for_split=42,
        val_split=0.05,
        mask_user_input=True,
        use_wandb=False,
        wandb_project="vlm-train",
        wandb_entity=None,
        wandb_group=None,
        wandb_tags=None,
        wandb_mode=None,
        wandb_log_grads=False,
        run_name=None,
        warmup_steps=None,
        warmup_epochs=None,
        warmup_type="linear",
        min_lr=0.0,
        eps=1e-8
    ):
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
            is_main_process = local_rank == 0
        else:
            local_rank = 0
            world_size = 1
            is_main_process = True
        
        if is_main_process:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        if is_distributed:
            dist.barrier()
        save_path = Path(save_dir)
        
        if isinstance(data, dict):
            dataset = QAImagePathDataset(data)
        else:
            dataset = data
        
        val_loader = None
        val_sampler = None
        if val_data is not None:
            if isinstance(val_data, dict):
                val_dataset = QAImagePathDataset(val_data)
            else:
                val_dataset = val_data
            if is_distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=local_rank,
                    shuffle=False
                )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=num_workers,
                collate_fn=_qa_collate,
                pin_memory=True
            )
            if is_main_process:
                print(f"Using provided validation set: {len(val_dataset)} samples")
        elif val_split > 0.0:
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            generator = torch.Generator().manual_seed(seed_for_split)
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=generator
            )
            dataset = train_dataset
            if is_distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=local_rank,
                    shuffle=False
                )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=num_workers,
                collate_fn=_qa_collate,
                pin_memory=True
            )
            if is_main_process:
                print(f"Split dataset: {train_size} train, {val_size} val samples")
        
        train_sampler = None
        if is_distributed:
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=shuffle,
                seed=seed_for_split
            )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if not is_distributed else False,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=_qa_collate,
            pin_memory=True
        )
        
        model_to_train = self.model
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
            model_to_train = self.model.to(device)
            model_to_train = DDP(
                model_to_train,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
            if is_main_process:
                print(f"[DDP] Model wrapped with DistributedDataParallel across {world_size} GPUs")
        
        trainable_params = [p for p in model_to_train.parameters() if p.requires_grad]
        if is_main_process:
            print(f"Training {len(trainable_params)} parameter groups")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )
        
        model_to_train.train()
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, batch in enumerate(dataloader):
                loss, num_valid = self._compute_batch_loss(batch, mask_user_input)
                if loss is None:
                    if is_main_process:
                        print(f"Skipping batch {batch_idx}: no valid samples")
                    continue
                optimizer.zero_grad()
                loss.backward()
                grad_norm = None
                if grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                if batch_idx % 10 == 0 and is_main_process:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                        f"Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Step: {global_step}")
                if val_loader is not None and eval_every_steps is not None:
                    if global_step % eval_every_steps == 0:
                        if is_main_process:
                            print(f"\n[Eval] Running validation at step {global_step}...")
                        
                        val_loss = self.evaluate(val_loader, max_eval_batches, mask_user_input)
                        
                        if is_distributed:
                            device = torch.device(f"cuda:{local_rank}")
                            val_loss_tensor = torch.tensor(val_loss, device=device)
                            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                            val_loss = (val_loss_tensor / world_size).item()
                        
                        if is_main_process:
                            print(f"[Eval] Validation Loss: {val_loss:.4f}\n")
                            
                            if save_on_best_val and val_loss < best_val_loss:
                                best_val_loss = val_loss
                                print(f"[Eval] New best validation loss: {val_loss:.4f}")
                                
                                checkpoint_name = f"{save_tag}_best_step{global_step}.pt"
                                checkpoint_path = save_path / checkpoint_name
                                
                                model_state = model_to_train.module.state_dict() if is_distributed else model_to_train.state_dict()
                                
                                if save_fn is not None:
                                    save_fn(checkpoint_path)
                                else:
                                    torch.save({
                                        'epoch': epoch,
                                        'step': global_step,
                                        'model_state_dict': model_state,
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'train_loss': loss.item(),
                                        'val_loss': val_loss,
                                        'best_val_loss': best_val_loss,
                                    }, checkpoint_path)
                                
                                print(f"[Eval] Best checkpoint saved: {checkpoint_path}\n")
                                self._cleanup_old_checkpoints(save_dir, checkpoint_name)
                        
                        if is_distributed:
                            dist.barrier()
            
            avg_loss = epoch_loss / max(num_batches, 1)
            if is_main_process:
                print(f"\nEpoch {epoch+1} completed | Average Train Loss: {avg_loss:.4f}")
            
            if val_loader is not None:
                if is_main_process:
                    print(f"[Eval] Running end-of-epoch validation...")
                
                val_loss = self.evaluate(val_loader, max_eval_batches, mask_user_input)
                
                if is_distributed:
                    device = torch.device(f"cuda:{local_rank}")
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = (val_loss_tensor / world_size).item()
                
                if is_main_process:
                    print(f"[Eval] Validation Loss: {val_loss:.4f}\n")
            else:
                val_loss = None
            
            if (save_every_epoch or epoch == num_epochs - 1) and is_main_process:
                checkpoint_name = f"{save_tag}_epoch{epoch+1}_step{global_step}.pt"
                checkpoint_path = save_path / checkpoint_name
                
                model_state = model_to_train.module.state_dict() if is_distributed else model_to_train.state_dict()
                
                save_dict = {
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_loss,
                }
                if val_loss is not None:
                    save_dict['val_loss'] = val_loss
                    save_dict['best_val_loss'] = best_val_loss
                
                if save_fn is not None:
                    save_fn(checkpoint_path)
                else:
                    torch.save(save_dict, checkpoint_path)
                
                print(f"Checkpoint saved: {checkpoint_path}\n")
            
            if is_distributed:
                dist.barrier()
        
        if is_main_process:
            print("Training completed!")
            if val_loader is not None and best_val_loss < float('inf'):
                print(f"Best validation loss achieved: {best_val_loss:.4f}")

    def _compute_batch_loss(self, batch, mask_user_input=True):
        image_paths = batch["image_paths"]
        questions = batch["question"]
        answers = batch["answer"]

        # Gather PIL images + aligned Q/A
        images, valid_qs, valid_as = [], [], []
        for idx, p in enumerate(image_paths):
            try:
                p0 = p[0] if isinstance(p, list) else p
                img = Image.open(p0).convert("RGB")
                images.append(img)
                valid_qs.append(questions[idx])
                valid_as.append(answers[idx])
            except Exception as e:
                print(f"Error loading image {p}: {e}")

        if len(images) == 0:
            return None, 0

        batch_losses = []
        transform = self.build_transform(input_size=448)

        for i in range(len(images)):
            try:
                img = images[i]
                question = valid_qs[i]
                answer = valid_as[i]

                # Tile image -> tensor [num_patches, 3, 448, 448]
                tiles = self.dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=12)
                pixel_values = torch.stack([transform(t) for t in tiles]).to(self.dtype).to(self.device)

                # image_flags: one flag per tile indicating all tiles are valid images
                num_tiles = pixel_values.shape[0]
                image_flags = torch.ones(num_tiles, dtype=torch.long, device=self.device)  # [num_tiles]
                
                # Simple prompt with <image> placeholder
                prompt_text = f"<image>\n{question}"
                
                # Tokenize prompt
                prompt_ids = self.tokenizer.encode(
                    prompt_text,
                    add_special_tokens=True,
                    return_tensors="pt"
                ).to(self.device)  # [1, prompt_len]

                # Tokenize answer
                answer_ids = self.tokenizer.encode(
                    answer,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.device)  # [1, answer_len]

                # Concatenate prompt and answer
                target_ids = torch.cat([prompt_ids, answer_ids], dim=1)  # [1, total_len]
                attention_mask = torch.ones_like(target_ids)

                # Create labels
                labels = target_ids.clone()
                if mask_user_input:
                    labels[:, :prompt_ids.shape[1]] = -100  # Mask prompt tokens
                print()
                print(f"labels has shape: {labels.shape}")
                print(f"attention_mask has shape: {attention_mask.shape}")
                print(f"pixel_values has shape: {pixel_values.shape}")
                print(f"image_flags has shape: {image_flags.shape}")
                print(f"labels has shape: {labels.shape}")

                # Forward pass
                outputs = self.model(
                    input_ids=target_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,      # [1, num_patches, 3, 448, 448]
                    image_flags=image_flags,        # [1, 1]
                    labels=labels,
                )
                
                batch_losses.append(outputs.loss)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()

        if len(batch_losses) == 0:
            return None, 0

        return torch.stack(batch_losses).mean(), len(batch_losses)
