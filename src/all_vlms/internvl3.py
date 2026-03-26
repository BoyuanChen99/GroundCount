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
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration
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




# ===== InternVL3-2/8/14/38/78B =====
# This code is adapted from the official team's code on huggingface: https://huggingface.co/OpenGVLab/InternVL3-8B. Note that for this model, "max_num" means "max_partition".
class InternVL3(VLM):
    ### ========== Initialization ========== ###
    def __init__(self, model="OpenGVLab/InternVL3-8B", dtype=torch.float32, load_in_8bit=False, use_flash_attn=True, device=None):
        # Note that the image dimension must be fixed. The original code uses Imagenet settings by default. 
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        if "2B" in model.upper():
            model = "OpenGVLab/InternVL3-2B"
        elif "8B" in model.upper():
            model = "OpenGVLab/InternVL3-8B"
        elif "14B" in model.upper():
            model = "OpenGVLab/InternVL3-14B"
        elif "38B" in model.upper():
            model = "OpenGVLab/InternVL3-38B"
        elif "78B" in model.upper():
            model = "OpenGVLab/InternVL3-78B"
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

    def process_input(self, prompt, image_path=None, video=None, max_num=12, multi_rounds=False):
        # 1-image 1-round
        if video is None and isinstance(image_path, str) and not multi_rounds:
            pixel_values = self.load_image(image_path, max_num=max_num).to(self.dtype).to(self.device)
            question = f"<image>\n{prompt}"
            inputs = (pixel_values, question)
            return inputs
        elif video is None and isinstance(image_path, list) and multi_rounds:
            pixel_values_raw = [self.load_image(img_path, max_num=12).to(self.dtype).to(self.device) for img_path in image_path]
            # Torch stack them together
            pixel_values = torch.cat(pixel_values_raw, dim=0)
            question = f"<image>\n{prompt}"
            inputs = (pixel_values, question)
            return inputs

    def infer(self, prompt, image_path=None, video=None, max_new_tokens=1024, temperature=0.0, multi_rounds=False):
        pixel_values, question = self.process_input(prompt, image_path, video, multi_rounds=multi_rounds)
        response = default_response
        if temperature > 0:
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        else:
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
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
        save_tag="internvl3-trained",
        save_dir="../../checkpoints",
        save_fn=None,
        val_data=None,
        eval_every_steps=200,
        max_eval_batches=None,
        save_on_best_val=True,
        seed_for_split=42,
        val_split=0.05,
        mask_user_input=True,
        # W&B
        use_wandb=False,
        wandb_project="vlm-train",
        wandb_entity=None,
        wandb_group=None,
        wandb_tags=None,
        wandb_mode=None,
        wandb_log_grads=False,
        run_name=None,
        # Warmup
        warmup_steps=None,
        warmup_epochs=None,
        warmup_type="linear",
        min_lr=0.0,
        eps=1e-8
    ):
        """
        Train the InternVL3 model with batch processing, evaluation, W&B logging, 
        LR warmup, and multi-GPU support.
        
        This method mirrors the Ovis2d5 training loop exactly.
        """
        # Setup distributed training
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
            is_main_process = local_rank == 0
        else:
            local_rank = 0
            world_size = 1
            is_main_process = True
        
        # Initialize W&B only on main process
        if use_wandb and is_main_process:
            try:
                final_run_name = run_name or f"{save_tag}_lr{lr}"
                wandb_config = {
                    'model': save_tag,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'global_batch_size': batch_size * world_size,
                    'world_size': world_size,
                    'lr': lr,
                    'betas': betas,
                    'weight_decay': weight_decay,
                    'grad_clip_norm': grad_clip_norm,
                    'val_split': val_split,
                    'mask_user_input': mask_user_input,
                    'warmup_steps': warmup_steps,
                    'warmup_epochs': warmup_epochs,
                    'warmup_type': warmup_type,
                    'min_lr': min_lr,
                }
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=final_run_name,
                    group=wandb_group,
                    tags=wandb_tags,
                    config=wandb_config,
                    mode=wandb_mode,
                )
                if is_main_process:
                    print(f"[W&B] Initialized run: {final_run_name}")
            except ImportError:
                if is_main_process:
                    print("[W&B] wandb not installed, disabling logging")
                use_wandb = False
        
        # Create save directory (only on main process)
        new_save_dir = f"{save_dir}/{wandb_project}/{run_name}" if run_name else save_dir
        save_dir = new_save_dir
        if is_main_process:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        if is_distributed:
            dist.barrier()
        save_path = Path(save_dir)
        
        # Convert dict to dataset if needed
        if isinstance(data, dict):
            dataset = QAImagePathDataset(data)
        else:
            dataset = data
        
        # Handle validation data setup
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
        # Create training sampler and dataloader
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
        # Wrap model with DDP if distributed
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
        # Setup optimizer
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
        # Setup warmup scheduler
        total_steps = len(dataloader) * num_epochs
        warmup_scheduler = None
        if warmup_steps is not None or warmup_epochs is not None:
            if warmup_steps is not None and warmup_epochs is not None:
                if is_main_process:
                    print("[Warmup] Both warmup_steps and warmup_epochs specified, using warmup_steps")
                final_warmup_steps = warmup_steps
            elif warmup_steps is not None:
                final_warmup_steps = warmup_steps
            else:
                final_warmup_steps = warmup_epochs * len(dataloader)
            final_warmup_steps = min(final_warmup_steps, total_steps)
            def get_warmup_lr(step):
                if step >= final_warmup_steps:
                    return lr
                if warmup_type == "linear":
                    return min_lr + (lr - min_lr) * (step / final_warmup_steps)
                elif warmup_type == "cosine":
                    progress = step / final_warmup_steps
                    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
                else:
                    raise ValueError(f"Unknown warmup_type: {warmup_type}")
            warmup_scheduler = get_warmup_lr
            if is_main_process:
                print(f"[Warmup] Enabled {warmup_type} warmup for {final_warmup_steps} steps "
                    f"(from {min_lr} to {lr})")
        # Training state
        model_to_train.train()
        global_step = 0
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, batch in enumerate(dataloader):
                # Apply warmup learning rate
                if warmup_scheduler is not None:
                    current_lr = warmup_scheduler(global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                else:
                    current_lr = lr
                # Compute loss
                loss, num_valid = self._compute_batch_loss(batch, mask_user_input)
                if loss is None:
                    if is_main_process:
                        print(f"Skipping batch {batch_idx}: no valid samples")
                    continue
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                grad_norm = None
                if grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                optimizer.step()
                # Logging
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                # W&B logging (only on main process)
                if use_wandb and is_main_process:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/epoch': epoch + batch_idx / len(dataloader),
                        'train/lr': current_lr,
                        'train/step': global_step,
                    }
                    if grad_norm is not None:
                        log_dict['train/grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    if wandb_log_grads:
                        grad_stats = {}
                        for name, param in model_to_train.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_stats[f'grads/{name}_mean'] = param.grad.mean().item()
                                grad_stats[f'grads/{name}_std'] = param.grad.std().item()
                                grad_stats[f'grads/{name}_max'] = param.grad.max().item()
                        log_dict.update(grad_stats)
                    wandb.log(log_dict, step=global_step)
                # Periodic logging
                if batch_idx % 10 == 0 and is_main_process:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                        f"Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Step: {global_step}")
                # Evaluation at fixed intervals
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
                            
                            if use_wandb:
                                wandb.log({
                                    'val/loss': val_loss,
                                    'val/step': global_step,
                                }, step=global_step)
                            
                            if save_on_best_val and val_loss < best_val_loss:
                                best_val_loss = val_loss
                                print(f"[Eval] New best validation loss: {val_loss:.4f}")
                                
                                if use_wandb:
                                    wandb.log({'val/best_loss': best_val_loss}, step=global_step)
                                
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
            # Epoch summary
            avg_loss = epoch_loss / max(num_batches, 1)
            if is_main_process:
                print(f"\nEpoch {epoch+1} completed | Average Train Loss: {avg_loss:.4f}")
            # End-of-epoch validation
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
                    if use_wandb:
                        wandb.log({
                            'val/epoch_loss': val_loss,
                            'val/epoch': epoch + 1,
                        }, step=global_step)
            else:
                val_loss = None
            # Save checkpoint (only on main process)
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
                if use_wandb:
                    wandb.run.summary['best_val_loss'] = best_val_loss
            
            if use_wandb:
                wandb.finish()
                print("[W&B] Run finished")

    def _compute_batch_loss(self, batch, mask_user_input=True):
        """
        Compute loss for a single batch for InternVL3 model.
        Returns: (total_loss, num_valid_samples)
        """
        imgs = batch.get("images", [])
        image_paths = batch["image_paths"]
        questions = batch["question"]
        answers = batch["answer"]
        
        # Load images
        images = []
        valid_paths = []
        valid_questions = []
        valid_answers = []
        
        if len(image_paths) > 0:
            for idx, img_path in enumerate(image_paths):
                try:
                    if isinstance(img_path, list):
                        # Multiple images for this sample
                        sample_images = []
                        for p in img_path:
                            img = Image.open(p).convert("RGB")
                            sample_images.append(img)
                        images.append(sample_images)
                        valid_paths.append(img_path)
                    else:
                        # Single image
                        img = Image.open(img_path).convert("RGB")
                        images.append([img])
                        valid_paths.append([img_path])
                    valid_questions.append(questions[idx])
                    valid_answers.append(answers[idx])
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
        else:
            for idx, img in enumerate(imgs):
                try:
                    if isinstance(img, list):
                        images.append(img)
                    else:
                        images.append([img])
                    valid_paths.append(None)
                    valid_questions.append(questions[idx])
                    valid_answers.append(answers[idx])
                except Exception as e:
                    print(f"Error processing image at index {idx}: {e}")
                    continue
        
        if len(images) == 0:
            return None, 0
        
        # Process each sample in the batch
        batch_losses = []
        for i in range(len(images)):
            try:
                prompt = valid_questions[i]
                answer = valid_answers[i]
                sample_images = images[i]
                sample_paths = valid_paths[i] if valid_paths[i] else [None] * len(sample_images)
                
                # Process images: load pixel values for each image
                pixel_values_list = []
                for idx, img_path in enumerate(sample_paths):
                    if img_path:
                        pv = self.load_image(img_path, max_num=12)
                    else:
                        # Process PIL image directly
                        img = sample_images[idx]
                        transform = self.build_transform(input_size=448)
                        tiles = self.dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=12)
                        pv = torch.stack([transform(tile) for tile in tiles])
                    pixel_values_list.append(pv)
                
                # Concatenate all pixel values
                pixel_values = torch.cat(pixel_values_list, dim=0).to(self.dtype).to(self.device)
                num_patches = pixel_values.shape[0]
                
                # Build prompt with image tokens
                num_images = len(sample_images)
                image_tokens = "<image>\n" * num_images
                full_prompt = f"{image_tokens}{prompt}"
                
                # Tokenize prompt
                prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=True, return_tensors="pt").to(self.device)
                
                # Tokenize answer
                answer_ids = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").to(self.device)
                
                # Concatenate prompt and answer
                target_ids = torch.cat([prompt_ids, answer_ids], dim=1)
                attention_mask = torch.ones_like(target_ids)
                
                # Create labels
                labels = target_ids.clone()
                if mask_user_input:
                    labels[:, :prompt_ids.shape[1]] = -100  # Mask prompt tokens
                
                # Create image_flags: marks positions where images appear
                # InternVL expects image_flags to indicate image token positions
                image_flags = torch.zeros(target_ids.shape[1], dtype=torch.long, device=self.device)
                
                # Find <image> token id
                image_token_id = self.tokenizer.encode("<image>", add_special_tokens=False)[0]
                
                # Mark all positions with <image> token as 1
                image_positions = (target_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
                if len(image_positions) > 0:
                    image_flags[image_positions] = 1
                
                # Reshape for batch dimension
                image_flags = image_flags.unsqueeze(0)  # [1, seq_len]
                
                # Forward pass
                outputs = self.model(
                    input_ids=target_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,  # Add batch dimension
                    image_flags=image_flags,  # ADD THIS
                    labels=labels,
                )
                
                loss = outputs.loss
                batch_losses.append(loss)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(batch_losses) == 0:
            return None, 0
        
        # Average loss across batch
        total_loss = torch.stack(batch_losses).mean()
        return total_loss, len(batch_losses)



        @torch.no_grad()
        def evaluate(self, val_loader, max_batches=None, mask_user_input=True):
            """
            Run evaluation on validation set for InternVL3 model.
            
            Args:
                val_loader: Validation data loader
                max_batches: Maximum number of batches to evaluate
                mask_user_input: Whether to mask user input tokens in loss
            
            Returns:
                avg_val_loss: Average validation loss
            """
            self.model.eval()
            total_loss = 0.0
            total_samples = 0
            
            for batch_idx, batch in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                loss, num_valid = self._compute_batch_loss(batch, mask_user_input)
                
                if loss is not None:
                    total_loss += loss.item() * num_valid
                    total_samples += num_valid
            
            avg_loss = total_loss / max(total_samples, 1)
            self.model.train()
            return avg_loss

        def _cleanup_old_checkpoints(self, save_dir, current_checkpoint):
            """
            Remove all checkpoint files except the current one.
            """
            save_path = Path(save_dir)
            if not save_path.exists():
                return
            
            all_checkpoints = list(save_path.glob("*.pt"))
            current_path = save_path / current_checkpoint
            
            for ckpt_path in all_checkpoints:
                if ckpt_path != current_path:
                    try:
                        ckpt_path.unlink()
                        print(f"[Cleanup] Removed old checkpoint: {ckpt_path.name}")
                    except Exception as e:
                        print(f"[Cleanup] Failed to remove {ckpt_path.name}: {e}")

        def load_checkpoint(self, checkpoint_path, load_optimizer=False, strict=True):
            """
            Load a saved checkpoint into the model.
            
            Args:
                checkpoint_path: Path to the checkpoint file (.pt)
                load_optimizer: Whether to return optimizer state dict for resuming training
                strict: Whether to strictly enforce that the keys in state_dict match
            
            Returns:
                checkpoint_info: Dictionary containing checkpoint metadata
                optimizer_state: Optimizer state dict (if load_optimizer=True, else None)
            """
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            print(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state dict
            model_state_dict = checkpoint['model_state_dict']
            
            # Handle DDP wrapped models
            try:
                self.model.load_state_dict(model_state_dict, strict=strict)
                print("Model state loaded successfully")
            except RuntimeError as e:
                print("Attempting to load with prefix adjustment...")
                new_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                
                self.model.load_state_dict(new_state_dict, strict=strict)
                print("Model state loaded successfully (with prefix adjustment)")
            
            # Prepare checkpoint info
            checkpoint_info = {
                'epoch': checkpoint.get('epoch', -1),
                'step': checkpoint.get('step', -1),
                'train_loss': checkpoint.get('train_loss', None),
                'val_loss': checkpoint.get('val_loss', None),
                'best_val_loss': checkpoint.get('best_val_loss', None),
            }
            
            print(f"Checkpoint info: Epoch {checkpoint_info['epoch']}, "
                f"Step {checkpoint_info['step']}, "
                f"Train Loss: {checkpoint_info['train_loss']}, "
                f"Val Loss: {checkpoint_info['val_loss']}")
            
            # Return optimizer state if requested
            optimizer_state = None
            if load_optimizer:
                optimizer_state = checkpoint.get('optimizer_state_dict', None)
                if optimizer_state is not None:
                    print("Optimizer state dict extracted (pass to optimizer.load_state_dict())")
                else:
                    print("Warning: No optimizer state found in checkpoint")
            
            return checkpoint_info, optimizer_state
        
        def log(
                self,
                images=None,
                prompt: str = "Describe the image.",
                max_new_tokens: int = 16,
                temperature: float = 0.0,
                include: str = "vision,mlp1,decoder,lm_head",
                print_fn=None):
            """
            Log layer outputs (shape, token length, hidden dim) during a single chat() call.

            Logs are written into 'logs.txt' instead of terminal.
            """
            include = {x.strip().lower() for x in include.split(",")}

            # --- setup file logger ---
            log_file = open("logs.txt", "w", encoding="utf-8")

            def print_fn(msg):
                log_file.write(msg + "\n")

            # -------- helpers --------
            def _first_tensor(x):
                if torch.is_tensor(x):
                    return x
                if isinstance(x, (list, tuple)):
                    for y in x:
                        t = _first_tensor(y)
                        if t is not None:
                            return t
                if isinstance(x, dict):
                    for k in ("last_hidden_state", "hidden_states"):
                        if k in x:
                            t = _first_tensor(x[k])
                            if t is not None:
                                return t
                    for v in x.values():
                        t = _first_tensor(v)
                        if t is not None:
                            return t
                return None

            def _tok_dim(t: torch.Tensor):
                if t.ndim == 3:
                    _, T, C = t.shape
                    return int(T), int(C)
                if t.ndim == 4:
                    _, C, H, W = t.shape
                    return int(H * W), int(C)
                return None, None

            @contextmanager
            def _no_grad_eval(model):
                was_training = model.training
                try:
                    model.eval()
                    with torch.no_grad():
                        yield
                finally:
                    if was_training:
                        model.train()

            def _nice_name(full_name: str):
                subs = [
                    (r"^vision_model\.embeddings\.patch_embedding", "vit.patch"),
                    (r"^vision_model\.embeddings", "vit.emb"),
                    (r"^vision_model\.encoder\.layers\.", "vit.enc."),
                    (r"^vision_model", "vit"),
                    (r"^mlp1", "proj"),
                    (r"^language_model\.model\.layers\.", "dec."),
                    (r"^language_model\.model\.embed_tokens", "tok_emb"),
                    (r"^language_model\.model\.norm", "dec.norm"),
                    (r"^language_model\.lm_head", "lm_head"),
                ]
                name = full_name
                for pat, rep in subs:
                    name = re.sub(pat, rep, name)
                return name

            def _want(name: str, mod: nn.Module) -> bool:
                lname = name.lower()
                if "vision" in include and lname.startswith("vision_model"):
                    return True
                if "mlp1" in include and (lname == "mlp1" or lname.startswith("mlp1")):
                    return True
                if "decoder" in include and lname.startswith("language_model.model.layers"):
                    return True
                if "embeds" in include and ("embed_tokens" in lname):
                    return True
                if "norms" in include and (lname.endswith(".norm") or lname.endswith("layernorm") or "norm" in lname):
                    return True
                if "lm_head" in include and lname.endswith("lm_head"):
                    return True
                return False

            def _make_hook(name):
                def hook(mod, inp, out):
                    t = _first_tensor(out)
                    if t is None:
                        print_fn(f"[{_nice_name(name)}] (no tensor output)")
                        return
                    shape_str = "×".join(str(int(x)) for x in t.shape)
                    T, C = _tok_dim(t)
                    if T is not None and C is not None:
                        print_fn(f"[{_nice_name(name)}] shape={shape_str}  tokens={T}  hidden={C}")
                    else:
                        print_fn(f"[{_nice_name(name)}] shape={shape_str}")
                return hook

            # ---------- prepare inputs (0, 1, or many images) ----------
            def _to_pixel_values_from_pil(img: Image.Image, input_size=448, max_num=12):
                transform = self.build_transform(input_size=input_size)
                tiles = self.dynamic_preprocess(img.convert("RGB"), image_size=input_size, use_thumbnail=True, max_num=max_num)
                px = torch.stack([transform(t) for t in tiles])
                return px

            def _to_pixel_values_any(x, input_size=448, max_num=12):
                if isinstance(x, Image.Image):
                    return _to_pixel_values_from_pil(x, input_size=input_size, max_num=max_num)
                if isinstance(x, (str, os.PathLike)):
                    return self.load_image(str(x), input_size=input_size, max_num=max_num)
                raise TypeError(f"Unsupported image type: {type(x)}")

            if images is None:
                images_list = []
            elif isinstance(images, (str, os.PathLike, Image.Image)):
                images_list = [images]
            elif isinstance(images, (list, tuple)):
                images_list = list(images)
            else:
                raise TypeError("`images` must be None, a path/PIL image, or a list of them.")

            pixel_batches = []
            for itm in images_list:
                pv = _to_pixel_values_any(itm, input_size=448, max_num=12)
                pixel_batches.append(pv)

            pixel_values = torch.cat(pixel_batches, dim=0).to(self.dtype).to(self.device) if pixel_batches else None
            image_tokens = "\n".join(["<image>"] * len(pixel_batches))
            question = (image_tokens + ("\n" if image_tokens else "")) + prompt

            handles = []
            try:
                for name, module in self.model.named_modules():
                    if _want(name, module):
                        if isinstance(module, (nn.Dropout, nn.Identity)):
                            continue
                        handles.append(module.register_forward_hook(_make_hook(name)))

                gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=(temperature > 0), temperature=temperature) \
                        if temperature > 0 else dict(max_new_tokens=max_new_tokens, do_sample=False)

                with _no_grad_eval(self.model):
                    _ = self.model.chat(self.tokenizer, pixel_values, question, gen_cfg)

            finally:
                for h in handles:
                    h.remove()
                log_file.close()

