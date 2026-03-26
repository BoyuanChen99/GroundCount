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
warnings.filterwarnings("ignore", message=".*'repr' attribute.*Field().*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*Field().*")



# ===== Qwen2.5-VL-3/7/32/72B-Instruct =====
class Qwen2d5(VLM):
    def __init__(self, model="Qwen/Qwen2.5-VL-3B-Instruct", dtype=torch.float32):
        self.path = model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.path,
                        dtype=dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        device_map="auto"
                    ).eval()
        self.processor = AutoProcessor.from_pretrained(model)
        self.device = self.model.device
        self.dtype = dtype

    def _log_memory_stats(self, stage=""):
        """Log current GPU memory usage for debugging"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def _reset_peak_memory_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def process_input(self, prompt, image_path=None):
        # Define conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Process the message to generate inputs
        text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
        return inputs

    def infer(self, prompt, image_path=None, max_new_tokens=1024, temperature=0.0):
        inputs = self.process_input(prompt, image_path)
        # Generate output and decode
        generated_ids = self.model.generate(
                            **inputs, 
                            max_new_tokens=max_new_tokens,
                            do_sample=(temperature > 0),
                            temperature=temperature if temperature > 0 else None,
                        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
                            generated_ids_trimmed, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
                        )
        return output_text[0]

    def _get_warmup_lr(self, current_step, warmup_steps, base_lr, min_lr, warmup_type="linear"):
        """Calculate learning rate with warmup"""
        if current_step >= warmup_steps:
            return base_lr
        
        if warmup_type == "linear":
            # Linear warmup from min_lr to base_lr
            return min_lr + (base_lr - min_lr) * (current_step / warmup_steps)
        elif warmup_type == "cosine":
            # Cosine warmup
            import math
            return min_lr + (base_lr - min_lr) * (1 - math.cos(math.pi * current_step / warmup_steps)) / 2
        elif warmup_type == "constant":
            # Stay at min_lr for warmup period, then jump to base_lr
            return min_lr if current_step < warmup_steps else base_lr
        else:
            raise ValueError(f"Unknown warmup_type: {warmup_type}")

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
        debug_memory=False,  # Enable memory debugging
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
        # Wrapper class to ensure data is a proper Dataset
        class ListDataset(Dataset):
            def __init__(self, data):
                # Handle dict with parallel lists (e.g., {"image_paths": [...], "question": [...], "answer": [...]})
                if isinstance(data, dict) and all(isinstance(v, (list, tuple)) for v in data.values()):
                    # Check if it's a dict of parallel lists
                    keys = list(data.keys())
                    list_lengths = [len(data[k]) for k in keys]
                    
                    if len(set(list_lengths)) == 1:  # All lists have same length
                        # Convert to list of dicts
                        num_samples = list_lengths[0]
                        self.data = []
                        for i in range(num_samples):
                            sample = {k: data[k][i] for k in keys}
                            self.data.append(sample)
                    else:
                        raise ValueError(f"All lists in data dict must have same length, got {list_lengths}")
                elif isinstance(data, list):
                    self.data = data
                else:
                    # Assume it's already a Dataset
                    self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Set model to training mode
        self.model.train()
        
        # Wrap data if it's not already a proper Dataset
        if not isinstance(data, Dataset):
            data = ListDataset(data)
        
        # Handle data splitting if needed
        if val_data is None and val_split > 0:
            val_size = int(len(data) * val_split)
            train_size = len(data) - val_size
            generator = torch.Generator().manual_seed(seed_for_split)
            train_data, val_data = random_split(data, [train_size, val_size], generator=generator)
        else:
            train_data = data
            if val_data is not None and not isinstance(val_data, Dataset):
                val_data = ListDataset(val_data)
        
        # Create DataLoader
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        # Calculate total steps for warmup
        total_steps_per_epoch = len(train_loader)
        total_training_steps = total_steps_per_epoch * num_epochs
        
        # Determine warmup steps
        if warmup_epochs is not None:
            warmup_steps_calc = int(warmup_epochs * total_steps_per_epoch)
        elif warmup_steps is not None:
            warmup_steps_calc = warmup_steps
        else:
            warmup_steps_calc = 0  # No warmup
        
        # Initialize W&B if requested
        if use_wandb:
            import wandb
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group,
                tags=wandb_tags if wandb_tags else [],
                name=run_name,
                mode=wandb_mode if wandb_mode else "online",
                config={
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "weight_decay": weight_decay,
                    "grad_clip_norm": grad_clip_norm,
                    "warmup_steps": warmup_steps_calc,
                    "warmup_epochs": warmup_epochs,
                    "warmup_type": warmup_type,
                    "min_lr": min_lr,
                    "optimizer": "AdamW",
                    "betas": betas,
                    "eps": eps,
                    "mask_user_input": mask_user_input,
                    "model_path": self.path,
                    "total_training_steps": total_training_steps,
                    "train_size": len(train_data),
                    "val_size": len(val_data) if val_data else 0,
                }
            )
            
            # Watch model if gradient logging is enabled
            if wandb_log_grads:
                wandb.watch(self.model, log="all", log_freq=100)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=min_lr if warmup_steps_calc > 0 else lr,  # Start with min_lr if using warmup
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )
        
        # Training state
        best_val_loss = float('inf')
        global_step = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs ({total_training_steps} total steps)")
        if warmup_steps_calc > 0:
            print(f"Using {warmup_type} warmup for {warmup_steps_calc} steps (from {min_lr} to {lr})")
        
        # Training loop
        for epoch in range(num_epochs):
            if debug_memory:
                self._reset_peak_memory_stats()
                self._log_memory_stats(f"Epoch {epoch+1} Start")
            
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Update learning rate with warmup
                if warmup_steps_calc > 0 and global_step < warmup_steps_calc:
                    current_lr = self._get_warmup_lr(
                        global_step, 
                        warmup_steps_calc, 
                        lr, 
                        min_lr, 
                        warmup_type
                    )
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                elif warmup_steps_calc > 0 and global_step == warmup_steps_calc:
                    # Ensure we're at the target LR after warmup
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                if debug_memory and batch_idx % 20 == 0:
                    self._log_memory_stats(f"Epoch {epoch+1} Step {batch_idx} Before Forward")
                
                # Processdef process_i batch
                inputs, labels = self._process_batch_for_training(
                    batch, 
                    mask_user_input=mask_user_input
                )
                
                if debug_memory and batch_idx % 20 == 0:
                    self._log_memory_stats(f"Epoch {epoch+1} Step {batch_idx} After Batch Processing")
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                
                if debug_memory and batch_idx % 20 == 0:
                    self._log_memory_stats(f"Epoch {epoch+1} Step {batch_idx} After Forward")
                
                # Extract loss value before backward (to avoid retaining graph)
                loss_value = loss.item()
                
                # Backward pass
                loss.backward()
                
                if debug_memory and batch_idx % 20 == 0:
                    self._log_memory_stats(f"Epoch {epoch+1} Step {batch_idx} After Backward")
                
                # Gradient clipping
                grad_norm = None
                if grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm = grad_norm.item()
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More thorough than zero_grad()
                
                # Update metrics
                epoch_loss += loss_value
                global_step += 1
                
                # Log to W&B
                if use_wandb:
                    log_dict = {
                        "train/loss": loss_value,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + 1,
                        "train/step": global_step,
                    }
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm
                    wandb.log(log_dict, step=global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # Clear model cache if it exists (for models with KV cache)
                if hasattr(self.model, 'clear_cache'):
                    self.model.clear_cache()
                
                # Explicitly delete all tensors and outputs
                # Delete vision-specific tensors if they exist
                if 'pixel_values' in inputs:
                    del inputs['pixel_values']
                if 'image_grid_thw' in inputs:
                    del inputs['image_grid_thw']
                
                del inputs, labels, outputs, loss
                
                # Aggressive memory management after every step
                torch.cuda.synchronize()  # Wait for all operations to complete
                gc.collect()
                torch.cuda.empty_cache()
                
                if debug_memory and batch_idx % 20 == 0:
                    self._log_memory_stats(f"Epoch {epoch+1} Step {batch_idx} After Cleanup")
                
                # Validation
                if val_data is not None and global_step % eval_every_steps == 0:
                    val_loss = self._validate(
                        val_data, 
                        batch_size, 
                        num_workers,
                        max_eval_batches,
                        mask_user_input
                    )
                    print(f"\nValidation Loss at step {global_step}: {val_loss:.4f}")
                    
                    # Log validation to W&B
                    if use_wandb:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/step": global_step,
                        }, step=global_step)
                    
                    # Save best model
                    if save_on_best_val and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(save_dir, f"{save_tag}_best.pt")
                        if save_fn:
                            save_fn(self.model, save_path)
                        else:
                            torch.save(self.model.state_dict(), save_path)
                        print(f"Saved best model to {save_path} (val_loss: {val_loss:.4f})")
                        
                        # Log best model to W&B
                        if use_wandb:
                            wandb.run.summary["best_val_loss"] = val_loss
                            wandb.run.summary["best_val_step"] = global_step
                    
                    # Clear cache after validation and set back to train mode
                    self.model.train()
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # End of epoch cleanup
            del progress_bar  # Delete progress bar to release references
            gc.collect()
            torch.cuda.empty_cache()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
            
            # Log epoch summary to W&B
            if use_wandb:
                wandb.log({
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch_num": epoch + 1,
                }, step=global_step)
            
            # Save checkpoint every epoch if requested
            if save_every_epoch:
                save_path = os.path.join(save_dir, f"{save_tag}_epoch{epoch+1}.pt")
                if save_fn:
                    save_fn(self.model, save_path)
                else:
                    torch.save(self.model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")
        
        # Final save
        final_save_path = os.path.join(save_dir, f"{save_tag}_final.pt")
        if save_fn:
            save_fn(self.model, final_save_path)
        else:
            torch.save(self.model.state_dict(), final_save_path)
        print(f"Training completed. Final model saved to {final_save_path}")
        
        # Finish W&B run
        if use_wandb:
            wandb.finish()

    def _collate_fn(self, batch):
        """Collate function for DataLoader"""
        # If batch items are already lists, flatten them
        if batch and isinstance(batch[0], list):
            flattened = []
            for item in batch:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        return batch

    def _process_batch_for_training(self, batch, mask_user_input=True):
        """Process a batch of data for training"""
        from qwen_vl_utils import process_vision_info
        batch_messages = []
        batch_texts = []
        all_image_inputs = []
        all_video_inputs = []
        for item in batch:
            # item should now be a dict with keys like 'image_paths', 'question', 'answer'
            # Handle different possible key names
            image_path = (item.get("image_path") or item.get("image_paths") or 
                        item.get("image") or item.get("images"))
            question = item.get("question") or item.get("prompt")
            answer = item.get("answer") or item.get("response")
            
            # Construct conversation with both user and assistant messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},
                    ],
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # We already have the assistant response
            )
            
            batch_texts.append(text)
            batch_messages.append(messages)
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            all_image_inputs.extend(image_inputs if image_inputs else [])
            all_video_inputs.extend(video_inputs if video_inputs else [])
        
        # Process with processor
        inputs = self.processor(
            text=batch_texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Create labels
        labels = inputs.input_ids.clone()
        
        if mask_user_input:
            # Mask user input tokens (set to -100 so they're ignored in loss)
            for i, (text, messages) in enumerate(zip(batch_texts, batch_messages)):
                # Get just the user part to find where to start computing loss
                user_messages = [messages[0]]  # Only user message
                user_text = self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize to find the boundary
                user_tokens = self.processor.tokenizer(
                    user_text,
                    add_special_tokens=False
                )["input_ids"]
                
                # Mask out user tokens
                user_token_len = len(user_tokens)
                labels[i, :user_token_len] = -100
        
        # Clean up intermediate variables to avoid holding references
        del batch_messages, batch_texts, all_image_inputs, all_video_inputs
        
        return inputs, labels


    def _validate(self, val_data, batch_size, num_workers, max_eval_batches, mask_user_input):
        """Run validation and return average loss"""
        self.model.eval()
        
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if max_eval_batches and batch_idx >= max_eval_batches:
                    break
                
                inputs, labels = self._process_batch_for_training(batch, mask_user_input)
                outputs = self.model(**inputs, labels=labels)
                
                # Extract loss value immediately
                total_loss += outputs.loss.item()
                num_batches += 1
                
                # Clear model cache if it exists
                if hasattr(self.model, 'clear_cache'):
                    self.model.clear_cache()
                
                # Delete vision-specific tensors
                if 'pixel_values' in inputs:
                    del inputs['pixel_values']
                if 'image_grid_thw' in inputs:
                    del inputs['image_grid_thw']
                
                # Explicitly delete tensors after each validation batch
                del inputs, labels, outputs
                
                # Aggressive cleanup after every validation batch
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
        
        # Delete the dataloader to free any references
        del val_loader
        
        # Final aggressive cleanup after validation
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def load_checkpoint(self, checkpoint_path, strict=True, device=None, eval_mode=True):
        """
        Load model weights from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file (.pt)
            strict (bool): Whether to strictly enforce that the keys in checkpoint 
                        match the keys in the model. Default: True
            device (str or torch.device, optional): Device to load the checkpoint to.
                                                    If None, uses the current model device.
            eval_mode (bool): Whether to set the model to eval mode after loading. Default: True
        
        Returns:
            dict: Information about the loading process including:
                - 'status': 'success' or 'error'
                - 'missing_keys': List of keys in model but not in checkpoint
                - 'unexpected_keys': List of keys in checkpoint but not in model
                - 'message': Descriptive message about the loading result
        
        Example:
            >>> model = Qwen2d5("Qwen/Qwen2.5-VL-3B-Instruct")
            >>> result = model.load_checkpoint("checkpoints/model_best.pt")
            >>> print(result['message'])
        """
        # Validate checkpoint path
        if not os.path.exists(checkpoint_path):
            return {
                'status': 'error',
                'message': f"Checkpoint file not found: {checkpoint_path}"
            }
        
        try:
            # Determine device
            load_device = device if device is not None else self.model.device
            
            print(f"Loading checkpoint from: {checkpoint_path}")
            print(f"Loading to device: {load_device}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=load_device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Check if it's a full training checkpoint with metadata
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("Loaded checkpoint with metadata")
                    
                    # Optionally print additional info if available
                    if 'epoch' in checkpoint:
                        print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
                    if 'global_step' in checkpoint:
                        print(f"  Checkpoint from step: {checkpoint['global_step']}")
                    if 'best_val_loss' in checkpoint:
                        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
                else:
                    # Assume it's just a state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict into model
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, 
                strict=strict
            )
            
            # Set mode
            if eval_mode:
                self.model.eval()
                mode_msg = "Model set to eval mode"
            else:
                self.model.train()
                mode_msg = "Model set to train mode"
            
            # Prepare result message
            messages = [f"Successfully loaded checkpoint. {mode_msg}"]
            
            if missing_keys:
                messages.append(f"Warning: {len(missing_keys)} missing keys in checkpoint")
                if len(missing_keys) <= 10:
                    messages.append(f"  Missing keys: {missing_keys}")
            
            if unexpected_keys:
                messages.append(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
                if len(unexpected_keys) <= 10:
                    messages.append(f"  Unexpected keys: {unexpected_keys}")
            
            result_message = "\n".join(messages)
            print(result_message)
            
            return {
                'status': 'success',
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'message': result_message
            }
            
        except Exception as e:
            error_msg = f"Error loading checkpoint: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'error',
                'message': error_msg,
                'exception': str(e)
            }

