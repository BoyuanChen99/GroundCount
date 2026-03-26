from .base import VLM, default_response

# Basic packages
import math
import requests
import time
from pathlib import Path

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split

# Torch and transformers
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Training
import wandb

# Custom imports
from vqa_loader import QAImagePathDataset, _qa_collate

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*device_map keys do not match.*")  # Ovis2.5
warnings.filterwarnings("ignore", message=".*slow image processor.*") # Image processor for all models
warnings.filterwarnings("ignore", category=FutureWarning)


# ===== R-4B =====
# This code is adapted from the official team's code on huggingface: https://huggingface.co/YannQi/R-4B
class R(VLM):
    def __init__(self, model="YannQi/R-4B", dtype=torch.float32, device=None):
        self.path = model
        if not device:
            self.model = AutoModel.from_pretrained(
                            self.path,
                            dtype=dtype,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto"
                        ).eval()
        else:
            self.model = AutoModel.from_pretrained(
                            self.path,
                            dtype=dtype,
                            trust_remote_code=True,
                        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(
                            self.path,
                            trust_remote_code=True
                        )
        self.device = self.model.device
        self.dtype = dtype
    
    def process_input(self, prompt, image=None, thinking_mode="long"):
        # Handle image input - either path, URL, PIL Image, or None
        if image is None:
            pil_image = None
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        elif isinstance(image, str):
            if "http" in image:
                pil_image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
            else:
                pil_image = Image.open(image).convert('RGB')
        else:
            raise ValueError(f"image must be str path, PIL.Image, or None. Got {type(image)}")
        if pil_image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
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
            add_generation_prompt=True,
            thinking_mode=thinking_mode,
        )
        inputs = self.processor(
            images=pil_image if pil_image is not None else None,
            text=text,
            return_tensors="pt"
        ).to(self.model.device)
        return inputs

    def infer_with_stats(
        self, 
        prompt, 
        image=None,
        max_new_tokens=1024, 
        temperature=0.0,
        enable_thinking=True,
        thinking_budget=1024,
    ):
        if enable_thinking:
            thinking_mode = "long"
        else:
            thinking_mode = "short"
        start_time = time.perf_counter()
        inputs = self.process_input(
                    prompt, 
                    image, 
                    thinking_mode=thinking_mode
                )
        gen_kwargs = {
            "max_new_tokens": max_new_tokens + (thinking_budget if enable_thinking else 0),
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed_time = time.perf_counter() - start_time
        output_ids = generated_ids[0][len(inputs.input_ids[0]):]
        num_generated_tokens = len(output_ids)
        output_text = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return output_text, num_generated_tokens, elapsed_time

    def infer(
        self, 
        prompt, 
        image=None, 
        max_new_tokens=1024, 
        temperature=0.0,
        enable_thinking=True,
        thinking_budget=1024,
    ):
        response, _, _ = self.infer_with_stats(
            prompt=prompt,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
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
        save_tag="r4b-trained",
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
        eps=1e-8,
        # Gradient accumulation for better memory usage
        gradient_accumulation_steps=1,
    ):
        """
        Train the R-4B model with batch processing, evaluation, W&B logging, 
        LR warmup, and multi-GPU support with improved VRAM distribution.
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
                    'global_batch_size': batch_size * world_size * gradient_accumulation_steps,
                    'world_size': world_size,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
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
        
        # DON'T wrap with DDP when using device_map="auto"
        # The model is already distributed across GPUs by accelerate
        model_to_train = self.model
        
        if is_distributed:
            if is_main_process:
                print(f"[Multi-GPU] Training with device_map='auto' across {world_size} processes")
                print("[Info] Not using DDP wrapper - model already distributed by accelerate")
        
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
        total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
        warmup_scheduler = None
        if warmup_steps is not None or warmup_epochs is not None:
            if warmup_steps is not None and warmup_epochs is not None:
                if is_main_process:
                    print("[Warmup] Both warmup_steps and warmup_epochs specified, using warmup_steps")
                final_warmup_steps = warmup_steps
            elif warmup_steps is not None:
                final_warmup_steps = warmup_steps
            else:
                final_warmup_steps = warmup_epochs * len(dataloader) // gradient_accumulation_steps
            
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
            optimizer.zero_grad()
            
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
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Extract loss value for logging (before clearing tensor)
                loss_item = loss.item()
                
                # Only step optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    grad_norm = None
                    if grad_clip_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Only increment global_step after actual optimizer step
                    global_step += 1
                    
                    # Periodic CUDA cache clearing
                    if global_step % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    # W&B logging (only on main process)
                    if use_wandb and is_main_process:
                        log_dict = {
                            'train/loss': loss_item * gradient_accumulation_steps,
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
                    
                    # Evaluation at fixed intervals
                    if val_loader is not None and eval_every_steps is not None:
                        if global_step % eval_every_steps == 0:
                            # Clear cache before evaluation
                            torch.cuda.empty_cache()
                            
                            if is_main_process:
                                print(f"\n[Eval] Running validation at step {global_step}...")
                            
                            val_loss = self.evaluate(val_loader, max_eval_batches, mask_user_input)
                            
                            # Clear cache after evaluation
                            torch.cuda.empty_cache()
                            
                            if is_distributed:
                                device = next(model_to_train.parameters()).device
                                val_loss_tensor = torch.tensor(val_loss, device=device)
                                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                                val_loss = (val_loss_tensor / world_size).item()
                                del val_loss_tensor
                            
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
                                    
                                    if save_fn is not None:
                                        save_fn(checkpoint_path)
                                    else:
                                        torch.save({
                                            'epoch': epoch,
                                            'step': global_step,
                                            'model_state_dict': model_to_train.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'train_loss': loss_item * gradient_accumulation_steps,
                                            'val_loss': val_loss,
                                            'best_val_loss': best_val_loss,
                                        }, checkpoint_path)
                                    print(f"[Eval] Best checkpoint saved: {checkpoint_path}\n")
                                    self._cleanup_old_checkpoints(save_dir, checkpoint_name)
                            
                            if is_distributed:
                                dist.barrier()
                
                # Clear references to free memory
                del loss
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    del batch
                
                # Logging
                epoch_loss += loss_item * gradient_accumulation_steps
                num_batches += 1
                
                # Periodic logging
                if batch_idx % (10 * gradient_accumulation_steps) == 0 and is_main_process:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss_item * gradient_accumulation_steps:.4f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Step: {global_step}")
            
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
                    device = next(model_to_train.parameters()).device
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = (val_loss_tensor / world_size).item()
                    del val_loss_tensor
                
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
                
                save_dict = {
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model_to_train.state_dict(),
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
        Compute loss for a single batch for R-4B model with improved memory management.
        Returns: (total_loss, num_valid_samples)
        """
        imgs = batch.get("images", [])
        image_paths = batch["image_paths"]
        questions = batch["question"]
        answers = batch["answer"]
        
        # Load images
        images = []
        valid_questions = []
        valid_answers = []
        
        if len(image_paths) > 0:
            for idx, img_path in enumerate(image_paths):
                try:
                    if isinstance(img_path, list):
                        img_path = img_path[0]
                    if "http" in str(img_path):
                        img = Image.open(requests.get(img_path, stream=True).raw).convert('RGB')
                    else:
                        img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    valid_questions.append(questions[idx])
                    valid_answers.append(answers[idx])
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
        else:
            for idx, img in enumerate(imgs):
                try:
                    images.append(img)
                    valid_questions.append(questions[idx])
                    valid_answers.append(answers[idx])
                except Exception as e:
                    print(f"Error processing image at index {idx}: {e}")
                    continue
        
        if len(images) == 0:
            return None, 0
        
        try:
            # Prepare all prompts with chat template (but NO answers yet)
            all_prompt_texts = []
            for i in range(len(images)):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": None},  # Placeholder
                            {"type": "text", "text": valid_questions[i]},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    thinking_mode="long",
                )
                all_prompt_texts.append(text)
            
            # Process prompts and images together
            # Move to first model device instead of self.model.device
            first_device = next(self.model.parameters()).device
            
            prompt_inputs = self.processor(
                images=images,
                text=all_prompt_texts,
                return_tensors="pt",
                padding=True
            ).to(first_device)
            
            # Clear PIL images immediately after processing
            del images
            
            # Tokenize answers separately (no special tokens, no padding yet)
            answer_token_ids = []
            for answer in valid_answers:
                ans_ids = self.processor.tokenizer(
                    answer,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids[0]
                answer_token_ids.append(ans_ids)
            
            # Find max total length for padding
            max_answer_len = max(len(ans) for ans in answer_token_ids)
            prompt_len = prompt_inputs.input_ids.shape[1]
            max_total_len = prompt_len + max_answer_len
            
            # Build input_ids and labels by concatenating prompt + answer for each sample
            batch_input_ids = []
            batch_labels = []
            batch_attention_mask = []
            
            for i in range(len(valid_questions)):
                # Get this sample's prompt tokens (already padded by processor)
                prompt_ids = prompt_inputs.input_ids[i]  # [prompt_len]
                answer_ids = answer_token_ids[i].to(first_device)  # [answer_len]
                
                # Concatenate prompt and answer
                full_ids = torch.cat([prompt_ids, answer_ids])  # [prompt_len + answer_len]
                
                # Pad to max_total_len if needed
                if len(full_ids) < max_total_len:
                    pad_len = max_total_len - len(full_ids)
                    padding = torch.full(
                        (pad_len,), 
                        self.processor.tokenizer.pad_token_id,
                        dtype=full_ids.dtype,
                        device=first_device
                    )
                    full_ids = torch.cat([full_ids, padding])
                
                # Create labels: mask prompt tokens and padding
                labels = full_ids.clone()
                if mask_user_input:
                    labels[:prompt_len] = -100  # Mask prompt
                # Mask padding tokens
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                
                # Create attention mask
                attn_mask = (full_ids != self.processor.tokenizer.pad_token_id).long()
                
                batch_input_ids.append(full_ids)
                batch_labels.append(labels)
                batch_attention_mask.append(attn_mask)
            
            # Clear intermediate lists
            del answer_token_ids
            
            # Stack into batch tensors
            input_ids = torch.stack(batch_input_ids)
            labels = torch.stack(batch_labels)
            attention_mask = torch.stack(batch_attention_mask)
            
            # Clear lists after stacking
            del batch_input_ids, batch_labels, batch_attention_mask
            
            # Prepare forward pass arguments
            forward_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
            
            # Add pixel_values
            if 'pixel_values' in prompt_inputs:
                forward_kwargs['pixel_values'] = prompt_inputs.pixel_values
            
            # Add image_sizes if it exists
            if 'image_sizes' in prompt_inputs and prompt_inputs.image_sizes is not None:
                forward_kwargs['image_sizes'] = prompt_inputs.image_sizes
            
            # Clear prompt_inputs after extracting needed components
            del prompt_inputs
            
            # Forward pass
            outputs = self.model(**forward_kwargs)
            loss = outputs.loss
            
            # Detach loss and clear outputs
            loss_value = loss.clone()
            num_valid = len(valid_questions)
            del outputs
            
            # Clear forward_kwargs
            del forward_kwargs
            
            return loss_value, num_valid
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            # Ensure cleanup even on error
            if 'images' in locals():
                del images
            return None, 0

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches=None, mask_user_input=True):
        """
        Run evaluation on validation set for R-4B model.
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
                # Clear loss after extracting value
                del loss
            
            # Clear batch
            del batch
            
            # Periodic cache clearing during validation
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(total_samples, 1)
        self.model.train()
        
        # Final cache clear after evaluation
        torch.cuda.empty_cache()
        
        return avg_loss
    
    def _cleanup_old_checkpoints(self, save_dir, current_checkpoint):
        """
        Keep only the most recent 'best' checkpoint and delete older ones.
        """
        save_path = Path(save_dir)
        best_checkpoints = sorted(save_path.glob("*_best_*.pt"))
        
        # Keep only the current checkpoint, delete older best checkpoints
        for ckpt in best_checkpoints:
            if ckpt.name != current_checkpoint:
                try:
                    ckpt.unlink()
                    print(f"[Cleanup] Removed old checkpoint: {ckpt.name}")
                except Exception as e:
                    print(f"[Cleanup] Error removing {ckpt.name}: {e}")
