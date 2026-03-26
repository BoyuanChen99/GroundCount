from .base import VLM

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




# ===== Ovis2.5-2B/9B =====
# This code is adapted from the official huggingface page: https://huggingface.co/AIDC-AI/Ovis2.5-2B
class Ovis2d5(VLM):
    def __init__(
            self, 
            model="AIDC-AI/Ovis2.5-2B", 
            dtype=torch.float32, 
            device=None
        ):
        self.path = model
        # Set environment variable to use slow image processor
        os.environ["TRANSFORMERS_USE_FAST_PROCESSOR"] = "0"
        if not device:
            device_map = "auto"
        else:
            device_map = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(
                        self.path,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        device_map=device_map
                    ).eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                        self.path, 
                        trust_remote_code=True, 
                        use_fast=False
                    )
        self.device = self.model.device
        self.dtype = dtype
        ### Ovis2.5 processes images internally. These processors cannot be set.
        # self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
        # self.visual_tokenizer = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
    
    def process_input(
        self, 
        prompt, 
        image=None,  # Can be path (str) or PIL.Image
        enable_thinking=True, 
        add_generation_prompt=True
    ):
        # Handle image input - either path or PIL Image
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
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ] if pil_image is not None else [{"type": "text", "text": prompt}],
        }]
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking
        )
        input_ids = input_ids.to(self.model.device)
        pixel_values = pixel_values.to(self.model.device) if pixel_values is not None else None
        grid_thws = grid_thws.to(self.model.device) if grid_thws is not None else None
        return input_ids, pixel_values, grid_thws

    def infer_with_stats(
        self, 
        prompt, 
        image=None,  # Can be path (str) or PIL.Image
        max_new_tokens=1024, 
        thinking_budget=1024, 
        temperature=0.0, 
        enable_thinking=True, 
        add_generation_prompt=True
    ):
        start_time = time.perf_counter()
        input_ids, pixel_values, grid_thws = self.process_input(
            prompt=prompt, 
            image=image, 
            enable_thinking=(enable_thinking and thinking_budget > 0), 
            add_generation_prompt=add_generation_prompt
        )
        if enable_thinking:
            max_new_tokens += thinking_budget
        outputs = self.model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=enable_thinking,
            enable_thinking_budget=(enable_thinking and thinking_budget > 0),
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.text_tokenizer.eos_token_id,
        )
        elapsed_time = time.perf_counter() - start_time
        num_generated_tokens = outputs.shape[1]
        if isinstance(prompt, str):
            response = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = [self.model.text_tokenizer.decode(o, skip_special_tokens=True) for o in outputs][0]
        return response, num_generated_tokens, elapsed_time
    
    def infer(
        self, 
        prompt, 
        image=None,  # Can be path (str) or PIL.Image
        max_new_tokens=1024, 
        thinking_budget=1024, 
        temperature=0.0, 
        enable_thinking=True, 
        add_generation_prompt=True
    ):
        response, _, _ = self.infer_with_stats(
            prompt=prompt, 
            image=image, 
            max_new_tokens=max_new_tokens, 
            thinking_budget=thinking_budget, 
            temperature=temperature, 
            enable_thinking=enable_thinking, 
            add_generation_prompt=add_generation_prompt
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
        save_tag="ovis2d5-ori",
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
        Train the Ovis2.5 model with batch processing, evaluation, W&B logging, LR warmup, and multi-GPU support.
        
        Args:
            data: Training dataset (Dict or QAImagePathDataset)
            num_epochs: Number of training epochs
            batch_size: Batch size per GPU for training
            shuffle: Whether to shuffle training data
            num_workers: Number of data loading workers
            lr: Learning rate (peak LR if using warmup)
            betas: Adam optimizer betas
            weight_decay: Weight decay coefficient
            grad_clip_norm: Gradient clipping norm (None to disable)
            save_every_epoch: Whether to save checkpoint every epoch
            save_tag: Tag for checkpoint saving
            save_fn: Custom save function (optional)
            val_data: Validation dataset (optional)
            eval_every_steps: Run evaluation every N training steps
            max_eval_batches: Maximum number of validation batches
            save_on_best_val: Save checkpoint when validation loss improves
            seed_for_split: Random seed for train/val split
            val_split: Fraction of training data to use for validation
            mask_user_input: Whether to mask user input tokens in loss
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            wandb_entity: W&B entity/username
            wandb_group: W&B group for organizing runs
            wandb_tags: List of tags for the W&B run
            wandb_mode: W&B mode ('online', 'offline', 'disabled')
            wandb_log_grads: Whether to log gradient statistics to W&B
            run_name: Alias for wandb_run_name
            warmup_steps: Number of warmup steps (mutually exclusive with warmup_epochs)
            warmup_epochs: Number of warmup epochs (mutually exclusive with warmup_steps)
            warmup_type: Type of warmup schedule ('linear', 'cosine')
            min_lr: Minimum learning rate (for cosine warmup)
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
                # Use run_name or fall back to run_name
                final_run_name = run_name or run_name or f"{save_tag}_lr{lr}"
                # Prepare W&B config
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
        new_save_dir = f"{save_dir}/{wandb_project}/{run_name}"
        save_dir = new_save_dir+""
        if is_main_process:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        # Barrier to ensure directory is created before other processes continue
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
            # Create validation sampler for distributed training
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
                shuffle=False if is_distributed else False,
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
            # Create validation sampler for distributed training
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
                shuffle=False if is_distributed else False,
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
            # Move model to the correct device first
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
            eps=eps  # Add epsilon for numerical stability
        )
        
        # Setup warmup scheduler
        total_steps = len(dataloader) * num_epochs
        warmup_scheduler = None
        if warmup_steps is not None or warmup_epochs is not None:
            # Calculate warmup steps
            if warmup_steps is not None and warmup_epochs is not None:
                if is_main_process:
                    print("[Warmup] Both warmup_steps and warmup_epochs specified, using warmup_steps")
                final_warmup_steps = warmup_steps
            elif warmup_steps is not None:
                final_warmup_steps = warmup_steps
            else:
                final_warmup_steps = warmup_epochs * len(dataloader)
            final_warmup_steps = min(final_warmup_steps, total_steps)
            
            # Create warmup scheduler function
            def get_warmup_lr(step):
                if step >= final_warmup_steps:
                    return lr
                if warmup_type == "linear":
                    # Linear warmup from min_lr to lr
                    return min_lr + (lr - min_lr) * (step / final_warmup_steps)
                elif warmup_type == "cosine":
                    # Cosine warmup from min_lr to lr
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
            # Set epoch for distributed sampler
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
                
                # Compute loss (using the original model for _compute_batch_loss)
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
                    
                    # Log gradient statistics if requested
                    if wandb_log_grads:
                        grad_stats = {}
                        for name, param in model_to_train.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_stats[f'grads/{name}_mean'] = param.grad.mean().item()
                                grad_stats[f'grads/{name}_std'] = param.grad.std().item()
                                grad_stats[f'grads/{name}_max'] = param.grad.max().item()
                        log_dict.update(grad_stats)
                    
                    wandb.log(log_dict, step=global_step)
                
                # Periodic logging (only on main process)
                if batch_idx % 10 == 0 and is_main_process:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                        f"Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Step: {global_step}")
                
                # Evaluation at fixed intervals (only on main process for saving)
                if val_loader is not None and eval_every_steps is not None:
                    if global_step % eval_every_steps == 0:
                        if is_main_process:
                            print(f"\n[Eval] Running validation at step {global_step}...")
                        
                        val_loss = self.evaluate(val_loader, max_eval_batches, mask_user_input)
                        
                        # Aggregate validation loss across all processes
                        if is_distributed:
                            val_loss_tensor = torch.tensor(val_loss, device=device)
                            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                            val_loss = (val_loss_tensor / world_size).item()
                        
                        if is_main_process:
                            print(f"[Eval] Validation Loss: {val_loss:.4f}\n")
                            
                            # W&B logging
                            if use_wandb:
                                wandb.log({
                                    'val/loss': val_loss,
                                    'val/step': global_step,
                                }, step=global_step)
                            
                            # Save on best validation loss
                            if save_on_best_val and val_loss < best_val_loss:
                                best_val_loss = val_loss
                                print(f"[Eval] New best validation loss: {val_loss:.4f}")
                                
                                # W&B logging
                                if use_wandb:
                                    wandb.log({'val/best_loss': best_val_loss}, step=global_step)
                                
                                # Save checkpoint
                                checkpoint_name = f"{save_tag}_best_step{global_step}.pt"
                                checkpoint_path = save_path / checkpoint_name
                                
                                # Extract model state dict (unwrap DDP if needed)
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
                        
                        # Barrier to ensure all processes wait for evaluation to complete
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
                
                # Aggregate validation loss across all processes
                if is_distributed:
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = (val_loss_tensor / world_size).item()
                
                if is_main_process:
                    print(f"[Eval] Validation Loss: {val_loss:.4f}\n")
                    
                    # W&B logging
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
                
                # Extract model state dict (unwrap DDP if needed)
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
            
            # Barrier to ensure all processes complete the epoch together
            if is_distributed:
                dist.barrier()
        
        if is_main_process:
            print("Training completed!")
            if val_loader is not None and best_val_loss < float('inf'):
                print(f"Best validation loss achieved: {best_val_loss:.4f}")
                if use_wandb:
                    wandb.run.summary['best_val_loss'] = best_val_loss
            
            # Finish W&B run
            if use_wandb:
                wandb.finish()
                print("[W&B] Run finished")


    def _compute_batch_loss(self, batch, mask_user_input=True):
        """
        Compute loss for a single batch.
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
                        img_path = img_path[0]
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    valid_paths.append(img_path)
                    valid_questions.append(questions[idx])
                    valid_answers.append(answers[idx])
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
        else:
            for idx, img in enumerate(imgs):
                try:
                    images.append(img)
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
                # Process input
                input_ids, pixel_values, grid_thws = self.process_input(
                    prompt,
                    image=valid_paths[i],
                    enable_thinking=False,
                    add_generation_prompt=True
                )
                # Tokenize the answer
                tok = self.model.text_tokenizer
                answer_tokens = tok(
                    answer,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(self.device)
                # Concatenate prompt and answer
                target_ids = torch.cat([input_ids, answer_tokens], dim=1)
                attention_mask = torch.ones_like(target_ids)
                # Create labels
                labels = target_ids.clone()
                if mask_user_input:
                    labels[:, :input_ids.shape[1]] = -100  # Mask prompt tokens
                # Forward pass
                outputs = self.model(
                    input_ids=target_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    labels=labels,
                )
                loss = outputs.loss
                batch_losses.append(loss)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        if len(batch_losses) == 0:
            return None, 0
        # Average loss across batch
        total_loss = torch.stack(batch_losses).mean()
        return total_loss, len(batch_losses)

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches=None, mask_user_input=True):
        """
        Run evaluation on validation set.
        
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
        # Handle DDP wrapped models - the saved state might have 'module.' prefix
        # Try to load directly first
        try:
            self.model.load_state_dict(model_state_dict, strict=strict)
            print("Model state loaded successfully")
        except RuntimeError as e:
            # If it fails, try removing 'module.' prefix (in case saved from DDP)
            print("Attempting to load with prefix adjustment...")
            new_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
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
