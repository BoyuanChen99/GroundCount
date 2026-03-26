from .base import VLM

import math
import requests
import time
from pathlib import Path
import re
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

import wandb

from vqa_loader import QAImagePathDataset, _qa_collate

import warnings
warnings.filterwarnings("ignore", message=".*slow image processor.*")
warnings.filterwarnings("ignore", category=FutureWarning)


class Molmo2(VLM):
    def __init__(
        self,
        model="allenai/Molmo2-8B",
        dtype=torch.float32,
        device=None
    ):
        self.path = model
        if not device:
            device_map = "auto"
        else:
            device_map = {"": device}
        self.processor = AutoProcessor.from_pretrained(
            self.path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map
        ).eval()
        self.device = self.model.device
        self.dtype = dtype
        self.text_tokenizer = self.processor.tokenizer

    def process_input(
        self,
        prompt,
        image=None,
        add_generation_prompt=True
    ):
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
        content = [{"type": "text", "text": prompt}]
        if pil_image is not None:
            content.insert(0, {"type": "image", "image": pil_image})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def infer_with_stats(
        self,
        prompt,
        image=None,
        max_new_tokens=1024,
        temperature=0.0,
        add_generation_prompt=True,
        enable_thinking=True,
        thinking_budget=0,
    ):
        start_time = time.perf_counter()
        inputs = self.process_input(
            prompt=prompt,
            image=image,
            add_generation_prompt=add_generation_prompt
        )
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.text_tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        elapsed_time = time.perf_counter() - start_time
        generated_tokens = outputs[0, inputs['input_ids'].size(1):]
        num_generated_tokens = generated_tokens.shape[0]
        response = self.text_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response, num_generated_tokens, elapsed_time

    def infer(
        self,
        prompt,
        image=None,
        max_new_tokens=1024,
        temperature=0.0,
        add_generation_prompt=True
    ):
        response, _, _ = self.infer_with_stats(
            prompt=prompt,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            add_generation_prompt=add_generation_prompt
        )
        return response
    
    def infer_pointing(
        self,
        prompt,
        image=None,
        max_new_tokens=1024,
        temperature=0.0
    ):
        response, _, _ = self.infer_pointing_with_stats(
            prompt=prompt,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        return response

    def infer_pointing_with_stats(
        self,
        prompt,
        image=None,
        max_new_tokens=1024,
        temperature=0.0
    ):
        COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
        FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
        POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
        if image is None:
            raise ValueError("Image required for pointing tasks")
        if isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        elif isinstance(image, str):
            if "http" in image:
                pil_image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
            else:
                pil_image = Image.open(image).convert('RGB')
        else:
            raise ValueError(f"image must be str path or PIL.Image. Got {type(image)}")
        image_w, image_h = pil_image.width, pil_image.height
        start_time = time.perf_counter()
        inputs = self.process_input(prompt=prompt, image=pil_image, add_generation_prompt=True)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.text_tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        elapsed_time = time.perf_counter() - start_time
        generated_tokens = outputs[0, inputs['input_ids'].size(1):]
        num_generated_tokens = generated_tokens.shape[0]
        raw_text = self.text_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract points for single image
        points = []
        for coord in COORD_REGEX.finditer(raw_text):
            for point_grp in FRAME_REGEX.finditer(coord.group(1)):
                num_str = point_grp.group(2)
                for match in POINTS_REGEX.finditer(num_str):
                    idx, x, y = match.group(1), match.group(2), match.group(3)
                    x = float(x) / 1000 * image_w
                    y = float(y) / 1000 * image_h
                    if 0 <= x <= image_w and 0 <= y <= image_h:
                        points.append((x, y))
        # Format as string: "x1,y1;x2,y2;..."
        response = ";".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return response, num_generated_tokens, elapsed_time

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
            save_tag="molmo2-ori",
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
                print(f"[W&B] Initialized run: {final_run_name}")
            except ImportError:
                print("[W&B] wandb not installed, disabling logging")
                use_wandb = False

        new_save_dir = f"{save_dir}/{wandb_project}/{run_name}"
        save_dir = new_save_dir
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
                    val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False
                )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                sampler=val_sampler, num_workers=num_workers,
                collate_fn=_qa_collate, pin_memory=True
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
                    val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False
                )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                sampler=val_sampler, num_workers=num_workers,
                collate_fn=_qa_collate, pin_memory=True
            )
            if is_main_process:
                print(f"Split dataset: {train_size} train, {val_size} val samples")

        train_sampler = None
        if is_distributed:
            train_sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=local_rank,
                shuffle=shuffle, seed=seed_for_split
            )

        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=shuffle if not is_distributed else False,
            sampler=train_sampler, num_workers=num_workers,
            collate_fn=_qa_collate, pin_memory=True
        )

        model_to_train = self.model
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
            model_to_train = self.model.to(device)
            model_to_train = DDP(
                model_to_train, device_ids=[local_rank],
                output_device=local_rank, find_unused_parameters=False
            )
            if is_main_process:
                print(f"[DDP] Model wrapped with DistributedDataParallel across {world_size} GPUs")

        trainable_params = [p for p in model_to_train.parameters() if p.requires_grad]
        if is_main_process:
            print(f"Training {len(trainable_params)} parameter groups")

        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, betas=betas,
            weight_decay=weight_decay, eps=eps
        )

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

        model_to_train.train()
        global_step = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                if warmup_scheduler is not None:
                    current_lr = warmup_scheduler(global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                else:
                    current_lr = lr

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

                if batch_idx % 10 == 0 and is_main_process:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Step: {global_step}")

                if val_loader is not None and eval_every_steps is not None:
                    if global_step % eval_every_steps == 0:
                        if is_main_process:
                            print(f"\n[Eval] Running validation at step {global_step}...")

                        val_loss = self.evaluate(val_loader, max_eval_batches, mask_user_input)

                        if is_distributed:
                            val_loss_tensor = torch.tensor(val_loss, device=device)
                            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                            val_loss = (val_loss_tensor / world_size).item()

                        if is_main_process:
                            print(f"[Eval] Validation Loss: {val_loss:.4f}\n")
                            if use_wandb:
                                wandb.log({'val/loss': val_loss, 'val/step': global_step}, step=global_step)

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

            avg_loss = epoch_loss / max(num_batches, 1)
            if is_main_process:
                print(f"\nEpoch {epoch+1} completed | Average Train Loss: {avg_loss:.4f}")

            if val_loader is not None:
                if is_main_process:
                    print(f"[Eval] Running end-of-epoch validation...")

                val_loss = self.evaluate(val_loader, max_eval_batches, mask_user_input)

                if is_distributed:
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = (val_loss_tensor / world_size).item()

                if is_main_process:
                    print(f"[Eval] Validation Loss: {val_loss:.4f}\n")
                    if use_wandb:
                        wandb.log({'val/epoch_loss': val_loss, 'val/epoch': epoch + 1}, step=global_step)
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
                if use_wandb:
                    wandb.run.summary['best_val_loss'] = best_val_loss
            if use_wandb:
                wandb.finish()
                print("[W&B] Run finished")

    def _compute_batch_loss(self, batch, mask_user_input=True):
        imgs = batch.get("images", [])
        image_paths = batch["image_paths"]
        questions = batch["question"]
        answers = batch["answer"]

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

        batch_losses = []
        for i in range(len(images)):
            try:
                prompt = valid_questions[i]
                answer = valid_answers[i]
                pil_image = images[i]

                content = [{"type": "text", "text": prompt}]
                if pil_image is not None:
                    content.insert(0, {"type": "image", "image": pil_image})
                messages = [{"role": "user", "content": content}]

                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                answer_tokens = self.text_tokenizer(
                    answer, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)

                target_ids = torch.cat([inputs['input_ids'], answer_tokens], dim=1)
                attention_mask = torch.ones_like(target_ids)

                labels = target_ids.clone()
                if mask_user_input:
                    labels[:, :inputs['input_ids'].shape[1]] = -100

                pixel_values = inputs.get('pixel_values', None)
                image_input_idx = inputs.get('image_input_idx', None)
                image_masks = inputs.get('image_masks', None)

                forward_kwargs = {
                    'input_ids': target_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                }
                if pixel_values is not None:
                    forward_kwargs['pixel_values'] = pixel_values
                if image_input_idx is not None:
                    forward_kwargs['image_input_idx'] = image_input_idx
                if image_masks is not None:
                    forward_kwargs['image_masks'] = image_masks

                outputs = self.model(**forward_kwargs)
                loss = outputs.loss
                batch_losses.append(loss)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        if len(batch_losses) == 0:
            return None, 0

        total_loss = torch.stack(batch_losses).mean()
        return total_loss, len(batch_losses)

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches=None, mask_user_input=True):
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
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state_dict = checkpoint['model_state_dict']

        try:
            self.model.load_state_dict(model_state_dict, strict=strict)
            print("Model state loaded successfully")
        except RuntimeError:
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

        optimizer_state = None
        if load_optimizer:
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            if optimizer_state is not None:
                print("Optimizer state dict extracted (pass to optimizer.load_state_dict())")
            else:
                print("Warning: No optimizer state found in checkpoint")

        return checkpoint_info, optimizer_state
    
    def visualize_pointing(
        self,
        image_path,
        points_str,
        point_color='red',
        point_size=100,
        figsize=(10, 10),
        save_path=None,
        show=True
    ):
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path.convert('RGB')
        
        points = []
        if points_str:
            for pt in points_str.split(';'):
                if ',' in pt:
                    x, y = pt.split(',')
                    points.append((float(x), float(y)))
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(img)
        
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, c=point_color, s=point_size, marker='o', edgecolors='white', linewidths=1.5)
            for i, (x, y) in enumerate(points):
                ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), 
                        fontsize=9, color='white', fontweight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, ax