import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import traceback
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import OneCycleLR

from all_vlms.ovis2d5 import Ovis2d5
from utils import disable_huggingface_warnings
from ultralytics import YOLO




class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}



@dataclass
class FusedConfig:
    vlm_name: str
    yolo_name: str = "yolov13x"
    dtype: torch.dtype = torch.float32
    device: str = "cuda:0"
    ckpt_dir: str = "../../../../checkpoints/"
    max_new_tokens: int = 1024
    enable_thinking: bool = True
    thinking_budget: int = 1024
    # Training
    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    grad_clip_norm: Optional[float] = 1.0
    num_steps_per_save: int = 1000
    # Trainable components
    train_vlm: bool = True
    train_yolo: bool = True
    train_fusion: bool = True
    train_language: bool = True
    vlm_num_layers_to_train: int = None  # If None, train all (default)
    yolo_num_layers_to_train: int = None  # If None, train all (default)
    language_num_layers_to_train: int = None  # If None, train all (default)
    vlm_unfreeze_block_from: Optional[int] = None
    yolo_unfreeze_layer_from: Optional[int] = None
    language_unfreeze_layer_from: Optional[int] = None
    # Fusion parameters
    fusion_num_heads: int = 8
    fusion_drop_path: float = 0.3
    yolo_feature_layers: Tuple[int, int, int] = (21, 26, 30)
    # Regularization
    alignment_loss_weight: float = 0.3
    alignment_loss_type: str = 'cosine'
    alignment_temperature: float = 0.07
    # Layer-wise LR decay for language decoder
    language_lr_decay: float = 0.9  # decay per layer (1.0 = no decay)




# ============= Feature Alignment Loss =============
class FeatureAlignmentLoss(nn.Module):
    """Regularization loss to align YOLO features with ViT feature space."""
    def __init__(self, loss_type='cosine', temperature=0.07):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
    
    def forward(self, yolo_proj: torch.Tensor, vit_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            yolo_proj: Projected YOLO features [B, T, C]
            vit_tokens: Original ViT tokens [B, T, C]
        """
        if self.loss_type == 'l2':
            return F.mse_loss(yolo_proj, vit_tokens.detach())
        
        elif self.loss_type == 'cosine':
            yolo_norm = F.normalize(yolo_proj, dim=-1)
            vit_norm = F.normalize(vit_tokens.detach(), dim=-1)
            similarity = (yolo_norm * vit_norm).sum(dim=-1).mean()
            return 1.0 - similarity
        
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(yolo_proj, vit_tokens.detach())
        
        elif self.loss_type == 'contrastive':
            B, T, C = yolo_proj.shape
            yolo_flat = F.normalize(yolo_proj.reshape(-1, C), dim=-1)
            vit_flat = F.normalize(vit_tokens.detach().reshape(-1, C), dim=-1)
            
            logits = torch.matmul(yolo_flat, vit_flat.T) / self.temperature
            labels = torch.arange(B * T, device=yolo_proj.device)
            
            return F.cross_entropy(logits, labels)
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")




# ============= Utility Modules =============
class DepthwiseConv2d(nn.Module):
    """Depthwise 2D convolution for spatial feature mixing"""
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DropPath(nn.Module):
    """Stochastic Depth"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor






# ============= Helper Functions =============
def extract_yes_no(text: str) -> Optional[str]:
    """Extract YES/NO from model output, checking final characters."""
    text = text.strip().upper()
    # Check last word/characters
    if text.endswith("YES"):
        return "YES"
    elif text.endswith("NO"):
        return "NO"
    # Fallback: search for last occurrence
    last_yes = text.rfind("YES")
    last_no = text.rfind("NO")
    if last_yes > last_no:
        return "YES"
    elif last_no > last_yes:
        return "NO"
    return None




# ============= Fusion Module =============
class MultiScalePyramidFusion(nn.Module):
    def __init__(
            self, 
            vit_dim: int, 
            yolo_dims: Dict[int, int], 
            num_heads: int = 8, 
            drop_path: float = 0.1
        ):
        super().__init__()
        self.scale_indices = sorted(yolo_dims.keys())
        
        self.scale_projs = nn.ModuleDict({
            str(idx): nn.Sequential(
                nn.Conv2d(dim, vit_dim, 1, bias=False),
                nn.BatchNorm2d(vit_dim),
                nn.GELU()
            ) for idx, dim in yolo_dims.items()
        })
        
        self.scale_weights = nn.Parameter(torch.ones(len(yolo_dims)) / len(yolo_dims))
        
        self.norm_q = nn.LayerNorm(vit_dim)
        self.norm_kv = nn.LayerNorm(vit_dim)
        self.cross_attn = nn.MultiheadAttention(vit_dim, num_heads, dropout=0.1, batch_first=True)
        
        # CHANGED: Initialize gate to 1.0 instead of 0.0 for stronger initial YOLO influence
        self.gate = nn.Parameter(torch.tensor([0.5]))
        self.out_norm = nn.LayerNorm(vit_dim)
        self.drop_path = DropPath(drop_path)
        
        # Store intermediate features for regularization
        self._last_yolo_proj = None
        self._last_vit_tokens = None

    def forward(
            self, 
            vit_tokens: torch.Tensor, 
            yolo_features: Dict[int, torch.Tensor], 
            grid_size: Tuple[int, int]
        ) -> torch.Tensor:
        squeezed = vit_tokens.dim() == 2
        if squeezed:
            vit_tokens = vit_tokens.unsqueeze(0)
            yolo_features = {k: v.unsqueeze(0) if v.dim() == 3 else v for k, v in yolo_features.items()}
        
        B, T, C = vit_tokens.shape
        H, W = grid_size
        
        # Store original ViT tokens for regularization
        self._last_vit_tokens = vit_tokens.detach() if self.training else None
        
        # Aggregate multi-scale
        aligned = []
        for idx in self.scale_indices:
            proj = self.scale_projs[str(idx)](yolo_features[idx])
            aligned.append(F.interpolate(proj, (H, W), mode='bilinear', align_corners=False))
        
        weights = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1)
        yolo_agg = sum(w * f for w, f in zip(weights, aligned))
        yolo_flat = yolo_agg.flatten(2).transpose(1, 2)
        
        # Store projected YOLO features for regularization
        self._last_yolo_proj = yolo_flat if self.training else None
        
        # Cross-attention
        attn_out, _ = self.cross_attn(self.norm_q(vit_tokens), self.norm_kv(yolo_flat), self.norm_kv(yolo_flat))
        
        # Gated residual
        out = self.out_norm(vit_tokens + self.drop_path(self.gate.tanh() * attn_out))
        return out.squeeze(0) if squeezed else out
    
    def get_alignment_loss(self, loss_fn: FeatureAlignmentLoss) -> Optional[torch.Tensor]:
        """Compute feature alignment loss if intermediates are available."""
        if self._last_yolo_proj is None or self._last_vit_tokens is None:
            return None
        return loss_fn(self._last_yolo_proj, self._last_vit_tokens)
    
    def get_gate_value(self) -> float:
        """Get current gate value for monitoring."""
        return self.gate.tanh().item()
