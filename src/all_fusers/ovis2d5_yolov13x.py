from .base import EMA, FusedConfig, MultiScalePyramidFusion, FeatureAlignmentLoss, extract_yes_no

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import traceback
import random
import time

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import OneCycleLR

from all_vlms.ovis2d5 import Ovis2d5
from utils import disable_huggingface_warnings
from ultralytics import YOLO




# ============= Full Fused Model =============
class FusedOvis2d5Yolo(nn.Module):
    """Fuses Ovis2.5 VLM with YOLOv13x object detection."""
    def __init__(self, cfg: FusedConfig):
        super().__init__()
        self.cfg = cfg
        disable_huggingface_warnings()
        
        # Load models
        self.ovisvlm = Ovis2d5(model=cfg.vlm_name, dtype=cfg.dtype, device=cfg.device)
        self.vlm = self.ovisvlm.model
        
        # Load YOLOv13x with hard-coded weights link and HF_HOME path
        print(f"[YOLO] Loading {cfg.yolo_name}...")
        yolo_link = "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13x.pt"
        path_HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/huggingface_cache"))
        weights_path = os.path.join(path_HF_HOME, "yolov13x.pt")
        if not os.path.exists(weights_path):
            os.system(f"wget {yolo_link} -O {weights_path}")
        self.yolo = YOLO(weights_path)
        self.yolo.model.to(cfg.device)
        
        # Extract YOLO model backbone
        self.yolo_backbone = self.yolo.model.model
        
        # Determine YOLO feature dimension
        self._setup_yolo_hooks(cfg.yolo_feature_layers)
        
        # Extract ViT dimensions
        self.vlm_vision_dim = (self.vlm.visual_tokenizer.vit.vision_model
                              .embeddings.patch_embedding.out_channels)
        
        # Setup encoder hook for fusion
        self._ovis_encoder = self.vlm.visual_tokenizer.vit.vision_model.encoder
        self._ovis_encoder_orig_forward = self._ovis_encoder.forward
        
        # Current YOLO features and grid size
        self._current_yolo_features: Optional[torch.Tensor] = None
        self._current_grid_size: Optional[Tuple[int, int]] = None
        
        # Create spatially-aligned fusion module
        vit_param = next(self._ovis_encoder.parameters())
        
        fusion_config = {
            'vit_dim': self.vlm_vision_dim,
            'yolo_dim': self.yolo_feature_dim,
            'out_dim': self.vlm_vision_dim,
            'num_heads': cfg.fusion_num_heads,
            'drop_path': cfg.fusion_drop_path,
        }
        
        self.fusion_proj = MultiScalePyramidFusion(
            vit_dim=self.vlm_vision_dim,
            yolo_dims=self.yolo_feature_dims,
            num_heads=cfg.fusion_num_heads,
            drop_path=cfg.fusion_drop_path,
        ).to(device=vit_param.device, dtype=vit_param.dtype)

        self.alignment_loss_fn = FeatureAlignmentLoss(
            loss_type=cfg.alignment_loss_type,
            temperature=cfg.alignment_temperature
        )
        print(f"[Regularization] Alignment loss: {cfg.alignment_loss_type}, weight: {cfg.alignment_loss_weight}")
        
        print(f"[Fusion] Initialized spatially-aligned fusion")
        print(f"  ViT dim: {self.vlm_vision_dim}")
        print(f"  YOLO dim: {self.yolo_feature_dim}")
        print(f"  Config: {fusion_config}")
        
        # Create fusion forward function
        self._create_fusion_forward()
        
        # Configure trainable parameters
        self._set_trainable()
    
    def _setup_yolo_hooks(self, feature_layer_indices: Tuple[int, ...]):
        """Setup hooks to extract multi-scale YOLO features."""
        self.yolo_feature_dims = {}
        self._yolo_features_multiscale = {}
        
        model_sequential = self.yolo.model.model
        layers = list(model_sequential)
        
        for layer_idx in feature_layer_indices:
            if layer_idx >= len(layers):
                raise ValueError(f"Layer {layer_idx} out of range")
            
            target_layer = layers[layer_idx]
            
            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    self._yolo_features_multiscale[idx] = output
                    if idx not in self.yolo_feature_dims:
                        self.yolo_feature_dims[idx] = output.shape[1]
                        print(f"[YOLO] Layer {idx} feature dim: {output.shape[1]}, spatial: {output.shape[2:]}")
                return hook_fn
            
            target_layer.register_forward_hook(make_hook(layer_idx))
            print(f"[YOLO] Registered hook at layer {layer_idx}: {target_layer.__class__.__name__}")
        
        # Dummy forward
        dummy_input = torch.randn(1, 3, 640, 640).to(self.cfg.device)
        with torch.no_grad():
            _ = self.yolo.model(dummy_input)
        
        # Store dimensions in order
        self.yolo_feature_dim = sum(self.yolo_feature_dims.values())  # Total for compatibility
        print(f"[YOLO] Multi-scale dims: {self.yolo_feature_dims}, total: {self.yolo_feature_dim}")
    
    def _create_fusion_forward(self):
        """Creates the fused forward function for the encoder."""
        def _encoder_fw_spatial_fuse(*args, **kwargs):
            out = self._ovis_encoder_orig_forward(*args, **kwargs)
            
            # Normalize output format
            if isinstance(out, tuple):
                hs = out[0]
                all_hs = out[1] if len(out) > 1 else None
            elif hasattr(out, "last_hidden_state"):
                hs = out.last_hidden_state
                all_hs = getattr(out, "hidden_states", None)
            else:
                hs, all_hs = out, None
            
            # Early return if no fusion needed
            if hs is None or self._current_yolo_features is None or self._current_grid_size is None:
                return (hs, all_hs)
            
            # Apply spatially-aligned fusion
            fused = self.fusion_proj(
                hs, 
                self._current_yolo_features,
                self._current_grid_size
            )
            # Verify shape preservation
            if fused.shape != hs.shape:
                raise RuntimeError(
                    f"Fusion shape mismatch! Input: {hs.shape}, Output: {fused.shape}"
                )
            # Update hidden states if present
            if all_hs is not None:
                is_tuple = isinstance(all_hs, tuple)
                hs_list = list(all_hs)
                if hs_list:
                    hs_list[-1] = fused
                all_hs = tuple(hs_list) if is_tuple else hs_list
            return (fused, all_hs)
        self._ovis_encoder_fused_forward = _encoder_fw_spatial_fuse
    
    def _freeze_for_language_only(self):
        """Freeze all modules except language decoder for language-only training phase."""
        # Freeze fusion
        for p in self.fusion_proj.parameters():
            p.requires_grad_(False)
        self.fusion_proj.eval()
        # Freeze VLM vision components
        for p in self.vlm.visual_tokenizer.parameters():
            p.requires_grad_(False)
        self.vlm.visual_tokenizer.eval()
        # Freeze cross-modal components
        for name, p in self.vlm.named_parameters():
            if any(k in name for k in ["vision_proj", "mm_projector", "cross_attn", "adapter", "gating"]):
                p.requires_grad_(False)
        # Freeze YOLO
        for p in self.yolo_backbone.parameters():
            p.requires_grad_(False)
        self.yolo.model.eval()
        # Keep language decoder trainable (re-enable in case it was partially frozen)
        try:
            lang_layers = self._get_language_layers()
            num_lang_layers = len(lang_layers)
            num_to_train = self.cfg.language_num_layers_to_train or num_lang_layers
            
            for i, layer in enumerate(lang_layers):
                if self.cfg.language_num_layers_to_train is None or i < num_to_train:
                    for p in layer.parameters():
                        p.requires_grad_(True)
                    layer.train()
            print(f"[Language-Only] Enabled {min(num_to_train, num_lang_layers)}/{num_lang_layers} language layers")
        except RuntimeError as e:
            print(f"[Language-Only] Warning: Could not setup language training - {e}")
        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Language-Only] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.1f}%)")


    def _setup_language_only_optimizer(
        self,
        base_lr: float = 1e-5,
        lr_multipliers: Optional[Dict[str, float]] = None,
    ):
        """Setup optimizer for language-only training phase."""
        default_multipliers = {'language': 100.0}
        if lr_multipliers is not None:
            default_multipliers.update(lr_multipliers)
        multipliers = default_multipliers
        param_groups = []
        try:
            lang_layers = self._get_language_layers()
            num_lang_layers = len(lang_layers)
            num_to_train = self.cfg.language_num_layers_to_train or num_lang_layers
            decay = self.cfg.language_lr_decay
            lang_base_lr = base_lr * multipliers['language']
            print(f"[Language-Only LR] Layer-wise LR (decay={decay}):")
            for i, layer in enumerate(lang_layers):
                if self.cfg.language_num_layers_to_train is not None and i >= num_to_train:
                    continue
                layer_params = [p for p in layer.parameters() if p.requires_grad]
                if not layer_params:
                    continue
                num_trainable = min(num_to_train, num_lang_layers)
                layer_lr = lang_base_lr * (decay ** (num_trainable - i - 1))
                param_groups.append({
                    'params': layer_params,
                    'lr': layer_lr,
                    'name': f'language_layer_{i}'
                })
                if i < 3 or i >= num_trainable - 3:
                    print(f"      Layer {i:2d}: {layer_lr:.2e}")
                elif i == 3:
                    print(f"      ...")
        except RuntimeError as e:
            print(f"[Language-Only LR] Error: {e}")
            return []
        return param_groups


    def _get_language_layers(self) -> List[nn.Module]:
        """Get language model transformer layers."""
        # Try to find LLM module
        llm_module = None
        for name in ['llm', 'language_model', 'lm', 'model']:
            if hasattr(self.vlm, name):
                llm_module = getattr(self.vlm, name)
                # Check if this module has transformer layers
                for attr in ['layers', 'layer', 'h', 'blocks', 'transformer_blocks']:
                    if hasattr(llm_module, attr):
                        layers_attr = getattr(llm_module, attr)
                        if isinstance(layers_attr, (list, tuple, nn.ModuleList)):
                            return list(layers_attr)
                # If not found at top level, search recursively
                for child in llm_module.children():
                    for attr in ['layers', 'layer', 'h', 'blocks', 'transformer_blocks']:
                        if hasattr(child, attr):
                            layers_attr = getattr(child, attr)
                            if isinstance(layers_attr, (list, tuple, nn.ModuleList)):
                                return list(layers_attr)
        raise RuntimeError("Could not find language model transformer layers")


    def _set_trainable(self, val_mode: bool = False):
        """Configure which parameters are trainable."""
        # Freeze everything by default
        for p in self.parameters():
            p.requires_grad_(False)
        if val_mode:
            return
        
        # Fusion block is fully trainable
        if self.cfg.train_fusion:
            for p in self.fusion_proj.parameters():
                p.requires_grad_(True)
            print("[Train] Fusion block enabled")
        
        # VLM training (last N ViT blocks + cross-modal components)
        if self.cfg.train_vlm:
            # Cross-modal components
            language_components = []
            for name, p in self.vlm.named_parameters():
                if any(k in name for k in ["vision_proj", "mm_projector", "lm_head", 
                                        "cross_attn", "adapter", "gating"]):
                    p.requires_grad_(True)
                    component = name.split('.')[0]
                    if component not in language_components:
                        language_components.append(component)
            # Last N ViT blocks
            blocks = self._get_vit_layers()
            num_blocks = len(blocks)
            if self.cfg.vlm_num_layers_to_train is not None:
                # Train last N layers in VLM
                num_to_train = self.cfg.vlm_num_layers_to_train
                start_idx = max(0, num_blocks - num_to_train)
                for i, blk in enumerate(blocks):
                    for p in blk.parameters():
                        p.requires_grad_(i >= start_idx)
                print(f"[Train] VLM enabled: last {num_to_train}/{num_blocks} ViT blocks, components: {language_components}")
            else:
                # Train full VLM
                for i, blk in enumerate(blocks):
                    for p in blk.parameters():
                        p.requires_grad_(True)
                print(f"[Train] VLM enabled: all {num_blocks}/{num_blocks} ViT blocks, components: {language_components}")
            
        # Language model training (first N transformer layers - interact with vision)
        if self.cfg.train_language:
            try:
                lang_layers = self._get_language_layers()
                num_lang_layers = len(lang_layers)
                num_to_train = self.cfg.language_num_layers_to_train
                if num_to_train is None:
                    for layer in lang_layers:
                        for p in layer.parameters():
                            p.requires_grad_(True)
                    print(f"[Train] Language model enabled: all {num_lang_layers}/{num_lang_layers} layers")
                else:
                    for i, layer in enumerate(lang_layers):
                        for p in layer.parameters():
                            p.requires_grad_(i < num_to_train)
                    print(f"[Train] Language model enabled: first {num_to_train}/{num_lang_layers} layers")
            except RuntimeError as e:
                print(f"[Train] Warning: Could not enable language training - {e}")
        
        # YOLO training (last N layers or full model)
        if self.cfg.train_yolo:
            layers = list(self.yolo_backbone.children())
            num_layers = len(layers)
            if self.cfg.yolo_num_layers_to_train is not None:
                # Train last N layers
                num_to_train = self.cfg.yolo_num_layers_to_train
                start_idx = max(0, num_layers - num_to_train)
                for i, layer in enumerate(layers):
                    for p in layer.parameters():
                        p.requires_grad_(i >= start_idx)
                print(f"[Train] YOLO enabled: last {num_to_train}/{num_layers} layers")
            else:
                # Train full model (default behavior)
                for p in self.yolo_backbone.parameters():
                    p.requires_grad_(True)
                print(f"[Train] YOLO enabled: FULL MODEL")
        return


    
    def _get_vit_layers(self) -> List[nn.Module]:
        """Get ViT encoder layers."""
        enc = self.vlm.visual_tokenizer.vit.vision_model.encoder
        layers = getattr(enc, "layers", None) or getattr(enc, "layer", None)
        if layers is None:
            raise RuntimeError("Could not find vision encoder layers")
        return list(layers)
    

    def _extract_yolo_features(self, images: List[Image.Image]) -> Dict[int, torch.Tensor]:
        """Extract multi-scale features from YOLO."""
        img_tensors = []
        for img in images:
            img_resized = img.resize((640, 640))
            img_array = np.array(img_resized).transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            img_tensors.append(img_tensor)
        batch_tensor = torch.stack(img_tensors).to(self.cfg.device)
        self._yolo_features_multiscale.clear()
        if self.training:
            _ = self.yolo.model(batch_tensor)
        else:
            with torch.no_grad():
                _ = self.yolo.model(batch_tensor)
        return {k: v.clone() for k, v in self._yolo_features_multiscale.items()}


    @staticmethod
    def _extract_grid_size(grid_thws: torch.Tensor) -> Tuple[int, int]:
        """Extract (H, W) from grid_thws tensor."""
        # grid_thws format: [n_images, 3] where columns are (temporal, height, width)
        # For images, temporal=1, so we take height and width
        if grid_thws.dim() == 2 and grid_thws.shape[1] == 3:
            # Take the first image's dimensions
            _, h, w = grid_thws[0].tolist()
            return (int(h), int(w))
        elif grid_thws.dim() == 1 and len(grid_thws) == 3:
            _, h, w = grid_thws.tolist()
            return (int(h), int(w))
        else:
            raise ValueError(f"Unexpected grid_thws format: {grid_thws.shape}")

    @contextmanager
    def _with_yolo_features(self, yolo_features: Dict[int, torch.Tensor], grid_size: Tuple[int, int]):
        """Context manager to temporarily inject YOLO features into the encoder."""
        # Store features and grid size
        self._current_yolo_features = yolo_features
        self._current_grid_size = grid_size
        # Replace encoder forward with fused version
        original_forward = self._ovis_encoder.forward
        self._ovis_encoder.forward = self._ovis_encoder_fused_forward
        try:
            yield
        finally:
            # Restore original forward and clear features
            self._ovis_encoder.forward = original_forward
            self._current_yolo_features = None
            self._current_grid_size = None


    def print_anatomy(self):
        """Print detailed architecture breakdown with layer counts and parameters."""
        def count_params(module):
            """Count total and trainable parameters."""
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable
        def fmt_param(n):
            """Format parameter count."""
            if n >= 1e9:
                return f"{n/1e9:.2f}B"
            elif n >= 1e6:
                return f"{n/1e6:.2f}M"
            elif n >= 1e3:
                return f"{n/1e3:.2f}K"
            return str(n)
        def count_transformer_layers(module):
            """Recursively find transformer layers count."""
            # Common attributes for transformer layers
            for attr in ['layers', 'layer', 'h', 'blocks', 'transformer_blocks']:
                if hasattr(module, attr):
                    layers_attr = getattr(module, attr)
                    if isinstance(layers_attr, (list, tuple, nn.ModuleList)):
                        return len(layers_attr)
            # Recursively search in children
            for child in module.children():
                count = count_transformer_layers(child)
                if count > 0:
                    return count
            return 0
        anatomy = {}
        
        # ========== 1. Fusion Block ==========
        fusion_total, fusion_train = count_params(self.fusion_proj)
        fusion_layers = len(list(self.fusion_proj.modules())) - 1
        anatomy['fusion'] = {
            'num_layers': fusion_layers,
            'total_params': fusion_total,
            'trainable_params': fusion_train,
        }
        
        # ========== 2. ViT Encoder ==========
        vit_encoder = self.vlm.visual_tokenizer.vit.vision_model.encoder
        vit_blocks = self._get_vit_layers()
        vit_total, vit_train = count_params(vit_encoder)
        anatomy['vit'] = {
            'num_blocks': len(vit_blocks),
            'total_params': vit_total,
            'trainable_params': vit_train,
        }
        
        # ========== 3. Language Model / Decoder ==========
        vlm_total, vlm_train = count_params(self.vlm)
        visual_tokenizer_total, visual_tokenizer_train = count_params(self.vlm.visual_tokenizer)
        
        lang_total = vlm_total - visual_tokenizer_total
        lang_train = vlm_train - visual_tokenizer_train
        
        # Find the actual LLM and count its layers
        llm_num_layers = 0
        llm_module = None
        
        # Try to find LLM module
        for name in ['llm', 'language_model', 'lm', 'model']:
            if hasattr(self.vlm, name):
                llm_module = getattr(self.vlm, name)
                llm_num_layers = count_transformer_layers(llm_module)
                if llm_num_layers > 0:
                    break
        
        # Count trainable language decoder layers
        lang_trainable_layers = 0
        if llm_module is not None:
            try:
                lang_layers = self._get_language_layers()
                for layer in lang_layers:
                    if any(p.requires_grad for p in layer.parameters()):
                        lang_trainable_layers += 1
            except:
                pass
        
        # Discover language model components
        lang_components = []
        lang_component_details = {}
        
        for name, module in self.vlm.named_children():
            if name != 'visual_tokenizer':
                lang_components.append(name)
                total, trainable = count_params(module)
                num_layers = count_transformer_layers(module)
                
                if num_layers > 0:
                    lang_component_details[name] = {
                        'num_layers': num_layers,
                        'total': total,
                        'trainable': trainable
                    }
                else:
                    lang_component_details[name] = {
                        'total': total,
                        'trainable': trainable
                    }
        
        anatomy['language'] = {
            'num_layers': llm_num_layers,  # Actual transformer layers
            'trainable_layers': lang_trainable_layers,
            'num_components': len(lang_components),
            'component_names': lang_components,
            'component_details': lang_component_details,
            'total_params': lang_total,
            'trainable_params': lang_train,
        }
        
        # ========== 4. YOLO ==========
        yolo_layers = list(self.yolo_backbone.children())
        yolo_total, yolo_train = count_params(self.yolo_backbone)
        anatomy['yolo'] = {
            'num_layers': len(yolo_layers),
            'total_params': yolo_total,
            'trainable_params': yolo_train,
        }
        
        # ========== Total ==========
        model_total, model_train = count_params(self)
        anatomy['total'] = {
            'total_params': model_total,
            'trainable_params': model_train,
        }
        
        # ========== Print ==========
        print("\n" + "="*70)
        print("MODEL ANATOMY")
        print("="*70)
        
        print(f"\n{'Component':<20} {'Layers':<15} {'Total':<15} {'Trainable':<15} {'% Train':<10}")
        print("-"*70)
        
        # Fusion
        print(f"{'Fusion Block':<20} {anatomy['fusion']['num_layers']:<15} "
            f"{fmt_param(anatomy['fusion']['total_params']):<15} "
            f"{fmt_param(anatomy['fusion']['trainable_params']):<15} "
            f"{anatomy['fusion']['trainable_params']/max(anatomy['fusion']['total_params'],1)*100:>6.1f}%")
        
        # ViT
        print(f"{'ViT Encoder':<20} {anatomy['vit']['num_blocks']:<15} "
            f"{fmt_param(anatomy['vit']['total_params']):<15} "
            f"{fmt_param(anatomy['vit']['trainable_params']):<15} "
            f"{anatomy['vit']['trainable_params']/max(anatomy['vit']['total_params'],1)*100:>6.1f}%")
        
        # Language
        lang_layers_str = str(anatomy['language']['num_layers']) if anatomy['language']['num_layers'] > 0 else f"{anatomy['language']['num_components']} comp"
        print(f"{'Language Model':<20} {lang_layers_str:<15} "
            f"{fmt_param(anatomy['language']['total_params']):<15} "
            f"{fmt_param(anatomy['language']['trainable_params']):<15} "
            f"{anatomy['language']['trainable_params']/max(anatomy['language']['total_params'],1)*100:>6.1f}%")
        
        # YOLO
        print(f"{'YOLO':<20} {anatomy['yolo']['num_layers']:<15} "
            f"{fmt_param(anatomy['yolo']['total_params']):<15} "
            f"{fmt_param(anatomy['yolo']['trainable_params']):<15} "
            f"{anatomy['yolo']['trainable_params']/max(anatomy['yolo']['total_params'],1)*100:>6.1f}%")
        
        print("-"*70)
        print(f"{'TOTAL MODEL':<20} {'':<15} "
            f"{fmt_param(anatomy['total']['total_params']):<15} "
            f"{fmt_param(anatomy['total']['trainable_params']):<15} "
            f"{anatomy['total']['trainable_params']/max(anatomy['total']['total_params'],1)*100:>6.1f}%")
        
        print("\n" + "-"*70)
        print("LANGUAGE MODEL BREAKDOWN:")
        print("-"*70)
        for comp in anatomy['language']['component_names']:
            details = anatomy['language']['component_details'][comp]
            if 'num_layers' in details:
                print(f"  • {comp:<25} {details['num_layers']:>3} layers  "
                    f"{fmt_param(details['total']):>10}  "
                    f"({fmt_param(details['trainable'])} trainable)")
            else:
                print(f"  • {comp:<25} {fmt_param(details['total']):>10}  "
                    f"({fmt_param(details['trainable'])} trainable)")
        
        print("\n" + "-"*70)
        print("VLM STRUCTURE:")
        print("-"*70)
        print(f"  • Visual Tokenizer:  {fmt_param(visual_tokenizer_total):>10}  "
            f"({fmt_param(visual_tokenizer_train)} trainable)")
        print(f"  • Language Model:    {fmt_param(lang_total):>10}  "
            f"({fmt_param(lang_train)} trainable)")
        print(f"  • Total VLM:         {fmt_param(vlm_total):>10}  "
            f"({fmt_param(vlm_train)} trainable)")
        
        print("\n" + "-"*70)
        print("TRAINING CONFIGURATION:")
        print("-"*70)
        print(f"  • Fusion:   {'✓ Enabled' if self.cfg.train_fusion else '✗ Frozen'}")
        num_to_train = self.cfg.vlm_num_layers_to_train
        print(f"  • VLM:      {'✓ Enabled' if self.cfg.train_vlm else '✗ Frozen'} "
            f"(last {num_to_train}/{anatomy['vit']['num_blocks']} ViT blocks + cross-modal)")
        print(f"  • Vision Encoder Layers: {anatomy['vit']['num_blocks']}")
        print(f"  • YOLO:     {'✓ Enabled' if self.cfg.train_yolo else '✗ Frozen'}")
        if self.cfg.train_language and anatomy['language']['trainable_layers'] > 0:
            print(f"  • Language Decoders Enabled: {anatomy['language']['trainable_layers']}/{anatomy['language']['num_layers']}")
        
        print("="*70 + "\n")



    @torch.no_grad()
    def infer_with_stats(
        self,
        images: List[Image.Image],
        image_paths: List[str],
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.0,
    ) -> Tuple[List[str], int, float]:
        """Run inference with YOLO-enhanced vision and return stats.
        
        Returns:
            Tuple of (outputs, total_generated_tokens, elapsed_time)
        """
        start_time = time.perf_counter()
        self._set_trainable(val_mode=True)
        config = generation_config or {}
        enable_thinking = config.get("enable_thinking", self.cfg.enable_thinking) if enable_thinking is None else enable_thinking
        max_new_tokens = config.get("max_new_tokens", self.cfg.max_new_tokens) + (config.get("thinking_budget", self.cfg.thinking_budget) if enable_thinking else 0)
        _enable_thinking = (
            enable_thinking if enable_thinking is not None 
            else config.get("enable_thinking", self.cfg.enable_thinking)
        )
        _thinking_budget = (
            thinking_budget if thinking_budget is not None 
            else config.get("thinking_budget", self.cfg.thinking_budget)
        )
        assert len(images) == len(image_paths)
        yolo_features = self._extract_yolo_features(images)
        outputs = []
        total_tokens = 0
        tok = getattr(self.vlm, "text_tokenizer", None)
        for i, pth in enumerate(image_paths):
            input_ids, pixel_values, grid_thws = self.ovisvlm.process_input(
                prompt=prompt, image=pth, enable_thinking=_enable_thinking
            )
            grid_size = self._extract_grid_size(grid_thws)
            single_yolo_features = {k: v[i:i+1] for k, v in yolo_features.items()}
            with self._with_yolo_features(single_yolo_features, grid_size):
                gen_kwargs = dict(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                )
                if _enable_thinking:
                    gen_kwargs["thinking_budget"] = _thinking_budget
                output_ids = self.vlm.generate(**gen_kwargs)
            if tok:
                text = tok.decode(output_ids[0], skip_special_tokens=True)
                total_tokens += len(tok.encode(text, add_special_tokens=False))
            else:
                text = str(output_ids)
                total_tokens += len(output_ids[0]) if hasattr(output_ids[0], '__len__') else 1
            outputs.append(text)
        elapsed_time = time.perf_counter() - start_time
        return outputs, total_tokens, elapsed_time


    @torch.no_grad()
    def infer(
        self,
        images: List[Image.Image],
        image_paths: List[str],
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.0,
    ) -> List[str]:
        """Run inference with YOLO-enhanced vision."""
        outputs, _, _ = self.infer_with_stats(
            images=images,
            image_paths=image_paths,
            prompt=prompt,
            generation_config=generation_config,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return outputs


    def _compute_batch_loss(
        self, 
        batch, 
        accumulate_gradients: bool = False,
        enable_thinking: bool = False,
        global_step: int = 0,
    ):
        """Compute loss with YOLO spatial features and alignment regularization."""
        image_paths = batch["image_paths"]
        questions = batch["question"]
        answers = batch["answer"]
        images = []
        valid_paths = []
        valid_questions = []
        valid_answers = []
        for idx, img_input in enumerate(image_paths):
            try:
                if isinstance(img_input, list):
                    img_input = img_input[0]
                # Handle PIL Image or file path
                if isinstance(img_input, Image.Image):
                    img = img_input.convert("RGB") if img_input.mode != "RGB" else img_input
                elif isinstance(img_input, str):
                    img = Image.open(img_input).convert("RGB")
                else:
                    print(f"Invalid image input type: {type(img_input)}")
                    continue
                images.append(img)
                valid_paths.append(img_input)
                valid_questions.append(questions[idx])
                valid_answers.append(answers[idx])
            except Exception as e:
                print(f"Error loading image {img_input}: {e}")
                continue
        if not images:
            return None, 0
        yolo_features = self._extract_yolo_features(images)
        total_loss_value = 0.0
        total_align_loss_value = 0.0
        num_valid_samples = 0
        for i in range(len(images)):
            try:
                single_yolo_features = {k: v[i:i+1] for k, v in yolo_features.items()}
                prompt = valid_questions[i]
                answer = valid_answers[i]
                input_ids, pixel_values, grid_thws = self.ovisvlm.process_input(
                    prompt, image=valid_paths[i], enable_thinking=enable_thinking
                )
                grid_size = self._extract_grid_size(grid_thws)
                with self._with_yolo_features(single_yolo_features, grid_size):
                    tok = self.vlm.text_tokenizer
                    answer_tokens = tok(
                        answer, return_tensors="pt", add_special_tokens=False
                    ).input_ids.to(self.cfg.device)
                    target_ids = torch.cat([input_ids, answer_tokens], dim=1)
                    attention_mask = torch.ones_like(target_ids)
                    labels = target_ids.clone()
                    labels[:, :input_ids.shape[1]] = -100
                    outputs = self.vlm(
                        input_ids=target_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        grid_thws=grid_thws,
                        labels=labels,
                    )
                    task_loss = outputs.loss
                    # Compute alignment regularization loss
                    align_loss = self.fusion_proj.get_alignment_loss(self.alignment_loss_fn)
                    if align_loss is not None:
                        # ADD WARMUP - pass global_step as parameter
                        # Note: You'll need to pass global_step to this method
                        alignment_warmup_steps = 500
                        current_align_weight = min(
                            self.cfg.alignment_loss_weight,
                            self.cfg.alignment_loss_weight * (global_step / max(alignment_warmup_steps, 1))
                        )
                        total_loss = task_loss + current_align_weight * align_loss
                        total_align_loss_value += align_loss.item()
                    else:
                        total_loss = task_loss
                    if accumulate_gradients:
                        scaled_loss = total_loss / len(images)
                        scaled_loss.backward()
                        total_loss_value += task_loss.item()
                    else:
                        total_loss_value += task_loss.item()
                    num_valid_samples += 1
                    del outputs, task_loss, total_loss
                    if align_loss is not None:
                        del align_loss
                    del input_ids, pixel_values, grid_thws
                    del answer_tokens, target_ids, attention_mask, labels
                    if accumulate_gradients:
                        del scaled_loss
                    if i % 4 == 0:
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                traceback.print_exc()
                continue
        if num_valid_samples == 0:
            return None, 0
        avg_loss_value = total_loss_value / num_valid_samples
        avg_align_loss = total_align_loss_value / num_valid_samples if total_align_loss_value > 0 else 0
        if avg_align_loss > 0:
            print(f"    Task Loss: {avg_loss_value:.4f} | Align Loss: {avg_align_loss:.4f}")
        return torch.tensor(avg_loss_value, device=self.cfg.device), num_valid_samples


    def _setup_optimizer_with_groups(
        self, 
        base_lr: float = 1e-5,
        lr_multipliers: Optional[Dict[str, float]] = None
    ):
        """Setup optimizer with component-specific learning rates.
        
        Language decoder uses layer-wise LR decay: lr_i = base_lr * decay^(num_layers - i - 1)
        where i=0 is the first layer (lowest LR) and i=num_layers-1 is the last layer (highest LR).
        """
        default_multipliers = {
            'fusion': 10.0,
            'vit': 0.1,
            'language': 0.1,
            'yolo': 100.0,
        }
        if lr_multipliers is not None:
            default_multipliers.update(lr_multipliers)
        multipliers = default_multipliers
        param_groups = []
        
        # Fusion parameters
        if self.cfg.train_fusion:
            fusion_lr = base_lr * multipliers['fusion']
            param_groups.append({
                'params': list(self.fusion_proj.parameters()),
                'lr': fusion_lr,
                'name': 'fusion'
            })
            print(f"[LR] Fusion: {fusion_lr:.2e} ({multipliers['fusion']}x base)")
        
        # VLM parameters (ViT blocks)
        if self.cfg.train_vlm:
            if self.cfg.vlm_num_layers_to_train:
                blocks = self._get_vit_layers()
                num_blocks = len(blocks)
                num_to_train = self.cfg.vlm_num_layers_to_train
                start_idx = max(0, num_blocks - num_to_train)
                vision_params = []
                for i, blk in enumerate(blocks):
                    if i >= start_idx:
                        vision_params.extend([p for p in blk.parameters() if p.requires_grad])
                if vision_params:
                    vit_lr = base_lr * multipliers['vit']
                    param_groups.append({
                        'params': vision_params,
                        'lr': vit_lr,
                        'name': 'vit'
                    })
                    print(f"[LR] ViT: {vit_lr:.2e} ({multipliers['vit']}x base)")
            else:
                vit_params = []
                blocks = self._get_vit_layers()
                for blk in blocks:
                    vit_params.extend([p for p in blk.parameters() if p.requires_grad])
                if vit_params:
                    vit_lr = base_lr * multipliers['vit']
                    param_groups.append({
                        'params': vit_params,
                        'lr': vit_lr,
                        'name': 'vit'
                    })
                    print(f"[LR] ViT: {vit_lr:.2e} ({multipliers['vit']}x base)")
            
            # Cross-modal components (non-layerwise)
            crossmodal_params = []
            for name, p in self.vlm.named_parameters():
                if p.requires_grad and any(k in name for k in 
                    ["lm_head", "vision_proj", "mm_projector", "cross_attn", "adapter", "gating"]):
                    crossmodal_params.append(p)
            if crossmodal_params:
                crossmodal_lr = base_lr * multipliers['language']
                param_groups.append({
                    'params': crossmodal_params,
                    'lr': crossmodal_lr,
                    'name': 'crossmodal'
                })
                print(f"[LR] Cross-modal: {crossmodal_lr:.2e} ({multipliers['language']}x base)")
        
        # Language decoder with layer-wise LR decay
        if self.cfg.train_language:
            try:
                lang_layers = self._get_language_layers()
                num_lang_layers = len(lang_layers)
                num_to_train = self.cfg.language_num_layers_to_train or num_lang_layers
                decay = self.cfg.language_lr_decay
                lang_base_lr = base_lr * multipliers['language']
                
                print(f"[LR] Language decoder layer-wise LR (decay={decay}):")
                
                for i, layer in enumerate(lang_layers):
                    # Skip frozen layers
                    if self.cfg.language_num_layers_to_train is not None and i >= num_to_train:
                        continue
                    
                    layer_params = [p for p in layer.parameters() if p.requires_grad]
                    if not layer_params:
                        continue
                    
                    # Layer-wise decay: last layer gets full LR, earlier layers get decayed LR
                    # lr_i = base_lr * decay^(num_trainable - i - 1)
                    num_trainable = min(num_to_train, num_lang_layers)
                    layer_lr = lang_base_lr * (decay ** (num_trainable - i - 1))
                    
                    param_groups.append({
                        'params': layer_params,
                        'lr': layer_lr,
                        'name': f'language_layer_{i}'
                    })
                    
                    if i < 3 or i >= num_trainable - 3:
                        print(f"      Layer {i:2d}: {layer_lr:.2e}")
                    elif i == 3:
                        print(f"      ...")
                        
            except RuntimeError as e:
                print(f"[LR] Warning: Could not setup language LR - {e}")
        
        # YOLO parameters
        if self.cfg.train_yolo:
            layers = list(self.yolo_backbone.children())
            num_layers = len(layers)
            yolo_params = []
            for i, layer in enumerate(layers):
                yolo_params.extend([p for p in layer.parameters() if p.requires_grad])
            if yolo_params:
                yolo_lr = base_lr * multipliers['yolo']
                param_groups.append({
                    'params': yolo_params,
                    'lr': yolo_lr,
                    'name': 'yolo'
                })
                print(f"[LR] YOLO: {yolo_lr:.2e} ({multipliers['yolo']}x base)")
        
        return param_groups
    

    @torch.no_grad()
    def validate(
        self,
        val_data: Dict[str, List[str]],
        batch_size: int = 1,
        enable_thinking: bool = True,
        thinking_budget: int = 1024,
        max_new_tokens: int = 1024
    ) -> Dict[str, float]:
        """Run validation and compute accuracy."""
        # Store original training states
        fusion_training = self.fusion_proj.training
        vlm_training = self.vlm.training
        yolo_training = self.yolo.model.training
        
        # Set eval mode (handles dropout, batchnorm, etc.)
        self.fusion_proj.eval()
        self.vlm.eval()
        self.yolo.model.eval()
        """Run validation and compute accuracy."""
        self._set_trainable(val_mode=True)
        image_paths = val_data['image_paths']
        questions = val_data['question']
        labels = val_data['label']
        num_samples = len(image_paths)
        correct = 0
        total = 0
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_paths = image_paths[i:end_idx]
            batch_questions = questions[i:end_idx]
            batch_labels = labels[i:end_idx]
            # Load images
            images = []
            valid_indices = []
            for j, img_path in enumerate(batch_paths):
                try:
                    if isinstance(img_path, list):
                        img_path = img_path[0]
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"[Val] Error loading {img_path}: {e}")
                    continue
            
            if not images:
                continue
            
            # Process each sample individually (different questions)
            for j, idx in enumerate(valid_indices):
                try:
                    outputs = self.infer(
                        [images[j]], 
                        [batch_paths[idx]], 
                        batch_questions[idx],
                        enable_thinking=enable_thinking,
                        thinking_budget=thinking_budget or self.cfg.thinking_budget,
                        max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                    )
                    pred = extract_yes_no(outputs[0])
                    gt = batch_labels[idx].strip().upper()
                    print(f"[Val] \n ---> Output: {outputs[0]} \n ---> Pred: {pred} \n ---> Label: {gt}")
                    if pred is not None:
                        if pred == gt:
                            correct += 1
                        total += 1
                    else:
                        print(f"[Val] Could not extract YES/NO from: {outputs[0][-100:]}")
                        total += 1  # Count as incorrect
                except Exception as e:
                    print(f"[Val] Error on sample {i + idx}: {e}")
                    continue
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        accuracy = correct / total if total > 0 else 0.0
        self._set_trainable(val_mode=False)
        # Restore original training states at the end (instead of _set_trainable)
        if fusion_training:
            self.fusion_proj.train()
        if vlm_training:
            self.vlm.train()
        if yolo_training:
            self.yolo.model.train()
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }


    def _get_full_state_dict(self) -> Dict[str, Any]:
        """Get complete state dict for all modules (fusion, VLM, YOLO)."""
        return {
            'fusion_state_dict': self.fusion_proj.state_dict(),
            'vlm_state_dict': self.vlm.state_dict(),
            'yolo_state_dict': self.yolo_backbone.state_dict(),
        }

    def _save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        epoch: int,
        config: FusedConfig,
        val_accuracy: Optional[float] = None,
        language_only_phase: bool = False,
        scheduler: Optional[Any] = None,
    ):
        """Save a complete checkpoint with all modules: vision encoder + language decoder (vlm), fusion block, yolo backbone. Optimizer and scheduler states are also saved."""
        checkpoint = {
            'step': global_step,
            'epoch': epoch,
            'gate_value': self.fusion_proj.get_gate_value(),
            'fusion_state_dict': self.fusion_proj.state_dict(),
            'vlm_state_dict': self.vlm.state_dict(),
            'yolo_state_dict': self.yolo_backbone.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'language_only_phase': language_only_phase,
        }
        if val_accuracy is not None:
            checkpoint['val_accuracy'] = val_accuracy
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved all modules to {path}")


    def train_loop(
        self,
        train_data: Dict[str, List[str]],
        num_epochs: int = 1,
        batch_size: int = 4,
        save_every: int = 1000,
        log_every: int = 100,
        gradient_accumulation_steps: int = 1,
        lr_multipliers: Optional[Dict[str, float]] = None,
        val_data: Optional[Dict[str, List[str]]] = None,
        val_every: int = 50,
        val_size: Optional[int] = 1,
        save_dir: str = "./checkpoints",
        checkpoint_title: str = "fusion",
        shuffle_train: bool = True,
        enable_thinking: bool = False,
        language_only_after_steps: Optional[int] = None,
    ):
        """Training loop with optional language-only phase after N steps.
        
        Args:
            language_only_after_steps: If set, freeze all modules except language decoder
                after this many optimizer steps. Useful for two-phase training where
                fusion/vision is trained first, then language model is fine-tuned.
        """
        save_every = save_every or self.cfg.num_steps_per_save
        print(f"====== Training starts! Batch size: {batch_size}, epochs: {num_epochs}, thinking: {enable_thinking} ======")
        if language_only_after_steps is not None:
            print(f"====== Language-only training will begin after step {language_only_after_steps} ======")
        cfg = self.cfg
        os.makedirs(save_dir, exist_ok=True)
        image_paths = train_data['image_paths']
        prompts = train_data['question']
        answers = train_data['answer']
        num_samples = len(image_paths)
        assert len(prompts) == num_samples and len(answers) == num_samples, \
            "All fields must have the same length"
        param_groups = self._setup_optimizer_with_groups(
            base_lr=cfg.lr, 
            lr_multipliers=lr_multipliers
        )
        if not param_groups:
            raise ValueError("No trainable parameters found")
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        num_training_steps = (num_samples // (batch_size * gradient_accumulation_steps)) * num_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in param_groups],
            total_steps=num_training_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0,
        )
        print(f"[Scheduler] OneCycleLR with {num_training_steps} total steps, 10% warmup")
        ema = EMA(self.fusion_proj, decay=0.999)
        print(f"[EMA] Initialized with decay=0.999")
        if self.cfg.train_fusion:
            self.fusion_proj.train()
        if self.cfg.train_vlm:
            self.vlm.train()
        if self.cfg.train_yolo:
            self.yolo.model.train()
        global_step = 0
        running_loss = 0.0
        best_val_acc = 0.0
        train_start_time = time.time()
        last_log_time = train_start_time
        samples_since_log = 0
        language_only_activated = False
        # Initial validation before training
        if val_data is not None:
            print(f"[Step 0] Running initial validation...")
            if val_size is not None and val_size < len(val_data['image_paths']):
                sampled_val_data = {k: v[:val_size] for k, v in val_data.items()}
            else:
                sampled_val_data = val_data
            val_results = self.validate(sampled_val_data, batch_size=1, enable_thinking=True)
            val_acc = val_results['accuracy']
            gate_val = self.fusion_proj.get_gate_value()
            print(f"[Val] Accuracy: {val_acc:.4f} ({val_results['correct']}/{val_results['total']}) | Gate: {gate_val:.4f}")
            best_val_acc = val_acc
            if self.cfg.train_fusion:
                self.fusion_proj.train()
            if self.cfg.train_vlm:
                self.vlm.train()
            if self.cfg.train_yolo:
                self.yolo.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            indices = list(range(num_samples))
            if shuffle_train:
                random.shuffle(indices)
            num_batches = (num_samples + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                batch = {
                    'image_paths': [image_paths[i] for i in batch_indices],
                    'question': [prompts[i] for i in batch_indices],
                    'answer': [answers[i] for i in batch_indices],
                }
                loss, n_samples = self._compute_batch_loss(
                    batch, 
                    accumulate_gradients=True,
                    enable_thinking=enable_thinking,
                    global_step=global_step,
                )
                if loss is None or n_samples == 0:
                    continue
                running_loss += loss.item()
                epoch_loss += loss.item() * n_samples
                epoch_samples += n_samples
                samples_since_log += n_samples
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if global_step % log_every == 0:
                        grad_norms = {}
                        for pg in param_groups:
                            grad_norm = torch.sqrt(
                                sum(p.grad.norm()**2 for p in pg['params'] if p.grad is not None)
                            )
                            grad_norms[pg.get('name', 'unknown')] = f"{grad_norm.item():.4f}"
                        print(f"    Grad norms: {grad_norms}")
                    if cfg.grad_clip_norm is not None:
                        for pg in param_groups:
                            torch.nn.utils.clip_grad_norm_(pg['params'], cfg.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    ema.update()
                    global_step += 1
                    # ===== Language-only transition =====
                    if (language_only_after_steps is not None 
                        and global_step >= language_only_after_steps 
                        and not language_only_activated):
                        print(f"\n{'='*70}")
                        print(f"[Step {global_step}] TRANSITIONING TO LANGUAGE-ONLY TRAINING")
                        print(f"{'='*70}")
                        # Save pre-transition checkpoint with ALL modules
                        transition_path = os.path.join(save_dir, f"{checkpoint_title}_pre_language_only.pt")
                        self._save_checkpoint(
                            path=transition_path,
                            optimizer=optimizer,
                            global_step=global_step,
                            epoch=epoch,
                            config=cfg,
                            language_only_phase=False,
                            scheduler=scheduler,
                        )
                        print(f"[Checkpoint] Pre-transition saved to {transition_path}")
                        # Freeze everything except language
                        self._freeze_for_language_only()
                        # Rebuild optimizer with only language parameters
                        lang_param_groups = self._setup_language_only_optimizer(
                            base_lr=cfg.lr,
                            lr_multipliers=lr_multipliers,
                        )
                        if not lang_param_groups:
                            print("[Warning] No language parameters found, continuing with original optimizer")
                        else:
                            optimizer = torch.optim.AdamW(
                                lang_param_groups,
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                betas=cfg.betas,
                            )
                            remaining_steps = num_training_steps - global_step
                            scheduler = OneCycleLR(
                                optimizer,
                                max_lr=[pg['lr'] for pg in lang_param_groups],
                                total_steps=remaining_steps,
                                pct_start=0.05,
                                anneal_strategy='cos',
                                div_factor=10.0,
                                final_div_factor=100.0,
                            )
                            print(f"[Scheduler] New OneCycleLR for {remaining_steps} remaining steps")
                            # Update param_groups reference for grad norm logging
                            param_groups = lang_param_groups
                        language_only_activated = True
                        print(f"{'='*70}\n")
                    # Logging
                    if global_step % log_every == 0:
                        current_time = time.time()
                        time_elapsed = current_time - last_log_time
                        samples_per_sec = samples_since_log / time_elapsed if time_elapsed > 0 else 0
                        total_time = current_time - train_start_time
                        avg_loss = running_loss / log_every
                        gate_val = self.fusion_proj.get_gate_value()
                        phase = "[Lang-Only] " if language_only_activated else ""
                        print(f"{phase}[Epoch {epoch+1}/{num_epochs}] Step {global_step} | "
                            f"Loss: {avg_loss:.4f} | Gate: {gate_val:.4f} | "
                            f"Samples/s: {samples_per_sec:.2f} | "
                            f"Time: {time_elapsed:.1f}s | Total: {total_time/60:.1f}min")
                        running_loss = 0.0
                        last_log_time = current_time
                        samples_since_log = 0
                    # Validation
                    if val_data is not None and global_step % val_every == 0:
                        print(f"[Step {global_step}] Running validation...")
                        if val_size is not None and val_size < len(val_data['image_paths']):
                            sampled_val_data = {k: v[:val_size] for k, v in val_data.items()}
                        else:
                            sampled_val_data = val_data
                        ema.apply_shadow()
                        val_results = self.validate(sampled_val_data, batch_size=1, enable_thinking=True)
                        ema.restore()
                        val_acc = val_results['accuracy']
                        gate_val = self.fusion_proj.get_gate_value()
                        print(f"[Val] Accuracy: {val_acc:.4f} ({val_results['correct']}/{val_results['total']}) | Gate: {gate_val:.4f}")
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_path = os.path.join(save_dir, f"{checkpoint_title}_best.pt")
                            self._save_checkpoint(
                                path=best_path,
                                optimizer=optimizer,
                                global_step=global_step,
                                epoch=epoch,
                                config=cfg,
                                val_accuracy=val_acc,
                                language_only_phase=language_only_activated,
                                scheduler=scheduler,
                            )
                            print(f"[Best] New best accuracy: {val_acc:.4f}, saved to {best_path}")
                        # Restore training mode
                        if not language_only_activated:
                            if self.cfg.train_fusion:
                                self.fusion_proj.train()
                            if self.cfg.train_vlm:
                                self.vlm.train()
                            if self.cfg.train_yolo:
                                self.yolo.model.train()
                        else:
                            # In language-only mode, only language layers should be in train mode
                            try:
                                lang_layers = self._get_language_layers()
                                num_to_train = self.cfg.language_num_layers_to_train or len(lang_layers)
                                for i, layer in enumerate(lang_layers):
                                    if i < num_to_train:
                                        layer.train()
                            except:
                                pass
                    # Checkpointing
                    if global_step % save_every == 0:
                        ckpt_path = os.path.join(save_dir, f"{checkpoint_title}_step_{global_step}.pt")
                        self._save_checkpoint(
                            path=ckpt_path,
                            optimizer=optimizer,
                            global_step=global_step,
                            epoch=epoch,
                            config=cfg,
                            language_only_phase=language_only_activated,
                            scheduler=scheduler,
                        )
            if epoch_samples > 0:
                avg_epoch_loss = epoch_loss / epoch_samples
                print(f"[Epoch {epoch+1}/{num_epochs}] Complete | Avg Loss: {avg_epoch_loss:.4f}")
                epoch_path = os.path.join(save_dir, f"{checkpoint_title}_epoch_{epoch+1}.pt")
                self._save_checkpoint(
                    path=epoch_path,
                    optimizer=optimizer,
                    global_step=global_step,
                    epoch=epoch + 1,
                    config=cfg,
                    language_only_phase=language_only_activated,
                    scheduler=scheduler,
                )
                print(f"[Epoch Checkpoint] Saved to {epoch_path}")
        # Final validation
        if val_data is not None:
            print("[Final] Running final validation...")
            if val_size is not None and val_size < len(val_data['image_paths']):
                sampled_val_data = {k: v[:val_size] for k, v in val_data.items()}
            else:
                sampled_val_data = val_data
            ema.apply_shadow()
            val_results = self.validate(sampled_val_data, batch_size=1, enable_thinking=True)
            ema.restore()
            gate_val = self.fusion_proj.get_gate_value()
            print(f"[Final Val] Accuracy: {val_results['accuracy']:.4f} ({val_results['correct']}/{val_results['total']}) | Gate: {gate_val:.4f}")
        final_path = os.path.join(save_dir, f"{checkpoint_title}_final.pt")
        self._save_checkpoint(
            path=final_path,
            optimizer=optimizer,
            global_step=global_step,
            epoch=num_epochs,
            config=cfg,
            language_only_phase=language_only_activated,
            scheduler=scheduler,
        )
        print(f"[Training Complete] Final checkpoint: {final_path}")
        return {'best_val_acc': best_val_acc}


    def _get_trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Collect state dict for all trainable parameters."""
        state = {'fusion': self.fusion_proj.state_dict()}
        if self.cfg.train_vlm:
            state['vlm'] = {k: v for k, v in self.vlm.state_dict().items() 
                            if any(p.requires_grad for n, p in self.vlm.named_parameters() if n == k.replace('.', ''))}
        if self.cfg.train_yolo:
            state['yolo'] = {k: v for k, v in self.yolo_backbone.state_dict().items()}
        return state

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_scheduler: bool = False,
        scheduler: Optional[Any] = None,
        strict: bool = True,
        map_location: Optional[str] = None,
        load_vlm: bool = True,
        load_yolo: bool = True,
        load_fusion: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint with all modules (fusion, VLM, YOLO).
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            optimizer: Optimizer instance (required if load_optimizer=True)
            load_scheduler: Whether to load scheduler state
            scheduler: Scheduler instance (required if load_scheduler=True)
            strict: Whether to strictly enforce state_dict keys match
            map_location: Device to map tensors to
            load_vlm: Whether to load VLM (vision encoder + language decoder) weights
            load_yolo: Whether to load YOLO weights
            load_fusion: Whether to load fusion block weights
        Returns:
            Dict with metadata (step, epoch, val_accuracy, language_only_phase)
        """
        map_location = map_location or self.cfg.device
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        
        # Load fusion
        if load_fusion and 'fusion_state_dict' in ckpt:
            self.fusion_proj.load_state_dict(ckpt['fusion_state_dict'], strict=strict)
            print(f"[Checkpoint] Fusion loaded ({len(ckpt['fusion_state_dict'])} keys)")
        elif load_fusion:
            print("[Checkpoint] Warning: No fusion_state_dict in checkpoint")
        
        # Load VLM (vision encoder + language decoder)
        if load_vlm and 'vlm_state_dict' in ckpt:
            missing, unexpected = self.vlm.load_state_dict(ckpt['vlm_state_dict'], strict=strict)
            print(f"[Checkpoint] VLM loaded ({len(ckpt['vlm_state_dict'])} keys)")
            if missing:
                print(f"[Checkpoint] VLM missing keys: {len(missing)}")
            if unexpected:
                print(f"[Checkpoint] VLM unexpected keys: {len(unexpected)}")
        elif load_vlm:
            print("[Checkpoint] Warning: No vlm_state_dict in checkpoint")
        
        # Load YOLO
        if load_yolo and 'yolo_state_dict' in ckpt:
            missing, unexpected = self.yolo_backbone.load_state_dict(ckpt['yolo_state_dict'], strict=strict)
            print(f"[Checkpoint] YOLO loaded ({len(ckpt['yolo_state_dict'])} keys)")
            if missing:
                print(f"[Checkpoint] YOLO missing keys: {len(missing)}")
            if unexpected:
                print(f"[Checkpoint] YOLO unexpected keys: {len(unexpected)}")
        elif load_yolo:
            print("[Checkpoint] Warning: No yolo_state_dict in checkpoint")
        
        # Load optimizer
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("[Checkpoint] Optimizer loaded")
        
        # Load scheduler
        if load_scheduler and scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("[Checkpoint] Scheduler loaded")
        
        return {
            'step': ckpt.get('step', 0),
            'epoch': ckpt.get('epoch', 0),
            'val_accuracy': ckpt.get('val_accuracy'),
            'gate_value': ckpt.get('gate_value'),
            'language_only_phase': ckpt.get('language_only_phase', False),
        }



    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        cfg: Optional["FusedConfig"] = None,
        override_device: Optional[str] = None,
        load_vlm: bool = True,
        load_yolo: bool = True,
        load_fusion: bool = True,
    ) -> Tuple["FusedOvis2d5Yolo", Dict[str, Any]]:
        """Initialize model and load checkpoint in one call.
        
        Args:
            checkpoint_path: Path to checkpoint
            cfg: Config override. If None, uses config from checkpoint
            override_device: Override device for loading
            load_vlm: Whether to load VLM weights
            load_yolo: Whether to load YOLO weights
            load_fusion: Whether to load fusion block weights
            
        Returns:
            Tuple of (model, metadata)
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if cfg is None:
            cfg = ckpt.get('config')
            if cfg is None:
                raise ValueError("No config in checkpoint and none provided")
        if override_device:
            cfg.device = override_device
        model = cls(cfg)
        metadata = model.load_checkpoint(
            checkpoint_path, 
            map_location=cfg.device,
            load_vlm=load_vlm,
            load_yolo=load_yolo,
            load_fusion=load_fusion,
        )
        return model, metadata
