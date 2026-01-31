"""
He-LMAS Bridge: The Heterogeneous Manifold Projector

This is the core component of He-LMAS. The Bridge projects the Teacher's
KV cache into the Student's geometry, enabling reasoning transfer across
models with different architectures.

The Bridge handles:
1. Layer mapping: 36 Teacher layers → 28 Student layers (strided selection)
2. Dimension projection: 4096 Teacher dim → 2048 Student dim
3. RoPE handling: De-rotate Teacher, re-rotate Student (optional)

Key insight: Both Qwen3-8B and Qwen3-1.7B have 8 KV heads, so we have
perfect 1:1 head mapping. We only need to project the hidden dimensions.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Literal
import math

from .rope_utils import RoPEHandler
from .kv_cache_utils import KVCache


class HeLMAS_Bridge(nn.Module):
    """
    The Heterogeneous Latent Manifold Alignment Bridge.
    
    Projects KV cache from a larger Teacher model to a smaller Student model.
    
    Architecture (Qwen3-8B → Qwen3-1.7B):
        - Teacher: 36 layers, 8 KV heads, head_dim=128
        - Student: 28 layers, 8 KV heads, head_dim=128
        - Layer mapping: Uniform stride (select 28 from 36)
        - Head mapping: 1:1 (both have 8 KV heads)
        - Dimension: May need projection if head_dim differs
    
    Args:
        teacher_layers: Number of layers in Teacher model
        student_layers: Number of layers in Student model
        teacher_head_dim: Head dimension in Teacher
        student_head_dim: Head dimension in Student
        num_kv_heads: Number of KV heads (must match between models)
        rope_handling: "naive" (skip rotation) or "full" (de-rotate/re-rotate)
        per_layer: If True, use separate projectors per layer
        init_strategy: "identity" or "random"
    """
    
    def __init__(
        self,
        teacher_layers: int = 36,
        student_layers: int = 28,
        teacher_head_dim: int = 128,
        student_head_dim: int = 128,
        num_kv_heads: int = 8,
        rope_handling: Literal["naive", "full"] = "naive",
        per_layer: bool = True,
        init_strategy: Literal["identity", "random"] = "identity",
        teacher_rope_base: float = 10000.0,
        student_rope_base: float = 10000.0,
        max_seq_len: int = 32768
    ):
        super().__init__()
        
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.teacher_head_dim = teacher_head_dim
        self.student_head_dim = student_head_dim
        self.num_kv_heads = num_kv_heads
        self.rope_handling = rope_handling
        self.per_layer = per_layer
        
        # Compute layer mapping: which Teacher layer maps to which Student layer
        # Using uniform stride: layer_map[i] = round(i * teacher_layers / student_layers)
        self.register_buffer(
            'layer_map',
            torch.linspace(0, teacher_layers - 1, student_layers).round().long()
        )
        
        # Initialize projectors
        if per_layer:
            # Separate projector for each Student layer
            self.k_proj = nn.ModuleList([
                nn.Linear(teacher_head_dim, student_head_dim, bias=False)
                for _ in range(student_layers)
            ])
            self.v_proj = nn.ModuleList([
                nn.Linear(teacher_head_dim, student_head_dim, bias=False)
                for _ in range(student_layers)
            ])
        else:
            # Shared projector across all layers
            self.k_proj = nn.Linear(teacher_head_dim, student_head_dim, bias=False)
            self.v_proj = nn.Linear(teacher_head_dim, student_head_dim, bias=False)
        
        # Initialize weights
        self._init_weights(init_strategy)
        
        # RoPE handlers (only used if rope_handling == "full")
        if rope_handling == "full":
            self.teacher_rope = RoPEHandler(
                head_dim=teacher_head_dim,
                max_seq_len=max_seq_len,
                base=teacher_rope_base
            )
            self.student_rope = RoPEHandler(
                head_dim=student_head_dim,
                max_seq_len=max_seq_len,
                base=student_rope_base
            )
        else:
            self.teacher_rope = None
            self.student_rope = None
    
    def _init_weights(self, strategy: str):
        """Initialize projection weights."""
        modules = []
        if self.per_layer:
            modules.extend(self.k_proj)
            modules.extend(self.v_proj)
        else:
            modules = [self.k_proj, self.v_proj]
        
        for module in modules:
            if strategy == "identity":
                # Initialize close to identity for stability
                # If dimensions match, this is exact identity
                # If not, it's a truncated/padded identity
                with torch.no_grad():
                    min_dim = min(self.teacher_head_dim, self.student_head_dim)
                    module.weight.zero_()
                    module.weight[:min_dim, :min_dim] = torch.eye(min_dim)
            elif strategy == "random":
                # Standard Xavier initialization
                nn.init.xavier_uniform_(module.weight)
            else:
                raise ValueError(f"Unknown init strategy: {strategy}")
    
    def forward(
        self,
        teacher_cache: KVCache,
        position_ids: Optional[torch.Tensor] = None
    ) -> KVCache:
        """
        Project Teacher's KV cache to Student's geometry.
        
        Args:
            teacher_cache: List of (K, V) tuples from Teacher
                          Each K/V: [batch, num_kv_heads, seq_len, head_dim]
            position_ids: Optional position indices for RoPE
            
        Returns:
            Projected KV cache for Student (same format, different shapes)
        """
        student_cache: KVCache = []
        
        for i in range(self.student_layers):
            # Get the corresponding Teacher layer
            teacher_idx = self.layer_map[i].item()
            k_t, v_t = teacher_cache[teacher_idx]
            
            # Move to same device as projector weights
            proj_device = self._get_proj_device(i)
            k_t = k_t.to(proj_device)
            v_t = v_t.to(proj_device)
            
            # Step 1: De-rotate (remove Teacher's RoPE)
            if self.rope_handling == "full" and self.teacher_rope is not None:
                self.teacher_rope.to(proj_device)
                k_t = self.teacher_rope.derotate(k_t, position_ids)
                # Note: Values are typically not rotated in standard RoPE
            
            # Step 2: Project dimensions
            # k_t: [batch, heads, seq, head_dim_t] → [batch, heads, seq, head_dim_s]
            if self.per_layer:
                k_s = self.k_proj[i](k_t)
                v_s = self.v_proj[i](v_t)
            else:
                k_s = self.k_proj(k_t)
                v_s = self.v_proj(v_t)
            
            # Step 3: Re-rotate (apply Student's RoPE)
            if self.rope_handling == "full" and self.student_rope is not None:
                self.student_rope.to(proj_device)
                k_s = self.student_rope.rotate(k_s, position_ids)
            
            student_cache.append((k_s, v_s))
        
        return student_cache
    
    def _get_proj_device(self, layer_idx: int) -> torch.device:
        """Get the device of projector weights."""
        if self.per_layer:
            return self.k_proj[layer_idx].weight.device
        else:
            return self.k_proj.weight.device
    
    def get_layer_mapping_info(self) -> dict:
        """Get information about the layer mapping."""
        mapping = self.layer_map.tolist()
        
        # Find which Teacher layers are used and which are skipped
        used = set(mapping)
        skipped = set(range(self.teacher_layers)) - used
        
        return {
            "teacher_layers": self.teacher_layers,
            "student_layers": self.student_layers,
            "ratio": self.teacher_layers / self.student_layers,
            "mapping": mapping,
            "used_teacher_layers": sorted(used),
            "skipped_teacher_layers": sorted(skipped),
            "num_skipped": len(skipped)
        }
    
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HeLMAS_Bridge_GQA(HeLMAS_Bridge):
    """
    Extended Bridge for models with different GQA (Grouped Query Attention) configurations.
    
    If Teacher and Student have different numbers of KV heads, we need to
    handle the head dimension differently. This class adds head interpolation.
    
    Note: For Qwen3-8B → Qwen3-1.7B, both have 8 KV heads, so the base
    HeLMAS_Bridge is sufficient. This class is for future extensions.
    """
    
    def __init__(
        self,
        teacher_layers: int = 36,
        student_layers: int = 28,
        teacher_kv_heads: int = 8,
        student_kv_heads: int = 8,
        teacher_head_dim: int = 128,
        student_head_dim: int = 128,
        **kwargs
    ):
        if teacher_kv_heads != student_kv_heads:
            raise NotImplementedError(
                f"Head count mismatch ({teacher_kv_heads} → {student_kv_heads}) "
                "not yet supported. Use models with matching KV heads."
            )
        
        super().__init__(
            teacher_layers=teacher_layers,
            student_layers=student_layers,
            teacher_head_dim=teacher_head_dim,
            student_head_dim=student_head_dim,
            num_kv_heads=teacher_kv_heads,
            **kwargs
        )


def create_bridge_from_config(config: dict) -> HeLMAS_Bridge:
    """
    Factory function to create a Bridge from configuration.
    
    Args:
        config: Configuration dictionary with 'models' and 'bridge' sections
        
    Returns:
        Initialized HeLMAS_Bridge
    """
    teacher = config['models']['teacher']
    student = config['models']['student']
    bridge = config['bridge']
    
    return HeLMAS_Bridge(
        teacher_layers=teacher['layers'],
        student_layers=student['layers'],
        teacher_head_dim=teacher['head_dim'],
        student_head_dim=student['head_dim'],
        num_kv_heads=teacher['num_kv_heads'],  # Assumes both match
        rope_handling=bridge.get('rope_handling', 'naive'),
        per_layer=bridge.get('per_layer_projectors', True),
        init_strategy=bridge.get('init_strategy', 'identity')
    )
