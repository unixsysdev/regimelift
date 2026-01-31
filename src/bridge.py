"""
He-LMAS Bridge: The Heterogeneous Manifold Projector

This is the core component of He-LMAS. The Bridge projects the Teacher's
KV cache into the Student's geometry, enabling reasoning transfer across
models with different architectures.

The Bridge handles:
1. Layer mapping: 36 Teacher layers → 28 Student layers (strided selection)
2. Dimension projection: Through learned projectors (shallow or deep)
3. RoPE handling: De-rotate Teacher, re-rotate Student (full mode)

Key insight: Both Qwen3-8B and Qwen3-1.7B have 8 KV heads, so we have
perfect 1:1 head mapping. We only need to project the hidden dimensions.

V2 Changes:
- Added DeepProjector for more learning capacity
- Configurable projector_depth (1=shallow, 2+=deep)
- Full RoPE de-rotation/re-rotation support
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Literal
import math

from .rope_utils import RoPEHandler
from .kv_cache_utils import KVCache


class ShallowProjector(nn.Module):
    """
    Simple linear projector: Linear(in_dim, out_dim)
    
    ~32K parameters per projector.
    Fast but limited learning capacity.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_strategy: str = "identity"
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self._init_weights(init_strategy, in_dim, out_dim)
    
    def _init_weights(self, strategy: str, in_dim: int, out_dim: int):
        with torch.no_grad():
            if strategy == "identity":
                min_dim = min(in_dim, out_dim)
                self.proj.weight.zero_()
                self.proj.weight[:min_dim, :min_dim] = torch.eye(min_dim)
            else:
                nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DeepProjector(nn.Module):
    """
    Deep projector with hidden layers: Linear → GELU → Linear [→ GELU → Linear ...]
    
    Much more learning capacity for complex manifold alignment.
    
    Architecture:
        in_dim → hidden_dim → ... → out_dim
        
    For depth=2 with expansion=2:
        128 → 256 → 128 (~66K params, 2x the shallow projector)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        depth: int = 2,
        init_strategy: str = "identity"
    ):
        super().__init__()
        
        # Default hidden dim: 2x input for expansion
        if hidden_dim is None:
            hidden_dim = in_dim * 2
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        # Build layers
        layers = []
        current_dim = in_dim
        
        for i in range(depth):
            is_last = (i == depth - 1)
            next_dim = out_dim if is_last else hidden_dim
            
            layers.append(nn.Linear(current_dim, next_dim, bias=False))
            if not is_last:
                layers.append(nn.GELU())
            
            current_dim = next_dim
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize for near-identity at start
        self._init_weights(init_strategy)
    
    def _init_weights(self, strategy: str):
        if strategy == "identity":
            # Initialize to approximate identity
            # Scale down intermediate layers, identity-like last layer
            with torch.no_grad():
                for i, module in enumerate(self.layers):
                    if isinstance(module, nn.Linear):
                        is_last = (i == len(self.layers) - 1)
                        if is_last:
                            # Last layer: identity-like
                            min_dim = min(module.in_features, module.out_features)
                            module.weight.zero_()
                            module.weight[:min_dim, :min_dim] = torch.eye(min_dim)
                        else:
                            # Intermediate: small random for breaking symmetry
                            nn.init.xavier_uniform_(module.weight, gain=0.1)
        else:
            for module in self.layers:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


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
        projector_depth: 1=shallow linear, 2+=deep with hidden layers
        hidden_expansion: Hidden dim = head_dim * hidden_expansion (for deep)
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
        teacher_rope_base: float = 1000000.0,
        student_rope_base: float = 1000000.0,
        max_seq_len: int = 32768,
        projector_depth: int = 1,
        hidden_expansion: float = 2.0
    ):
        super().__init__()
        
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.teacher_head_dim = teacher_head_dim
        self.student_head_dim = student_head_dim
        self.num_kv_heads = num_kv_heads
        self.rope_handling = rope_handling
        self.per_layer = per_layer
        self.projector_depth = projector_depth
        
        # Compute layer mapping: which Teacher layer maps to which Student layer
        # Using uniform stride: layer_map[i] = round(i * teacher_layers / student_layers)
        self.register_buffer(
            'layer_map',
            torch.linspace(0, teacher_layers - 1, student_layers).round().long()
        )
        
        # Hidden dimension for deep projectors
        hidden_dim = int(teacher_head_dim * hidden_expansion)
        
        # Choose projector class
        ProjectorClass = DeepProjector if projector_depth > 1 else ShallowProjector
        
        # Initialize projectors
        if per_layer:
            # Separate projector for each Student layer
            self.k_proj = nn.ModuleList([
                ProjectorClass(
                    teacher_head_dim, student_head_dim,
                    hidden_dim=hidden_dim if projector_depth > 1 else None,
                    depth=projector_depth if projector_depth > 1 else None,
                    init_strategy=init_strategy
                ) if projector_depth > 1 else
                ShallowProjector(teacher_head_dim, student_head_dim, init_strategy)
                for _ in range(student_layers)
            ])
            self.v_proj = nn.ModuleList([
                ProjectorClass(
                    teacher_head_dim, student_head_dim,
                    hidden_dim=hidden_dim if projector_depth > 1 else None,
                    depth=projector_depth if projector_depth > 1 else None,
                    init_strategy=init_strategy
                ) if projector_depth > 1 else
                ShallowProjector(teacher_head_dim, student_head_dim, init_strategy)
                for _ in range(student_layers)
            ])
        else:
            # Shared projector across all layers
            if projector_depth > 1:
                self.k_proj = DeepProjector(
                    teacher_head_dim, student_head_dim,
                    hidden_dim=hidden_dim, depth=projector_depth,
                    init_strategy=init_strategy
                )
                self.v_proj = DeepProjector(
                    teacher_head_dim, student_head_dim,
                    hidden_dim=hidden_dim, depth=projector_depth,
                    init_strategy=init_strategy
                )
            else:
                self.k_proj = ShallowProjector(teacher_head_dim, student_head_dim, init_strategy)
                self.v_proj = ShallowProjector(teacher_head_dim, student_head_dim, init_strategy)
        
        # RoPE handlers (always created for full mode)
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
            print(f"  → Full RoPE handling enabled")
            print(f"    Teacher θ={teacher_rope_base:,.0f}, Student θ={student_rope_base:,.0f}")
        else:
            self.teacher_rope = None
            self.student_rope = None
        
        # Log architecture
        projector_type = f"Deep (depth={projector_depth}, hidden={hidden_dim})" if projector_depth > 1 else "Shallow"
        print(f"  → Bridge: {projector_type} projectors")
        print(f"  → Layer mapping: {teacher_layers} → {student_layers} (stride={teacher_layers/student_layers:.2f})")
    
    def forward(
        self,
        teacher_cache: KVCache,
        position_ids: Optional[torch.Tensor] = None
    ) -> KVCache:
        """
        Project Teacher's KV cache to Student's geometry.
        
        Full RoPE flow:
        1. De-rotate Teacher's K using Teacher's RoPE (undo position encoding)
        2. Project K and V through learned projectors
        3. Re-rotate Student's K using Student's RoPE (apply new position encoding)
        
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
            
            # Determine position IDs if not provided
            if position_ids is None:
                seq_len = k_t.shape[2]
                position_ids_local = torch.arange(seq_len, device=proj_device)
            else:
                position_ids_local = position_ids.to(proj_device)
            
            # === STEP 1: De-rotate Teacher's K (Full RoPE mode) ===
            # This removes the Teacher's positional encoding, exposing
            # the "raw" semantic features for projection
            if self.rope_handling == "full" and self.teacher_rope is not None:
                self.teacher_rope.to(proj_device)
                k_t = self.teacher_rope.derotate(k_t, position_ids_local)
                # Note: V is NOT rotated in standard RoPE
            
            # === STEP 2: Project dimensions through learned transform ===
            # k_t: [batch, heads, seq, head_dim_t] → [batch, heads, seq, head_dim_s]
            if self.per_layer:
                k_s = self.k_proj[i](k_t)
                v_s = self.v_proj[i](v_t)
            else:
                k_s = self.k_proj(k_t)
                v_s = self.v_proj(v_t)
            
            # === STEP 3: Re-rotate with Student's RoPE (Full mode) ===
            # This applies the Student's positional encoding to the
            # projected features, so they integrate correctly with
            # the Student's own attention computation
            if self.rope_handling == "full" and self.student_rope is not None:
                self.student_rope.to(proj_device)
                k_s = self.student_rope.rotate(k_s, position_ids_local)
            
            student_cache.append((k_s, v_s))
        
        return student_cache
    
    def _get_proj_device(self, layer_idx: int) -> torch.device:
        """Get the device of projector weights."""
        if self.per_layer:
            # DeepProjector has .layers, ShallowProjector has .proj
            proj = self.k_proj[layer_idx]
            if hasattr(proj, 'proj'):
                return proj.proj.weight.device
            else:
                return proj.layers[0].weight.device
        else:
            if hasattr(self.k_proj, 'proj'):
                return self.k_proj.proj.weight.device
            else:
                return self.k_proj.layers[0].weight.device
    
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
    bridge = config.get('bridge', {})
    
    return HeLMAS_Bridge(
        teacher_layers=teacher['layers'],
        student_layers=student['layers'],
        teacher_head_dim=teacher['head_dim'],
        student_head_dim=student['head_dim'],
        num_kv_heads=teacher['num_kv_heads'],  # Assumes both match
        rope_handling=bridge.get('rope_handling', 'naive'),
        per_layer=bridge.get('per_layer_projectors', True),
        init_strategy=bridge.get('init_strategy', 'identity'),
        teacher_rope_base=bridge.get('teacher_rope_base', 1000000.0),
        student_rope_base=bridge.get('student_rope_base', 1000000.0),
        projector_depth=bridge.get('projector_depth', 1),
        hidden_expansion=bridge.get('hidden_expansion', 2.0)
    )
