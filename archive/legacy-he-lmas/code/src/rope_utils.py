"""
RoPE (Rotary Position Embedding) Utilities for He-LMAS

This module provides functions for applying and inverting RoPE transformations,
which is critical for the heterogeneous KV cache projection between models
with potentially different RoPE configurations.

Key insight: RoPE rotates K/Q vectors based on position. When projecting
KV cache from Teacher to Student, we must:
1. De-rotate using Teacher's frequencies (expose raw semantic features)
2. Project dimensions
3. Re-rotate using Student's frequencies
"""

import torch
import math
from typing import Tuple, Optional


def get_rope_frequencies(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate the sin/cos frequency tables for RoPE.
    
    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length to precompute
        base: RoPE base frequency (check model config for rope_theta)
        device: Target device for tensors
        
    Returns:
        Tuple of (cos, sin) tensors with shape [max_seq_len, dim]
    """
    assert dim % 2 == 0, f"Head dimension must be even, got {dim}"
    
    # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Position indices
    positions = torch.arange(max_seq_len, device=device).float()
    
    # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
    freqs = torch.outer(positions, inv_freq)
    
    # Duplicate for pairs: [seq_len, dim]
    freqs = torch.cat([freqs, freqs], dim=-1)
    
    cos = freqs.cos()
    sin = freqs.sin()
    
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    For a vector [x1, x2, x3, x4], returns [-x3, -x4, x1, x2].
    
    This is the standard RoPE rotation helper.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply RoPE (forward rotation) to input tensor.
    
    Args:
        x: Input tensor of shape [..., seq_len, head_dim]
        cos: Cosine frequencies [max_seq_len, head_dim] or [seq_len, head_dim]
        sin: Sine frequencies [max_seq_len, head_dim] or [seq_len, head_dim]
        position_ids: Optional position indices [seq_len] to select from cos/sin
        
    Returns:
        Rotated tensor with same shape as input
    """
    seq_len = x.shape[-2]
    
    if position_ids is not None:
        # Select specific positions
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        # Use first seq_len positions
        cos = cos[:seq_len]
        sin = sin[:seq_len]
    
    # Broadcast to match input dimensions
    # x: [batch, heads, seq, dim] or [batch, seq, heads, dim]
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    
    # Apply rotation: x * cos + rotate_half(x) * sin
    return (x * cos) + (rotate_half(x) * sin)


def apply_inverse_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply inverse RoPE (de-rotation) to input tensor.
    
    This is the critical operation for He-LMAS: we need to remove the
    Teacher's positional encoding before projecting, then re-apply
    the Student's encoding.
    
    The inverse of rotation by angle θ is rotation by -θ, which means
    we use cos (unchanged, since cos(-θ) = cos(θ)) and -sin.
    
    Args:
        x: Input tensor of shape [..., seq_len, head_dim]
        cos: Cosine frequencies [max_seq_len, head_dim]
        sin: Sine frequencies [max_seq_len, head_dim]
        position_ids: Optional position indices
        
    Returns:
        De-rotated tensor with same shape as input
    """
    # Inverse rotation: use -sin instead of sin
    return apply_rotary_pos_emb(x, cos, -sin, position_ids)


class RoPEHandler:
    """
    Handles RoPE transformations for a specific model configuration.
    
    Usage:
        handler = RoPEHandler(head_dim=128, max_seq_len=32768, base=10000.0)
        rotated = handler.rotate(keys, position_ids)
        derotated = handler.derotate(keys, position_ids)
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = device
        
        # Precompute frequency tables
        self.cos, self.sin = get_rope_frequencies(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=base,
            device=device
        )
    
    def to(self, device: torch.device) -> "RoPEHandler":
        """Move frequency tables to device."""
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)
        self.device = device
        return self
    
    def rotate(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply forward RoPE rotation."""
        return apply_rotary_pos_emb(x, self.cos, self.sin, position_ids)
    
    def derotate(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply inverse RoPE (de-rotation)."""
        return apply_inverse_rotary_pos_emb(x, self.cos, self.sin, position_ids)


def verify_rope_compatibility(
    teacher_base: float,
    student_base: float,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if Teacher and Student have compatible RoPE configurations.
    
    If they differ, the Bridge must implement full de-rotate/re-rotate.
    If they match, we can skip RoPE handling (naive projection).
    
    Args:
        teacher_base: Teacher's rope_theta value
        student_base: Student's rope_theta value
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if RoPE bases match (can skip rotation handling)
    """
    return abs(teacher_base - student_base) < tolerance
