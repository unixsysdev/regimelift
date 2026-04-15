"""
KV Cache Utilities for He-LMAS

This module provides functions for extracting, manipulating, and injecting
KV caches from/into HuggingFace transformer models.

The KV cache is the "fossilized record" of the model's reasoning. By
extracting it from the Teacher and injecting it into the Student, we
transfer the Teacher's "thought process" without text-based handoffs.
"""

import torch
from typing import List, Tuple, Optional, Union
from transformers import PreTrainedTokenizer


# Type alias for KV cache: list of (K, V) tuples per layer
# Each K/V has shape: [batch, num_kv_heads, seq_len, head_dim]
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


def extract_kv_cache(model_output) -> KVCache:
    """
    Extract KV cache from HuggingFace model output.
    
    Args:
        model_output: Output from model.generate() or model() with
                      return_dict_in_generate=True or output_hidden_states=True
                      
    Returns:
        List of (K, V) tuples, one per layer
    """
    if hasattr(model_output, 'past_key_values'):
        past = model_output.past_key_values
    elif isinstance(model_output, tuple) and len(model_output) > 1:
        # Some models return (logits, past_key_values, ...)
        past = model_output[1]
    else:
        raise ValueError(
            "Cannot extract KV cache from model output. "
            "Ensure you pass return_dict_in_generate=True or use_cache=True"
        )
    
    if past is None:
        raise ValueError("Model did not return KV cache. Check model configuration.")
    
    return list(past)


def kv_cache_info(cache: KVCache) -> dict:
    """
    Get information about a KV cache.
    
    Args:
        cache: KV cache to inspect
        
    Returns:
        Dictionary with cache statistics
    """
    if not cache:
        return {"layers": 0, "empty": True}
    
    k_sample, v_sample = cache[0]
    
    return {
        "layers": len(cache),
        "batch_size": k_sample.shape[0],
        "num_kv_heads": k_sample.shape[1],
        "seq_len": k_sample.shape[2],
        "head_dim": k_sample.shape[3],
        "k_dtype": k_sample.dtype,
        "v_dtype": v_sample.dtype,
        "k_device": str(k_sample.device),
        "total_elements": sum(k.numel() + v.numel() for k, v in cache),
        "memory_mb": sum(
            (k.element_size() * k.numel() + v.element_size() * v.numel())
            for k, v in cache
        ) / (1024 * 1024)
    }


def slice_cache_at_position(cache: KVCache, end_pos: int) -> KVCache:
    """
    Slice KV cache to include only positions [0, end_pos).
    
    Useful for cutting the cache at a specific token (e.g., </think>).
    
    Args:
        cache: Full KV cache
        end_pos: Position to cut at (exclusive)
        
    Returns:
        Sliced KV cache
    """
    return [
        (k[:, :, :end_pos, :], v[:, :, :end_pos, :])
        for k, v in cache
    ]


def find_token_position(
    token_ids: torch.Tensor,
    target_token_id: int,
    from_end: bool = True
) -> Optional[int]:
    """
    Find the position of a target token in the sequence.
    
    Args:
        token_ids: Tensor of token IDs [batch, seq_len] or [seq_len]
        target_token_id: Token ID to search for
        from_end: If True, find last occurrence; if False, find first
        
    Returns:
        Position index, or None if not found
    """
    if token_ids.dim() == 2:
        token_ids = token_ids[0]  # Take first batch
    
    matches = (token_ids == target_token_id).nonzero(as_tuple=True)[0]
    
    if len(matches) == 0:
        return None
    
    return matches[-1].item() if from_end else matches[0].item()


def slice_cache_at_token(
    cache: KVCache,
    token_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    end_token: str = "</think>",
    include_end_token: bool = True
) -> Tuple[KVCache, int]:
    """
    Slice KV cache at a specific token boundary.
    
    This is the key function for He-LMAS: we want to capture the Teacher's
    thinking (up to </think>) and inject only that portion.
    
    Args:
        cache: Full KV cache from Teacher
        token_ids: Generated token IDs from Teacher
        tokenizer: Tokenizer to encode the end_token
        end_token: Token string to cut at
        include_end_token: Whether to include the end token in the slice
        
    Returns:
        Tuple of (sliced_cache, cut_position)
    """
    # Encode the end token to get its ID
    end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)
    
    if not end_token_ids:
        raise ValueError(f"Cannot encode end token: {end_token}")
    
    # For multi-token end markers, search for the last token
    target_id = end_token_ids[-1]
    
    position = find_token_position(token_ids, target_id, from_end=True)
    
    if position is None:
        # End token not found, return full cache
        seq_len = cache[0][0].shape[2]
        return cache, seq_len
    
    cut_pos = position + 1 if include_end_token else position
    
    return slice_cache_at_position(cache, cut_pos), cut_pos


def move_cache_to_device(cache: KVCache, device: torch.device) -> KVCache:
    """Move all tensors in cache to specified device."""
    return [
        (k.to(device), v.to(device))
        for k, v in cache
    ]


def clone_cache(cache: KVCache) -> KVCache:
    """Create a deep copy of the KV cache."""
    return [
        (k.clone(), v.clone())
        for k, v in cache
    ]


def concatenate_caches(cache1: KVCache, cache2: KVCache) -> KVCache:
    """
    Concatenate two KV caches along the sequence dimension.
    
    Useful for appending new tokens to an existing cache.
    
    Args:
        cache1: First cache (earlier positions)
        cache2: Second cache (later positions)
        
    Returns:
        Concatenated cache
    """
    assert len(cache1) == len(cache2), "Caches must have same number of layers"
    
    return [
        (
            torch.cat([k1, k2], dim=2),
            torch.cat([v1, v2], dim=2)
        )
        for (k1, v1), (k2, v2) in zip(cache1, cache2)
    ]


def prepare_injection_input(
    tokenizer: PreTrainedTokenizer,
    handoff_text: str = "\nAnswer:",
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Prepare the handoff tokens to feed after cache injection.
    
    After injecting the Teacher's cache, we feed these tokens to trigger
    the Student to start answering.
    
    Args:
        tokenizer: Tokenizer (should be same for Teacher and Student)
        handoff_text: Text to append after injection
        device: Target device
        
    Returns:
        Token IDs tensor [1, seq_len]
    """
    token_ids = tokenizer.encode(handoff_text, add_special_tokens=False)
    tensor = torch.tensor([token_ids], dtype=torch.long)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor
