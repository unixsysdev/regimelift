"""
Tests for the RoPE utilities.

Run with: pytest tests/test_rope_utils.py -v
"""

import pytest
import torch
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rope_utils import (
    get_rope_frequencies,
    rotate_half,
    apply_rotary_pos_emb,
    apply_inverse_rotary_pos_emb,
    RoPEHandler,
    verify_rope_compatibility
)


class TestGetRopeFrequencies:
    """Test frequency table generation."""
    
    def test_output_shapes(self):
        """Test that output shapes are correct."""
        cos, sin = get_rope_frequencies(dim=128, max_seq_len=1000)
        
        assert cos.shape == (1000, 128)
        assert sin.shape == (1000, 128)
    
    def test_first_position_values(self):
        """Test that position 0 has correct values."""
        cos, sin = get_rope_frequencies(dim=128, max_seq_len=100)
        
        # At position 0, all rotations should be 0
        # cos(0) = 1, sin(0) = 0
        assert torch.allclose(cos[0], torch.ones(128))
        assert torch.allclose(sin[0], torch.zeros(128), atol=1e-6)
    
    def test_different_bases(self):
        """Test that different bases produce different frequencies."""
        cos1, sin1 = get_rope_frequencies(dim=128, max_seq_len=100, base=10000.0)
        cos2, sin2 = get_rope_frequencies(dim=128, max_seq_len=100, base=1000000.0)
        
        # With larger base, frequencies should be slower
        # At position 50, base 10000 should have rotated more than base 1000000
        diff1 = (cos1[50] - cos1[0]).abs().sum()
        diff2 = (cos2[50] - cos2[0]).abs().sum()
        
        assert diff1 > diff2  # Smaller base = faster rotation
    
    def test_device_placement(self):
        """Test that tensors are created on correct device."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            cos, sin = get_rope_frequencies(dim=128, max_seq_len=100, device=device)
            
            assert cos.device.type == "cuda"
            assert sin.device.type == "cuda"


class TestRotateHalf:
    """Test the rotate_half helper function."""
    
    def test_rotation_pattern(self):
        """Test that rotation pattern is correct."""
        # [x1, x2, x3, x4] → [-x3, -x4, x1, x2]
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        rotated = rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        
        torch.testing.assert_close(rotated, expected)
    
    def test_batch_rotation(self):
        """Test rotation with batch dimension."""
        x = torch.randn(2, 4, 100, 128)  # [batch, heads, seq, dim]
        
        rotated = rotate_half(x)
        
        # Shape should be preserved
        assert rotated.shape == x.shape
        
        # Check rotation is applied correctly
        assert torch.allclose(rotated[..., :64], -x[..., 64:])
        assert torch.allclose(rotated[..., 64:], x[..., :64])


class TestApplyRotaryPosEmb:
    """Test forward RoPE application."""
    
    def test_identity_at_position_zero(self):
        """Test that position 0 is identity (no rotation)."""
        x = torch.randn(1, 8, 1, 128)  # Single position at 0
        cos, sin = get_rope_frequencies(dim=128, max_seq_len=100)
        
        rotated = apply_rotary_pos_emb(x, cos, sin)
        
        # At position 0, cos=1, sin=0, so x should be unchanged
        torch.testing.assert_close(rotated, x, atol=1e-5, rtol=1e-5)
    
    def test_shape_preservation(self):
        """Test that output shape matches input."""
        x = torch.randn(2, 8, 50, 128)
        cos, sin = get_rope_frequencies(dim=128, max_seq_len=100)
        
        rotated = apply_rotary_pos_emb(x, cos, sin)
        
        assert rotated.shape == x.shape
    
    def test_position_ids(self):
        """Test rotation with specific position IDs."""
        x = torch.randn(1, 8, 5, 128)
        cos, sin = get_rope_frequencies(dim=128, max_seq_len=1000)
        
        # Non-contiguous positions
        position_ids = torch.tensor([0, 10, 20, 30, 40])
        
        rotated = apply_rotary_pos_emb(x, cos, sin, position_ids)
        
        assert rotated.shape == x.shape


class TestInverseRotation:
    """Test that inverse rotation correctly inverts forward rotation."""
    
    def test_inverse_rotation_identity(self):
        """Test that rotate → derotate = identity."""
        x = torch.randn(2, 8, 50, 128)
        cos, sin = get_rope_frequencies(dim=128, max_seq_len=100)
        
        # Apply rotation then inverse
        rotated = apply_rotary_pos_emb(x, cos, sin)
        recovered = apply_inverse_rotary_pos_emb(rotated, cos, sin)
        
        # Should recover original
        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)
    
    def test_inverse_then_forward(self):
        """Test that derotate → rotate = identity."""
        x = torch.randn(2, 8, 50, 128)
        cos, sin = get_rope_frequencies(dim=128, max_seq_len=100)
        
        # Apply inverse then forward
        derotated = apply_inverse_rotary_pos_emb(x, cos, sin)
        recovered = apply_rotary_pos_emb(derotated, cos, sin)
        
        # Should recover original
        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)
    
    def test_multiple_round_trips(self):
        """Test stability across multiple rotate/derotate cycles."""
        x = torch.randn(1, 8, 20, 128)
        cos, sin = get_rope_frequencies(dim=128, max_seq_len=100)
        
        current = x
        for _ in range(5):
            current = apply_rotary_pos_emb(current, cos, sin)
            current = apply_inverse_rotary_pos_emb(current, cos, sin)
        
        # Should still recover original after 5 round trips
        torch.testing.assert_close(current, x, atol=1e-4, rtol=1e-4)


class TestRoPEHandler:
    """Test the RoPEHandler class."""
    
    @pytest.fixture
    def handler(self):
        """Create a default handler."""
        return RoPEHandler(head_dim=128, max_seq_len=1000, base=10000.0)
    
    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler.head_dim == 128
        assert handler.max_seq_len == 1000
        assert handler.base == 10000.0
        assert handler.cos.shape == (1000, 128)
    
    def test_rotate_method(self, handler):
        """Test the rotate convenience method."""
        x = torch.randn(1, 8, 50, 128)
        
        rotated = handler.rotate(x)
        
        assert rotated.shape == x.shape
    
    def test_derotate_method(self, handler):
        """Test the derotate convenience method."""
        x = torch.randn(1, 8, 50, 128)
        
        rotated = handler.rotate(x)
        recovered = handler.derotate(rotated)
        
        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)
    
    def test_device_transfer(self, handler):
        """Test moving handler to different device."""
        if torch.cuda.is_available():
            handler.to(torch.device("cuda:0"))
            
            assert handler.cos.device.type == "cuda"
            assert handler.sin.device.type == "cuda"


class TestRoPECompatibility:
    """Test RoPE compatibility checking."""
    
    def test_matching_bases(self):
        """Test that matching bases are compatible."""
        assert verify_rope_compatibility(10000.0, 10000.0) == True
    
    def test_different_bases(self):
        """Test that different bases are incompatible."""
        assert verify_rope_compatibility(10000.0, 1000000.0) == False
    
    def test_tolerance(self):
        """Test tolerance for near-matching bases."""
        assert verify_rope_compatibility(10000.0, 10000.0 + 1e-8) == True
        assert verify_rope_compatibility(10000.0, 10001.0) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
