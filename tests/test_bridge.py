"""
Tests for the He-LMAS Bridge component.

Run with: pytest tests/test_bridge.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bridge import HeLMAS_Bridge, create_bridge_from_config


class TestHeLMAS_Bridge:
    """Test the HeLMAS_Bridge class."""
    
    @pytest.fixture
    def default_bridge(self):
        """Create a default bridge for Qwen3-8B → Qwen3-1.7B."""
        return HeLMAS_Bridge(
            teacher_layers=36,
            student_layers=28,
            teacher_head_dim=128,
            student_head_dim=128,
            num_kv_heads=8,
            rope_handling="naive",
            per_layer=True,
            init_strategy="identity"
        )
    
    @pytest.fixture
    def sample_teacher_cache(self):
        """Create a sample teacher KV cache."""
        batch = 1
        heads = 8
        seq_len = 100
        head_dim = 128
        layers = 36
        
        cache = []
        for _ in range(layers):
            k = torch.randn(batch, heads, seq_len, head_dim)
            v = torch.randn(batch, heads, seq_len, head_dim)
            cache.append((k, v))
        
        return cache
    
    def test_bridge_initialization(self, default_bridge):
        """Test that bridge initializes correctly."""
        assert default_bridge.teacher_layers == 36
        assert default_bridge.student_layers == 28
        assert len(default_bridge.layer_map) == 28
        
        # Check layer mapping is within bounds
        assert all(0 <= idx < 36 for idx in default_bridge.layer_map)
    
    def test_bridge_output_shape(self, default_bridge, sample_teacher_cache):
        """Test that bridge produces correct output shapes."""
        student_cache = default_bridge(sample_teacher_cache)
        
        # Should produce 28 layers
        assert len(student_cache) == 28
        
        # Each layer should have correct shape
        for k, v in student_cache:
            assert k.shape == (1, 8, 100, 128)  # [batch, heads, seq, head_dim]
            assert v.shape == (1, 8, 100, 128)
    
    def test_layer_mapping_coverage(self, default_bridge):
        """Test that layer mapping covers the teacher layers reasonably."""
        info = default_bridge.get_layer_mapping_info()
        
        # Should use at least some layers from beginning and end
        used = info["used_teacher_layers"]
        assert 0 in used or 1 in used  # Early layers used
        assert 34 in used or 35 in used  # Late layers used
        
        # Ratio should be roughly 1.28
        assert 1.2 < info["ratio"] < 1.4
    
    def test_identity_initialization(self, default_bridge, sample_teacher_cache):
        """Test that identity initialization preserves signal."""
        # For identity init with matching dimensions, output should be close to input
        student_cache = default_bridge(sample_teacher_cache)
        
        # Check first layer (should map to teacher layer 0)
        k_in = sample_teacher_cache[0][0]
        k_out = student_cache[0][0]
        
        # With identity init, these should be identical
        torch.testing.assert_close(k_out, k_in, rtol=1e-5, atol=1e-5)
    
    def test_parameter_count(self, default_bridge):
        """Test parameter counting."""
        num_params = default_bridge.num_parameters()
        
        # For per-layer projectors: 28 layers * 2 (K+V) * (128*128) weights
        expected = 28 * 2 * 128 * 128
        assert num_params == expected
    
    def test_shared_projector_mode(self, sample_teacher_cache):
        """Test bridge with shared projectors."""
        bridge = HeLMAS_Bridge(
            teacher_layers=36,
            student_layers=28,
            per_layer=False  # Shared projector
        )
        
        student_cache = bridge(sample_teacher_cache)
        
        # Should still produce correct output
        assert len(student_cache) == 28
        
        # Parameter count should be much smaller
        num_params = bridge.num_parameters()
        expected = 2 * 128 * 128  # Just 2 projectors (K and V)
        assert num_params == expected
    
    def test_dimension_projection(self, sample_teacher_cache):
        """Test bridge with different head dimensions."""
        bridge = HeLMAS_Bridge(
            teacher_layers=36,
            student_layers=28,
            teacher_head_dim=128,
            student_head_dim=96,  # Different dimension
            init_strategy="random"
        )
        
        student_cache = bridge(sample_teacher_cache)
        
        for k, v in student_cache:
            assert k.shape[-1] == 96  # Projected dimension
            assert v.shape[-1] == 96


class TestLayerMapping:
    """Test layer mapping strategies."""
    
    def test_uniform_stride(self):
        """Test that uniform stride mapping is correct."""
        bridge = HeLMAS_Bridge(teacher_layers=36, student_layers=28)
        
        # First and last should be mapped
        mapping = bridge.layer_map.tolist()
        assert mapping[0] == 0  # First student layer → first teacher layer
        assert mapping[-1] == 35  # Last student layer → last teacher layer
    
    def test_extreme_compression(self):
        """Test with extreme layer ratio (like Llama 70B → 8B)."""
        bridge = HeLMAS_Bridge(
            teacher_layers=80,
            student_layers=32,
            teacher_head_dim=128,
            student_head_dim=128
        )
        
        info = bridge.get_layer_mapping_info()
        
        # Ratio should be 2.5
        assert abs(info["ratio"] - 2.5) < 0.01
        
        # Should skip about 48 layers
        assert info["num_skipped"] == 48


class TestConfigFactory:
    """Test bridge creation from config."""
    
    def test_create_from_config(self, tmp_path):
        """Test creating bridge from config dictionary."""
        config = {
            'models': {
                'teacher': {
                    'layers': 36,
                    'head_dim': 128,
                    'num_kv_heads': 8
                },
                'student': {
                    'layers': 28,
                    'head_dim': 128,
                    'num_kv_heads': 8
                }
            },
            'bridge': {
                'rope_handling': 'naive',
                'per_layer_projectors': True,
                'init_strategy': 'identity'
            }
        }
        
        bridge = create_bridge_from_config(config)
        
        assert bridge.teacher_layers == 36
        assert bridge.student_layers == 28
        assert bridge.rope_handling == "naive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
