"""
Tests for the He-LMAS Bridge component.

Updated for V2 architecture with:
- Layer blending modes (skip, blend, attention)
- Deep projectors with configurable depth
- Full RoPE de-rotation/re-rotation

Run with: pytest tests/test_bridge.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bridge import (
    HeLMAS_Bridge, 
    create_bridge_from_config,
    ShallowProjector,
    DeepProjector,
    LayerBlender,
    AttentionLayerPooler
)


class TestProjectors:
    """Test the projector classes."""
    
    def test_shallow_projector_identity(self):
        """Test ShallowProjector with identity initialization."""
        proj = ShallowProjector(128, 128, init_strategy="identity")
        x = torch.randn(1, 8, 100, 128)
        y = proj(x)
        
        assert y.shape == x.shape
        torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-5)
    
    def test_shallow_projector_dimension_change(self):
        """Test ShallowProjector with dimension projection."""
        proj = ShallowProjector(128, 96, init_strategy="random")
        x = torch.randn(1, 8, 100, 128)
        y = proj(x)
        
        assert y.shape == (1, 8, 100, 96)
    
    def test_deep_projector_shape(self):
        """Test DeepProjector output shape."""
        proj = DeepProjector(128, 128, hidden_dim=256, depth=2)
        x = torch.randn(1, 8, 100, 128)
        y = proj(x)
        
        assert y.shape == x.shape
    
    def test_deep_projector_params(self):
        """Test DeepProjector has more parameters than shallow."""
        shallow = ShallowProjector(128, 128)
        deep = DeepProjector(128, 128, hidden_dim=256, depth=2)
        
        shallow_params = sum(p.numel() for p in shallow.parameters())
        deep_params = sum(p.numel() for p in deep.parameters())
        
        # Deep should have more params
        assert deep_params > shallow_params


class TestLayerBlending:
    """Test the layer blending classes."""
    
    @pytest.fixture
    def sample_teacher_cache(self):
        """Create a sample teacher KV cache (36 layers)."""
        batch, heads, seq_len, head_dim = 1, 8, 100, 128
        cache = []
        for _ in range(36):
            k = torch.randn(batch, heads, seq_len, head_dim)
            v = torch.randn(batch, heads, seq_len, head_dim)
            cache.append((k, v))
        return cache
    
    def test_layer_blender_output_shape(self, sample_teacher_cache):
        """Test LayerBlender (Conv1d) produces correct output."""
        blender = LayerBlender(teacher_layers=36, student_layers=28)
        result = blender(sample_teacher_cache)
        
        assert len(result) == 28
        for k, v in result:
            assert k.shape == (1, 8, 100, 128)
            assert v.shape == (1, 8, 100, 128)
    
    def test_attention_pooler_output_shape(self, sample_teacher_cache):
        """Test AttentionLayerPooler produces correct output."""
        pooler = AttentionLayerPooler(teacher_layers=36, student_layers=28)
        result = pooler(sample_teacher_cache)
        
        assert len(result) == 28
        for k, v in result:
            assert k.shape == (1, 8, 100, 128)
            assert v.shape == (1, 8, 100, 128)
    
    def test_attention_weights_sum_to_one(self):
        """Test attention weights are properly normalized."""
        pooler = AttentionLayerPooler(teacher_layers=36, student_layers=28)
        attn_k, attn_v = pooler.get_attention_weights()
        
        # Each row should sum to 1 (softmax)
        torch.testing.assert_close(
            attn_k.sum(dim=-1), 
            torch.ones(28), 
            rtol=1e-5, atol=1e-5
        )


class TestHeLMAS_Bridge:
    """Test the main HeLMAS_Bridge class."""
    
    @pytest.fixture
    def sample_teacher_cache(self):
        """Create a sample teacher KV cache."""
        batch, heads, seq_len, head_dim = 1, 8, 100, 128
        cache = []
        for _ in range(36):
            k = torch.randn(batch, heads, seq_len, head_dim)
            v = torch.randn(batch, heads, seq_len, head_dim)
            cache.append((k, v))
        return cache
    
    def test_bridge_skip_mode(self, sample_teacher_cache):
        """Test bridge with skip (uniform stride) layer mapping."""
        bridge = HeLMAS_Bridge(
            teacher_layers=36,
            student_layers=28,
            layer_blending="skip",
            projector_depth=1
        )
        
        result = bridge(sample_teacher_cache)
        
        assert len(result) == 28
        for k, v in result:
            assert k.shape == (1, 8, 100, 128)
            assert v.shape == (1, 8, 100, 128)
    
    def test_bridge_blend_mode(self, sample_teacher_cache):
        """Test bridge with Conv1d blending."""
        bridge = HeLMAS_Bridge(
            teacher_layers=36,
            student_layers=28,
            layer_blending="blend",
            projector_depth=2
        )
        
        result = bridge(sample_teacher_cache)
        
        assert len(result) == 28
        for k, v in result:
            assert k.shape == (1, 8, 100, 128)
    
    def test_bridge_attention_mode(self, sample_teacher_cache):
        """Test bridge with attention pooling (default)."""
        bridge = HeLMAS_Bridge(
            teacher_layers=36,
            student_layers=28,
            layer_blending="attention",
            projector_depth=2
        )
        
        result = bridge(sample_teacher_cache)
        
        assert len(result) == 28
        for k, v in result:
            assert k.shape == (1, 8, 100, 128)
    
    def test_bridge_full_rope(self, sample_teacher_cache):
        """Test bridge with full RoPE handling."""
        bridge = HeLMAS_Bridge(
            teacher_layers=36,
            student_layers=28,
            rope_handling="full",
            teacher_rope_base=1000000.0,
            student_rope_base=1000000.0
        )
        
        result = bridge(sample_teacher_cache)
        
        assert len(result) == 28
    
    def test_parameter_count_shallow(self):
        """Test parameter count for shallow projectors."""
        bridge = HeLMAS_Bridge(
            teacher_layers=36,
            student_layers=28,
            layer_blending="skip",
            projector_depth=1,
            per_layer=True
        )
        
        num_params = bridge.num_parameters()
        
        # 28 layers * 2 (K+V) * (128*128) = 917504
        expected_projector_params = 28 * 2 * 128 * 128
        assert num_params >= expected_projector_params
    
    def test_parameter_count_deep(self):
        """Test deep projectors have more params than shallow."""
        shallow = HeLMAS_Bridge(
            layer_blending="skip",
            projector_depth=1
        )
        deep = HeLMAS_Bridge(
            layer_blending="skip",
            projector_depth=2
        )
        
        assert deep.num_parameters() > shallow.num_parameters()


class TestConfigFactory:
    """Test bridge creation from config."""
    
    def test_create_from_config_defaults(self):
        """Test creating bridge with default values."""
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
            'bridge': {}
        }
        
        bridge = create_bridge_from_config(config)
        
        assert bridge.teacher_layers == 36
        assert bridge.student_layers == 28
        # Default is now 'attention' and 'full' rope
        assert bridge.layer_blending == "attention"
        assert bridge.rope_handling == "full"
    
    def test_create_from_config_custom(self):
        """Test creating bridge with custom values."""
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
                'layer_blending': 'blend',
                'projector_depth': 3,
                'rope_handling': 'naive'
            }
        }
        
        bridge = create_bridge_from_config(config)
        
        assert bridge.layer_blending == "blend"
        assert bridge.projector_depth == 3
        assert bridge.rope_handling == "naive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
