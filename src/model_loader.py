"""
Model Loader for He-LMAS

Handles loading Qwen3 models with proper configuration for KV cache
extraction and injection. Supports quantization for memory efficiency.
"""

import torch
from typing import Optional, Tuple, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
import yaml
import warnings


def get_quantization_config(quant_type: Optional[str]) -> Optional[BitsAndBytesConfig]:
    """
    Get BitsAndBytes quantization configuration.
    
    Args:
        quant_type: "int4", "int8", or None for no quantization
        
    Returns:
        BitsAndBytesConfig or None
    """
    if quant_type is None:
        return None
    
    if quant_type == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quant_type == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        raise ValueError(f"Unknown quantization type: {quant_type}")


def load_model(
    model_name: str,
    device: str = "cuda:0",
    quantization: Optional[str] = None,
    trust_remote_code: bool = True
) -> PreTrainedModel:
    """
    Load a HuggingFace model with optional quantization.
    
    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen3-8B")
        device: Target device
        quantization: "int4", "int8", or None
        trust_remote_code: Whether to trust remote code (required for Qwen)
        
    Returns:
        Loaded model
    """
    quant_config = get_quantization_config(quantization)
    
    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.float16,
    }
    
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = device
    else:
        load_kwargs["device_map"] = device
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    return model


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True
) -> PreTrainedTokenizer:
    """
    Load tokenizer for a model.
    
    Args:
        model_name: HuggingFace model ID
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def get_model_config(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Extract relevant configuration from a loaded model.
    
    This is useful for verifying architecture compatibility.
    
    Args:
        model: Loaded HuggingFace model
        
    Returns:
        Dictionary of relevant config values
    """
    config = model.config
    
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
            if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads")
            else None
        ),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }


def verify_compatibility(
    teacher_config: Dict[str, Any],
    student_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify that Teacher and Student models are compatible for He-LMAS.
    
    Args:
        teacher_config: Config dict from get_model_config(teacher)
        student_config: Config dict from get_model_config(student)
        
    Returns:
        Compatibility report
    """
    report = {
        "compatible": True,
        "warnings": [],
        "errors": [],
        "info": {}
    }
    
    # Check KV heads match (critical for current implementation)
    t_kv = teacher_config.get("num_key_value_heads")
    s_kv = student_config.get("num_key_value_heads")
    
    if t_kv != s_kv:
        report["compatible"] = False
        report["errors"].append(
            f"KV head mismatch: Teacher={t_kv}, Student={s_kv}. "
            "Current implementation requires matching KV heads."
        )
    else:
        report["info"]["kv_heads"] = t_kv
    
    # Check head_dim
    t_hd = teacher_config.get("head_dim")
    s_hd = student_config.get("head_dim")
    
    if t_hd != s_hd:
        report["warnings"].append(
            f"Head dimension mismatch: Teacher={t_hd}, Student={s_hd}. "
            "Bridge will project dimensions."
        )
    report["info"]["teacher_head_dim"] = t_hd
    report["info"]["student_head_dim"] = s_hd
    
    # Check RoPE theta
    t_rope = teacher_config.get("rope_theta", 10000.0)
    s_rope = student_config.get("rope_theta", 10000.0)
    
    if t_rope != s_rope:
        report["warnings"].append(
            f"RoPE theta mismatch: Teacher={t_rope}, Student={s_rope}. "
            "Consider using rope_handling='full' in Bridge."
        )
    report["info"]["rope_compatible"] = (t_rope == s_rope)
    
    # Layer info
    t_layers = teacher_config.get("num_hidden_layers")
    s_layers = student_config.get("num_hidden_layers")
    
    report["info"]["teacher_layers"] = t_layers
    report["info"]["student_layers"] = s_layers
    report["info"]["layer_ratio"] = t_layers / s_layers if s_layers else None
    
    # Vocab size (must match for token transfer)
    t_vocab = teacher_config.get("vocab_size")
    s_vocab = student_config.get("vocab_size")
    
    if t_vocab != s_vocab:
        report["warnings"].append(
            f"Vocab size mismatch: Teacher={t_vocab}, Student={s_vocab}. "
            "Models may not share tokenizer."
        )
    
    return report


class HeLMAS_ModelPair:
    """
    Container for Teacher-Student model pair with shared tokenizer.
    
    Usage:
        pair = HeLMAS_ModelPair.from_config("configs/default.yaml")
        teacher_output = pair.teacher_generate(prompt)
        student_output = pair.student_forward(tokens, past_key_values=cache)
    """
    
    def __init__(
        self,
        teacher: PreTrainedModel,
        student: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        teacher_config: Dict[str, Any],
        student_config: Dict[str, Any]
    ):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.teacher_config = teacher_config
        self.student_config = student_config
        
        # Freeze both models (we only train the Bridge)
        self.teacher.eval()
        self.student.eval()
        
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.student.parameters():
            param.requires_grad = False
    
    @classmethod
    def from_config(cls, config_path: str) -> "HeLMAS_ModelPair":
        """
        Load model pair from configuration file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Initialized HeLMAS_ModelPair
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        teacher_cfg = config['models']['teacher']
        student_cfg = config['models']['student']
        
        print(f"Loading Teacher: {teacher_cfg['name']}...")
        teacher = load_model(
            teacher_cfg['name'],
            device=teacher_cfg.get('device', 'cuda:0'),
            quantization=teacher_cfg.get('quantization')
        )
        
        print(f"Loading Student: {student_cfg['name']}...")
        student = load_model(
            student_cfg['name'],
            device=student_cfg.get('device', 'cuda:0'),
            quantization=student_cfg.get('quantization')
        )
        
        # Use shared tokenizer (from Teacher, should be identical)
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(teacher_cfg['name'])
        
        # Get configs
        teacher_config = get_model_config(teacher)
        student_config = get_model_config(student)
        
        # Verify compatibility
        report = verify_compatibility(teacher_config, student_config)
        
        if not report["compatible"]:
            for error in report["errors"]:
                print(f"ERROR: {error}")
            raise ValueError("Models are not compatible for He-LMAS")
        
        for warning in report["warnings"]:
            warnings.warn(warning)
        
        print(f"Compatibility check passed. Layer ratio: {report['info']['layer_ratio']:.2f}")
        
        return cls(
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            teacher_config=teacher_config,
            student_config=student_config
        )
    
    def teacher_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Tuple[torch.Tensor, Any]:
        """
        Generate tokens with Teacher and return KV cache.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated_ids, past_key_values)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.teacher.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.teacher.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                **kwargs
            )
        
        return outputs.sequences, outputs.past_key_values
    
    def student_forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        **kwargs
    ):
        """
        Run Student forward pass with optional injected cache.
        
        Args:
            input_ids: Input token IDs
            past_key_values: KV cache to inject (from Bridge projection)
            
        Returns:
            Model outputs
        """
        input_ids = input_ids.to(self.student.device)
        
        return self.student(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            **kwargs
        )
