"""
Training Infrastructure for He-LMAS

Implements the "Attention Consistency Loss" from RFC Section 3.2:
  L = ||Attn_Student(Q, Φ(KV_Teacher)) - Attn_Student(Q, KV_Student^TeacherForced)||²

Key insight: Instead of forcing caches to match (geometrically impossible),
we force the Attention OUTPUT to match. The Student should see the same
"Context Vector" when looking at the Projected Teacher memory as it would
if it had correctly computed the memory itself.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any, Iterator, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import time
import yaml

from .bridge import HeLMAS_Bridge, create_bridge_from_config
from .model_loader import HeLMAS_ModelPair
from .kv_cache_utils import extract_kv_cache, slice_cache_at_token, kv_cache_info


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 1
    max_steps: int = 10000
    checkpoint_every: int = 500
    log_every: int = 10
    max_thinking_tokens: int = 100
    gradient_accumulation: int = 4
    warmup_steps: int = 100
    
    # Loss weights (RFC: Attention Consistency + optional auxiliary CE)
    attention_consistency_weight: float = 1.0
    auxiliary_ce_weight: float = 0.1  # Small weight for next-token prediction
    
    # Which layers to compute attention consistency on
    consistency_layers: str = "all"  # "all", "last", or "sample"
    
    # Handoff mode: what to feed Student after injecting projected cache
    #   - "none": Empty input (relies purely on cache)
    #   - "prompt": Short handoff prompt (default: "\nAnswer:")
    #   - "full_context": Re-feed original question
    handoff_mode: str = "prompt"
    handoff_prompt: str = "\nAnswer:"
    
    # Validation: run evaluation on held-out set
    eval_every: int = 1000  # Run validation every N steps (0 to disable)
    eval_samples: int = 50  # Number of samples for validation
    
    # H200 Optimization: Automatic Mixed Precision
    use_amp: bool = False
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16" (H200 prefers BF16)
    
    # H200 Optimization: torch.compile()
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    # Gradient checkpointing (save VRAM at compute cost)
    gradient_checkpointing: bool = False
    
    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass 
class TrainingState:
    """Current training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    best_loss: float = float('inf')
    samples_seen: int = 0


class AttentionConsistencyLoss(nn.Module):
    """
    RFC Section 3.2: Attention Consistency Loss
    
    L = ||Attn(Q, Φ(KV_Big)) - Attn(Q, KV_Small^TeacherForced)||²
    
    Instead of forcing the caches to match (which is geometrically impossible),
    we force the Attention Output to match.
    
    Intuition: "The Student should see the same 'Context Vector' when looking
    at the Projected memory as it would if it had correctly computed the memory itself."
    
    IMPORTANT: With batch_size > 1, we must mask out padding tokens to avoid
    the model learning to match zeros on padding positions.
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        attn_output_projected: torch.Tensor,
        attn_output_native: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention consistency loss with optional masking.
        
        Args:
            attn_output_projected: Attention output using projected KV cache
                                   Shape: [batch, seq, hidden] or [batch, heads, seq, head_dim]
            attn_output_native: Attention output using Student's native KV cache
                                Shape: same as above
            attention_mask: Optional mask [batch, seq] where 1=real token, 0=padding
                           CRITICAL for batch_size > 1 to avoid learning on padding
        
        Returns:
            L2 loss between the attention outputs (masked if attention_mask provided)
        """
        if self.normalize:
            # Normalize to unit vectors for stable gradients
            attn_output_projected = nn.functional.normalize(attn_output_projected, dim=-1)
            attn_output_native = nn.functional.normalize(attn_output_native, dim=-1)
        
        # Compute per-element squared error
        squared_error = (attn_output_projected - attn_output_native) ** 2
        
        if attention_mask is not None:
            # Expand mask to match tensor dimensions
            # Input mask: [batch, seq] → expand to [batch, seq, 1, 1] or similar
            while attention_mask.dim() < squared_error.dim():
                attention_mask = attention_mask.unsqueeze(-1)
            
            # Zero out loss on padding positions
            squared_error = squared_error * attention_mask.float()
            
            # Mean over non-padding tokens only
            num_real_elements = attention_mask.sum() * squared_error.shape[-1]
            if num_real_elements > 0:
                loss = squared_error.sum() / num_real_elements
            else:
                loss = squared_error.sum()  # Fallback (shouldn't happen)
        else:
            # No mask: simple mean (batch_size=1 case)
            loss = squared_error.mean()
        
        return loss


class AttentionHook:
    """
    Hook to capture attention outputs from transformer layers.
    
    We attach this to the Student model to extract the attention output
    (the "context vector") for computing the consistency loss.
    """
    
    def __init__(self):
        self.outputs = []
        self.handles = []
    
    def hook_fn(self, module, input, output):
        """Capture the output of an attention layer."""
        # Handle different output formats
        if isinstance(output, tuple):
            # Usually (attn_output, attn_weights, ...)
            attn_output = output[0]
        else:
            attn_output = output
        
        self.outputs.append(attn_output.detach().clone())
    
    def attach(self, model, layer_indices: Optional[List[int]] = None):
        """
        Attach hooks to attention layers.
        
        Args:
            model: The transformer model
            layer_indices: Which layers to hook (None = all)
        """
        self.clear()
        
        # Find attention layers (Qwen uses 'self_attn')
        for i, layer in enumerate(model.model.layers):
            if layer_indices is not None and i not in layer_indices:
                continue
            
            if hasattr(layer, 'self_attn'):
                handle = layer.self_attn.register_forward_hook(self.hook_fn)
                self.handles.append(handle)
    
    def clear(self):
        """Clear captured outputs and remove hooks."""
        self.outputs = []
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def get_outputs(self) -> List[torch.Tensor]:
        """Get captured attention outputs."""
        return self.outputs


class HeLMAS_Trainer:
    """
    Trainer for the He-LMAS Bridge using Attention Consistency Loss.
    
    RFC Section 3.2 Training Strategy:
    1. Teacher generates reasoning trace → extract KV cache
    2. Bridge projects Teacher's KV cache to Student geometry
    3. Run Student with PROJECTED cache → capture attention outputs
    4. Run Student with NATIVE cache (teacher-forced) → capture attention outputs
    5. Loss = ||Attn_projected - Attn_native||²
    6. Backpropagate through Bridge only
    
    Usage:
        trainer = HeLMAS_Trainer.from_config("configs/default.yaml")
        trainer.train(data_iterator)
    """
    
    def __init__(
        self,
        model_pair: HeLMAS_ModelPair,
        bridge: HeLMAS_Bridge,
        config: TrainingConfig,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        eval_data: Optional[List[Tuple[str, str]]] = None
    ):
        self.model_pair = model_pair
        self.bridge = bridge
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.eval_data = eval_data  # List of (question, answer) for validation
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Move bridge to same device as student
        self.device = next(self.model_pair.student.parameters()).device
        self.bridge = self.bridge.to(self.device)
        
        # H200 Optimization: torch.compile() on Bridge
        if config.compile_mode and config.compile_mode != "default":
            print(f"🔥 Compiling Bridge with mode='{config.compile_mode}'")
            self.bridge = torch.compile(self.bridge, mode=config.compile_mode)
        
        # Optimizer (only for bridge parameters)
        self.optimizer = AdamW(
            self.bridge.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # H200 Optimization: Automatic Mixed Precision
        self.use_amp = config.use_amp
        self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        self.grad_scaler = torch.amp.GradScaler(enabled=self.use_amp and config.amp_dtype == "float16")
        
        if self.use_amp:
            print(f"⚡ AMP enabled with dtype={config.amp_dtype}")
        
        # Loss functions (RFC: Attention Consistency is primary)
        self.attn_consistency_loss = AttentionConsistencyLoss(normalize=True)
        self.ce_loss = nn.CrossEntropyLoss()  # Auxiliary
        
        # Attention hooks for capturing outputs
        self.attn_hook = AttentionHook()
        
        # Training state
        self.state = TrainingState()
        
        # Logging
        self.writer = None
    
    @classmethod
    def from_config(
        cls, 
        config_path: str,
        eval_data: Optional[List[Tuple[str, str]]] = None
    ) -> "HeLMAS_Trainer":
        """
        Create trainer from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            eval_data: Optional list of (question, answer) tuples for validation
                       If provided, validation runs every eval_every steps
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_pair = HeLMAS_ModelPair.from_config(config_path)
        bridge = create_bridge_from_config(config)
        train_config = TrainingConfig.from_dict(config.get('training', {}))
        
        return cls(
            model_pair=model_pair,
            bridge=bridge,
            config=train_config,
            checkpoint_dir=config.get('paths', {}).get('checkpoints', 'checkpoints'),
            log_dir=config.get('paths', {}).get('logs', 'logs'),
            eval_data=eval_data
        )
    
    def _get_layer_indices(self) -> Optional[List[int]]:
        """Determine which layers to compute consistency on."""
        n_layers = self.model_pair.student_config.get('num_hidden_layers', 28)
        
        if self.config.consistency_layers == "all":
            return None  # Hook all layers
        elif self.config.consistency_layers == "last":
            return [n_layers - 1]  # Only last layer
        elif self.config.consistency_layers == "sample":
            # Sample every 4th layer
            return list(range(0, n_layers, 4))
        else:
            return None
    
    def train_step(self, prompt: str) -> Dict[str, float]:
        """
        Single training step implementing RFC Section 3.2.
        
        The Attention Consistency approach:
        1. Get Teacher's KV cache for the prompt
        2. Project it through Bridge → projected_cache
        3. Run Student with projected_cache → capture attn outputs
        4. Run Student with its own cache (native) → capture attn outputs
        5. L = ||attn_projected - attn_native||²
        
        H200 Optimization: Uses AMP autocast for mixed precision training.
        """
        metrics = {}
        
        # AMP autocast context for H200 optimization
        amp_context = torch.amp.autocast(
            device_type='cuda',
            dtype=self.amp_dtype,
            enabled=self.use_amp
        )
        
        # === STEP 1: Teacher generates reasoning and we get KV cache ===
        with torch.no_grad():
            teacher_ids, teacher_cache = self.model_pair.teacher_generate(
                prompt,
                max_new_tokens=self.config.max_thinking_tokens
            )
        
        # Extract and slice cache at </think>
        teacher_cache = extract_kv_cache(
            type('Output', (), {'past_key_values': teacher_cache})()
        )
        sliced_cache, cut_pos = slice_cache_at_token(
            teacher_cache,
            teacher_ids,
            self.model_pair.tokenizer,
            end_token="</think>",
            include_end_token=True
        )
        
        metrics['teacher_thinking_tokens'] = cut_pos
        
        # === STEP 2: Bridge projects Teacher's cache (AMP enabled) ===
        with amp_context:
            projected_cache = self.bridge(sliced_cache)
        projected_cache_tuple = tuple(projected_cache)
        
        # === STEP 3: Prepare query tokens based on handoff mode ===
        # Handoff modes:
        #   - "none": Empty input, rely purely on the injected cache
        #   - "prompt": Short handoff prompt (e.g., "\nAnswer:")
        #   - "full_context": Re-feed the original question
        
        if self.config.handoff_mode == "none":
            # Empty token - just a single padding token to trigger generation
            query_ids = torch.tensor(
                [[self.model_pair.tokenizer.eos_token_id]], 
                dtype=torch.long
            ).to(self.device)
        elif self.config.handoff_mode == "full_context":
            # Re-feed the original question (extract from prompt)
            # Note: prompt already has thinking wrapper, we want just the question
            query_ids = self.model_pair.tokenizer.encode(
                prompt,  # Full original prompt
                return_tensors="pt"
            ).to(self.device)
        else:  # "prompt" mode (default)
            query_ids = self.model_pair.tokenizer.encode(
                self.config.handoff_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.device)
        
        metrics['handoff_mode'] = self.config.handoff_mode
        
        # === STEP 4: Forward pass with PROJECTED cache (AMP enabled) ===
        layer_indices = self._get_layer_indices()
        self.attn_hook.attach(self.model_pair.student, layer_indices)
        
        with amp_context:
            outputs_projected = self.model_pair.student(
                input_ids=query_ids,
                past_key_values=projected_cache_tuple,
                use_cache=True,
                output_attentions=False  # We use hooks instead
            )
        
        attn_outputs_projected = self.attn_hook.get_outputs()
        self.attn_hook.clear()
        
        # === STEP 5: Forward pass with NATIVE Student cache (teacher-forced) ===
        # First, compute Student's native cache for the same context
        # We feed the original prompt tokens to Student
        prompt_ids = self.model_pair.tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Get Student's own understanding of the prompt
            student_prefill = self.model_pair.student(
                input_ids=prompt_ids,
                use_cache=True
            )
            native_cache = student_prefill.past_key_values
        
        # Now run with native cache and capture attention
        self.attn_hook.attach(self.model_pair.student, layer_indices)
        
        with torch.no_grad():
            outputs_native = self.model_pair.student(
                input_ids=query_ids,
                past_key_values=native_cache,
                use_cache=True
            )
        
        attn_outputs_native = self.attn_hook.get_outputs()
        self.attn_hook.clear()
        
        # === STEP 6: Compute Attention Consistency Loss ===
        total_attn_loss = 0.0
        num_layers = len(attn_outputs_projected)
        
        for proj_attn, native_attn in zip(attn_outputs_projected, attn_outputs_native):
            # Ensure same shape (truncate to min length if needed)
            min_len = min(proj_attn.shape[-2], native_attn.shape[-2])
            proj_attn = proj_attn[..., :min_len, :]
            native_attn = native_attn[..., :min_len, :]
            
            layer_loss = self.attn_consistency_loss(proj_attn, native_attn)
            total_attn_loss += layer_loss
        
        attn_loss = total_attn_loss / max(num_layers, 1)
        metrics['attn_consistency_loss'] = attn_loss.item()
        
        # === STEP 7: Optional auxiliary CE loss ===
        if self.config.auxiliary_ce_weight > 0:
            # Predict next token after Teacher's thinking
            target_ids = teacher_ids[:, cut_pos:cut_pos + 1].to(self.device)
            
            if target_ids.shape[1] > 0:
                logits = outputs_projected.logits[:, -1, :]
                ce_loss = self.ce_loss(logits, target_ids[:, 0])
                metrics['ce_loss'] = ce_loss.item()
            else:
                ce_loss = torch.tensor(0.0, device=self.device)
                metrics['ce_loss'] = 0.0
        else:
            ce_loss = torch.tensor(0.0, device=self.device)
        
        # === STEP 8: Combined loss ===
        total_loss = (
            self.config.attention_consistency_weight * attn_loss +
            self.config.auxiliary_ce_weight * ce_loss
        )
        metrics['total_loss'] = total_loss.item()
        
        # === STEP 9: Backward and optimize (with GradScaler for AMP) ===
        total_loss = total_loss / self.config.gradient_accumulation
        
        # Use GradScaler for FP16 AMP, direct backward for BF16
        self.grad_scaler.scale(total_loss).backward()
        
        if (self.state.step + 1) % self.config.gradient_accumulation == 0:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), max_norm=1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            metrics['lr'] = self.scheduler.get_last_lr()[0]
        
        # === Cleanup VRAM ===
        del teacher_cache, sliced_cache, projected_cache
        del attn_outputs_projected, attn_outputs_native
        del native_cache, outputs_projected, outputs_native
        torch.cuda.empty_cache()
        
        return metrics
    
    def train_step_batched(self, prompts: List[str]) -> Dict[str, float]:
        """
        Batched training step for true parallel processing.
        
        Processes multiple prompts in parallel for better H200 utilization.
        
        Key differences from train_step:
        - Teacher generates for each prompt sequentially (can't batch generation)
        - KV caches are stacked and padded for batch projection
        - Student forward pass is fully batched
        - Attention mask is passed to loss to ignore padding
        
        Args:
            prompts: List of formatted prompts (batch_size items)
            
        Returns:
            Dict of training metrics (averaged across batch)
        """
        metrics = {}
        batch_size = len(prompts)
        
        amp_context = torch.amp.autocast(
            device_type='cuda',
            dtype=self.amp_dtype,
            enabled=self.use_amp
        )
        
        # === STEP 1: Teacher generates for each prompt (sequential) ===
        # Note: Teacher generation can't be easily batched due to varying output lengths
        teacher_caches = []
        max_seq_len = 0
        
        with torch.no_grad():
            for prompt in prompts:
                teacher_ids, teacher_cache = self.model_pair.teacher_generate(
                    prompt,
                    max_new_tokens=self.config.max_thinking_tokens
                )
                
                # Extract and slice at </think>
                cache = extract_kv_cache(
                    type('Output', (), {'past_key_values': teacher_cache})()
                )
                sliced_cache, cut_pos = slice_cache_at_token(
                    cache,
                    teacher_ids,
                    self.model_pair.tokenizer,
                    end_token="</think>",
                    include_end_token=True
                )
                
                teacher_caches.append(sliced_cache)
                seq_len = sliced_cache[0][0].shape[2]  # [batch, heads, seq, dim]
                max_seq_len = max(max_seq_len, seq_len)
        
        metrics['teacher_thinking_tokens'] = max_seq_len
        
        # === STEP 2: Pad and stack KV caches ===
        # Each cache: list of (k, v) tuples per layer
        # k/v shape: [1, heads, seq, dim] - need to pad seq dim and stack batch
        
        n_layers = len(teacher_caches[0])
        padded_cache = []
        attention_masks = []
        
        for layer_idx in range(n_layers):
            k_list = []
            v_list = []
            
            for cache in teacher_caches:
                k, v = cache[layer_idx]
                seq_len = k.shape[2]
                
                # Pad to max_seq_len
                if seq_len < max_seq_len:
                    pad_len = max_seq_len - seq_len
                    k = torch.nn.functional.pad(k, (0, 0, 0, pad_len))  # Pad seq dim
                    v = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
                
                k_list.append(k)
                v_list.append(v)
            
            # Stack: [batch, heads, seq, dim]
            k_stacked = torch.cat(k_list, dim=0)
            v_stacked = torch.cat(v_list, dim=0)
            padded_cache.append((k_stacked, v_stacked))
        
        # Build attention mask: [batch, max_seq_len]
        for cache in teacher_caches:
            seq_len = cache[0][0].shape[2]
            mask = torch.zeros(max_seq_len, device=self.device)
            mask[:seq_len] = 1.0
            attention_masks.append(mask)
        
        attention_mask = torch.stack(attention_masks, dim=0)  # [batch, seq]
        
        # === STEP 3: Bridge projection (batched) ===
        with amp_context:
            projected_cache = self.bridge(padded_cache)
        projected_cache_tuple = tuple(projected_cache)
        
        # === STEP 4: Prepare query tokens (batched) ===
        if self.config.handoff_mode == "prompt":
            query_text = self.config.handoff_prompt
            # Tokenize batch with padding
            query_inputs = self.model_pair.tokenizer(
                [query_text] * batch_size,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            query_ids = query_inputs.input_ids
            query_mask = query_inputs.attention_mask
        else:
            query_ids = torch.tensor(
                [[self.model_pair.tokenizer.eos_token_id]] * batch_size,
                dtype=torch.long
            ).to(self.device)
            query_mask = torch.ones_like(query_ids)
        
        # === STEP 5: Student forward with projected cache (batched) ===
        self.attn_hook.attach(self.model_pair.student, self._get_layer_indices())
        
        with amp_context:
            outputs_projected = self.model_pair.student_forward(
                query_ids,
                past_key_values=projected_cache_tuple,
                use_cache=True
            )
        
        attn_outputs_projected = self.attn_hook.get_outputs()
        self.attn_hook.clear()
        
        # === STEP 6: Teacher-forced native Student forward (batched) ===
        # For simplicity, we'll use the same query for native forward
        # and let the Student build its own cache
        
        with torch.no_grad():
            # Tokenize all prompts together
            native_inputs = self.model_pair.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            outputs_native_temp = self.model_pair.student_forward(
                native_inputs.input_ids,
                attention_mask=native_inputs.attention_mask,
                use_cache=True
            )
            native_cache = outputs_native_temp.past_key_values
        
        self.attn_hook.attach(self.model_pair.student, self._get_layer_indices())
        
        with amp_context:
            outputs_native = self.model_pair.student_forward(
                query_ids,
                past_key_values=native_cache,
                use_cache=True
            )
        
        attn_outputs_native = self.attn_hook.get_outputs()
        self.attn_hook.clear()
        
        # === STEP 7: Compute Attention Consistency Loss (with masking) ===
        total_attn_loss = 0.0
        num_layers = len(attn_outputs_projected)
        
        for proj_attn, native_attn in zip(attn_outputs_projected, attn_outputs_native):
            min_len = min(proj_attn.shape[-2], native_attn.shape[-2])
            proj_attn = proj_attn[..., :min_len, :]
            native_attn = native_attn[..., :min_len, :]
            
            # Use query_mask for the attention outputs
            layer_loss = self.attn_consistency_loss(
                proj_attn, 
                native_attn, 
                attention_mask=query_mask[..., :min_len]
            )
            total_attn_loss += layer_loss
        
        attn_loss = total_attn_loss / max(num_layers, 1)
        metrics['attn_consistency_loss'] = attn_loss.item()
        
        # === STEP 8: Combined loss ===
        total_loss = self.config.attention_consistency_weight * attn_loss
        metrics['total_loss'] = total_loss.item()
        metrics['batch_size'] = batch_size
        
        # === STEP 9: Backward and optimize ===
        total_loss = total_loss / self.config.gradient_accumulation
        self.grad_scaler.scale(total_loss).backward()
        
        if (self.state.step + 1) % self.config.gradient_accumulation == 0:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), max_norm=1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            metrics['lr'] = self.scheduler.get_last_lr()[0]
        
        # Cleanup
        del teacher_caches, padded_cache, projected_cache
        del attn_outputs_projected, attn_outputs_native
        torch.cuda.empty_cache()
        
        return metrics
    
    def train(
        self,
        data_iterator: Iterator[str],
        max_steps: Optional[int] = None
    ):
        """Main training loop."""
        max_steps = max_steps or self.config.max_steps
        
        self.bridge.train()
        self.optimizer.zero_grad()
        
        running_attn_loss = 0.0
        running_ce_loss = 0.0
        step_times = []
        
        print(f"Starting training for {max_steps} steps...")
        print(f"Bridge parameters: {self.bridge.num_parameters():,}")
        print(f"Loss: Attention Consistency (w={self.config.attention_consistency_weight}) + "
              f"CE (w={self.config.auxiliary_ce_weight})")
        
        for prompt in data_iterator:
            if self.state.step >= max_steps:
                break
            
            step_start = time.time()
            
            try:
                metrics = self.train_step(prompt)
                
                running_attn_loss += metrics.get('attn_consistency_loss', 0)
                running_ce_loss += metrics.get('ce_loss', 0)
                self.state.step += 1
                self.state.samples_seen += 1
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Logging
                if self.state.step % self.config.log_every == 0:
                    avg_attn = running_attn_loss / self.config.log_every
                    avg_ce = running_ce_loss / self.config.log_every
                    avg_time = sum(step_times[-self.config.log_every:]) / len(step_times[-self.config.log_every:])
                    
                    print(
                        f"Step {self.state.step}/{max_steps} | "
                        f"AttnLoss: {avg_attn:.4f} | "
                        f"CELoss: {avg_ce:.4f} | "
                        f"Time: {avg_time:.2f}s | "
                        f"LR: {metrics.get('lr', self.config.learning_rate):.2e}"
                    )
                    
                    running_attn_loss = 0.0
                    running_ce_loss = 0.0
                
                # Checkpointing
                if self.state.step % self.config.checkpoint_every == 0:
                    self.save_checkpoint()
                
                # Validation evaluation
                if (self.config.eval_every > 0 and 
                    self.state.step % self.config.eval_every == 0 and
                    self.eval_data is not None):
                    
                    print(f"\n📊 Running validation on {min(len(self.eval_data), self.config.eval_samples)} samples...")
                    eval_subset = self.eval_data[:self.config.eval_samples]
                    eval_results = self.evaluate(eval_subset, verbose=False)
                    
                    print(
                        f"  With Bridge: {eval_results['with_bridge_accuracy']:.1%} "
                        f"({eval_results['with_bridge_correct']}/{eval_results['total']})\n"
                        f"  Without Bridge: {eval_results['without_bridge_accuracy']:.1%} "
                        f"({eval_results['without_bridge_correct']}/{eval_results['total']})\n"
                        f"  Improvement: {eval_results['improvement']:+.1%}\n"
                    )
                    
            except Exception as e:
                print(f"Error at step {self.state.step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.save_checkpoint(final=True)
        print(f"Training complete. Final step: {self.state.step}")
    
    def train_batched(
        self,
        batch_iterator: Iterator[List[str]],
        max_steps: Optional[int] = None
    ):
        """
        Main training loop with true batching for H200 speedup.
        
        Args:
            batch_iterator: Iterator yielding lists of prompts (from get_batched_training_data)
            max_steps: Maximum training steps
        """
        max_steps = max_steps or self.config.max_steps
        
        self.bridge.train()
        self.optimizer.zero_grad()
        
        running_attn_loss = 0.0
        step_times = []
        
        print(f"Starting BATCHED training for {max_steps} steps...")
        print(f"Bridge parameters: {self.bridge.num_parameters():,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Loss: Attention Consistency (w={self.config.attention_consistency_weight})")
        
        for batch in batch_iterator:
            if self.state.step >= max_steps:
                break
            
            step_start = time.time()
            
            try:
                metrics = self.train_step_batched(batch)
                
                running_attn_loss += metrics.get('attn_consistency_loss', 0)
                self.state.step += 1
                self.state.samples_seen += len(batch)
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Logging
                if self.state.step % self.config.log_every == 0:
                    avg_attn = running_attn_loss / self.config.log_every
                    avg_time = sum(step_times[-self.config.log_every:]) / len(step_times[-self.config.log_every:])
                    samples_per_sec = len(batch) / avg_time
                    
                    print(
                        f"Step {self.state.step}/{max_steps} | "
                        f"AttnLoss: {avg_attn:.4f} | "
                        f"Time: {avg_time:.2f}s | "
                        f"Samples/s: {samples_per_sec:.1f} | "
                        f"LR: {metrics.get('lr', self.config.learning_rate):.2e}"
                    )
                    
                    running_attn_loss = 0.0
                
                # Checkpointing
                if self.state.step % self.config.checkpoint_every == 0:
                    self.save_checkpoint()
                
                # Validation
                if (self.config.eval_every > 0 and 
                    self.state.step % self.config.eval_every == 0 and
                    self.eval_data is not None):
                    
                    print(f"\n📊 Running validation on {min(len(self.eval_data), self.config.eval_samples)} samples...")
                    eval_subset = self.eval_data[:self.config.eval_samples]
                    eval_results = self.evaluate(eval_subset, verbose=False)
                    
                    print(
                        f"  With Bridge: {eval_results['with_bridge_accuracy']:.1%} "
                        f"({eval_results['with_bridge_correct']}/{eval_results['total']})\n"
                        f"  Without Bridge: {eval_results['without_bridge_accuracy']:.1%} "
                        f"({eval_results['without_bridge_correct']}/{eval_results['total']})\n"
                        f"  Improvement: {eval_results['improvement']:+.1%}\n"
                    )
                    
            except Exception as e:
                print(f"Error at step {self.state.step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.save_checkpoint(final=True)
        print(f"Training complete. Final step: {self.state.step}")
        print(f"Total samples processed: {self.state.samples_seen:,}")
    
    def save_checkpoint(self, final: bool = False):
        """Save checkpoint."""
        name = "bridge_final.pt" if final else f"bridge_step_{self.state.step}.pt"
        path = self.checkpoint_dir / name
        
        torch.save({
            'bridge_state_dict': self.bridge.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state': self.state.__dict__,
            'config': self.config.__dict__
        }, path)
        
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.bridge.load_state_dict(checkpoint['bridge_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.state = TrainingState(**checkpoint['state'])
        
        print(f"Loaded checkpoint from {path} (step {self.state.step})")
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_prompts: List[Tuple[str, str]],
        max_new_tokens: int = 256,
        verbose: bool = False
    ) -> dict:
        """
        Evaluate Bridge effectiveness on held-out problems.
        
        Tests if Student with injected Teacher reasoning can solve problems
        better than Student alone. This is the ground-truth metric that
        matters more than consistency loss.
        
        Args:
            eval_prompts: List of (question, ground_truth_answer) tuples
            max_new_tokens: Max tokens for Student generation
            verbose: Print individual results
            
        Returns:
            Dict with accuracy metrics:
            - with_bridge_accuracy: Accuracy using projected Teacher cache
            - without_bridge_accuracy: Student alone baseline
            - improvement: Absolute improvement
        """
        self.bridge.eval()
        
        with_bridge_correct = 0
        without_bridge_correct = 0
        total = 0
        
        for question, ground_truth in eval_prompts:
            total += 1
            prompt = create_thinking_prompt(question)
            
            # === Student WITH Bridge (injected Teacher reasoning) ===
            try:
                # 1. Teacher generates reasoning
                teacher_ids, teacher_cache = self.model_pair.teacher_generate(
                    prompt,
                    max_new_tokens=self.config.max_thinking_tokens
                )
                
                # 2. Extract and slice cache at </think>
                teacher_cache = extract_kv_cache(
                    type('Output', (), {'past_key_values': teacher_cache})()
                )
                sliced_cache, _ = slice_cache_at_token(
                    teacher_cache,
                    teacher_ids,
                    self.model_pair.tokenizer,
                    end_token="</think>",
                    include_end_token=True
                )
                
                # 3. Project through Bridge
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    projected_cache = self.bridge(sliced_cache)
                projected_cache_tuple = tuple(projected_cache)
                
                # 4. Student generates answer with injected cache
                query_text = self.config.handoff_prompt
                query_ids = self.model_pair.tokenizer.encode(
                    query_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model_pair.student.generate(
                    query_ids,
                    past_key_values=projected_cache_tuple,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.model_pair.tokenizer.pad_token_id
                )
                
                with_bridge_answer = self.model_pair.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                
                # Extract and compare answer
                with_bridge_num = extract_numeric_answer(with_bridge_answer)
                gt_num = extract_numeric_answer(ground_truth)
                
                if with_bridge_num is not None and gt_num is not None:
                    if abs(with_bridge_num - gt_num) < 0.001:
                        with_bridge_correct += 1
                
            except Exception as e:
                if verbose:
                    print(f"Error (with bridge): {e}")
            
            # === Student WITHOUT Bridge (baseline) ===
            try:
                student_prompt = f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
                input_ids = self.model_pair.tokenizer.encode(
                    student_prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model_pair.student.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.model_pair.tokenizer.pad_token_id
                )
                
                without_bridge_answer = self.model_pair.tokenizer.decode(
                    outputs[0][len(input_ids[0]):], skip_special_tokens=True
                )
                
                without_bridge_num = extract_numeric_answer(without_bridge_answer)
                
                if without_bridge_num is not None and gt_num is not None:
                    if abs(without_bridge_num - gt_num) < 0.001:
                        without_bridge_correct += 1
                
            except Exception as e:
                if verbose:
                    print(f"Error (without bridge): {e}")
            
            if verbose:
                print(f"Q: {question[:50]}...")
                print(f"  GT: {ground_truth}")
                print(f"  With Bridge: {with_bridge_num} | Without: {without_bridge_num}")
        
        self.bridge.train()
        
        with_acc = with_bridge_correct / max(total, 1)
        without_acc = without_bridge_correct / max(total, 1)
        
        return {
            'with_bridge_accuracy': with_acc,
            'without_bridge_accuracy': without_acc,
            'improvement': with_acc - without_acc,
            'with_bridge_correct': with_bridge_correct,
            'without_bridge_correct': without_bridge_correct,
            'total': total
        }


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from model output.
    
    Handles formats like:
    - "The answer is 42"
    - "#### 42"
    - "42"
    - "42.5"
    """
    import re
    
    # Try #### format first (GSM8K style)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    
    # Try "answer is X" format
    match = re.search(r'answer\s+(?:is|=)\s*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    
    # Try to find last number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    
    return None


def simple_prompt_iterator(prompts: list) -> Iterator[str]:
    """Simple iterator over a list of prompts."""
    for prompt in prompts:
        yield prompt


def create_thinking_prompt(question: str) -> str:
    """Wrap a question in Qwen3's thinking format."""
    return f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>
Let me think through this step by step.
"""
