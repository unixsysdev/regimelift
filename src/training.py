"""
Training Infrastructure for He-LMAS

Implements the "Online Streaming" training loop where KV caches are
generated on-the-fly from the Teacher, projected through the Bridge,
and used to train the Bridge via next-token prediction on the Student.

Key insight: We don't save terabytes of KV caches to disk. We generate
them live, use them once, and discard them.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any, Iterator, Tuple
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


class HeLMAS_Trainer:
    """
    Trainer for the He-LMAS Bridge.
    
    The training loop:
    1. Teacher generates reasoning trace (with <think> tags)
    2. Bridge projects Teacher's KV cache to Student geometry
    3. Student tries to predict continuation using injected cache
    4. Loss backpropagates through Bridge only (models are frozen)
    
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
        log_dir: str = "logs"
    ):
        self.model_pair = model_pair
        self.bridge = bridge
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Move bridge to same device as student
        self.device = next(self.model_pair.student.parameters()).device
        self.bridge = self.bridge.to(self.device)
        
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.state = TrainingState()
        
        # Logging
        self.writer = None  # TensorBoard writer (lazy init)
    
    @classmethod
    def from_config(cls, config_path: str) -> "HeLMAS_Trainer":
        """
        Create trainer from configuration file.
        
        Args:
            config_path: Path to YAML config
            
        Returns:
            Initialized trainer
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load models
        model_pair = HeLMAS_ModelPair.from_config(config_path)
        
        # Create bridge
        bridge = create_bridge_from_config(config)
        
        # Training config
        train_config = TrainingConfig.from_dict(config.get('training', {}))
        
        return cls(
            model_pair=model_pair,
            bridge=bridge,
            config=train_config,
            checkpoint_dir=config.get('paths', {}).get('checkpoints', 'checkpoints'),
            log_dir=config.get('paths', {}).get('logs', 'logs')
        )
    
    def train_step(self, prompt: str, target_continuation: Optional[str] = None) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            prompt: Input prompt for Teacher
            target_continuation: Optional target (if None, use Teacher's output)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # 1. Teacher: Generate reasoning trace
        with torch.no_grad():
            teacher_ids, teacher_cache = self.model_pair.teacher_generate(
                prompt,
                max_new_tokens=self.config.max_thinking_tokens
            )
        
        # 2. Slice cache at </think> if present
        teacher_cache = extract_kv_cache(type('Output', (), {'past_key_values': teacher_cache})())
        
        sliced_cache, cut_pos = slice_cache_at_token(
            teacher_cache,
            teacher_ids,
            self.model_pair.tokenizer,
            end_token="</think>",
            include_end_token=True
        )
        
        metrics['teacher_thinking_tokens'] = cut_pos
        metrics['teacher_cache_mb'] = kv_cache_info(sliced_cache)['memory_mb']
        
        # 3. Bridge: Project cache
        projected_cache = self.bridge(sliced_cache)
        
        # Convert to tuple format expected by HuggingFace
        projected_cache_tuple = tuple(projected_cache)
        
        # 4. Prepare handoff tokens
        handoff_text = "\nAnswer:"
        handoff_ids = self.model_pair.tokenizer.encode(
            handoff_text, 
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)
        
        # 5. Get target: Either provided or from Teacher's continuation
        if target_continuation is not None:
            target_ids = self.model_pair.tokenizer.encode(
                target_continuation,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.device)
        else:
            # Use tokens after the thinking section
            target_ids = teacher_ids[:, cut_pos:cut_pos + 20].to(self.device)
        
        # 6. Student forward with injected cache
        outputs = self.model_pair.student_forward(
            input_ids=handoff_ids,
            past_key_values=projected_cache_tuple
        )
        
        # 7. Calculate loss
        # We compare Student's prediction to the target continuation
        logits = outputs.logits[:, -1, :]  # Last token prediction
        
        if target_ids.shape[1] > 0:
            target_token = target_ids[:, 0]  # Predict first target token
            loss = self.criterion(logits, target_token)
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        metrics['loss'] = loss.item()
        
        # 8. Backward and optimize
        loss = loss / self.config.gradient_accumulation
        loss.backward()
        
        if (self.state.step + 1) % self.config.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            metrics['lr'] = self.scheduler.get_last_lr()[0]
        
        # 9. Cleanup VRAM
        del teacher_cache, sliced_cache, projected_cache, outputs
        torch.cuda.empty_cache()
        
        return metrics
    
    def train(
        self,
        data_iterator: Iterator[str],
        max_steps: Optional[int] = None
    ):
        """
        Main training loop.
        
        Args:
            data_iterator: Iterator yielding prompts
            max_steps: Override max_steps from config
        """
        max_steps = max_steps or self.config.max_steps
        
        self.bridge.train()
        self.optimizer.zero_grad()
        
        running_loss = 0.0
        step_times = []
        
        print(f"Starting training for {max_steps} steps...")
        print(f"Bridge parameters: {self.bridge.num_parameters():,}")
        
        for prompt in data_iterator:
            if self.state.step >= max_steps:
                break
            
            step_start = time.time()
            
            try:
                metrics = self.train_step(prompt)
                
                running_loss += metrics['loss']
                self.state.step += 1
                self.state.samples_seen += 1
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Logging
                if self.state.step % self.config.log_every == 0:
                    avg_loss = running_loss / self.config.log_every
                    avg_time = sum(step_times[-self.config.log_every:]) / len(step_times[-self.config.log_every:])
                    
                    print(
                        f"Step {self.state.step}/{max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Time: {avg_time:.2f}s | "
                        f"LR: {metrics.get('lr', self.config.learning_rate):.2e}"
                    )
                    
                    running_loss = 0.0
                
                # Checkpointing
                if self.state.step % self.config.checkpoint_every == 0:
                    self.save_checkpoint()
                    
            except Exception as e:
                print(f"Error at step {self.state.step}: {e}")
                continue
        
        # Final checkpoint
        self.save_checkpoint(final=True)
        print(f"Training complete. Final step: {self.state.step}")
    
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


def simple_prompt_iterator(prompts: list) -> Iterator[str]:
    """Simple iterator over a list of prompts."""
    for prompt in prompts:
        yield prompt


def create_thinking_prompt(question: str) -> str:
    """
    Wrap a question in Qwen3's thinking format.
    
    Args:
        question: The question to think about
        
    Returns:
        Formatted prompt that triggers <think> mode
    """
    return f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>
Let me think through this step by step.
"""
