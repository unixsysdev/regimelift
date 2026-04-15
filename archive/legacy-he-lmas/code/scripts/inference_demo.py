#!/usr/bin/env python3
"""
He-LMAS Inference Demo

Demonstrate the full He-LMAS pipeline:
1. Teacher thinks about a problem
2. Bridge projects the KV cache
3. Student answers using the injected memory

Compare outputs: raw 1.7B vs He-LMAS 1.7B vs 8B

Usage:
    python scripts/inference_demo.py --checkpoint checkpoints/bridge_final.pt
    python scripts/inference_demo.py --prompt "Solve for x: 2x + 5 = 15"
"""

import argparse
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.model_loader import HeLMAS_ModelPair
from src.bridge import create_bridge_from_config
from src.kv_cache_utils import extract_kv_cache, slice_cache_at_token, prepare_injection_input
from src.training import create_thinking_prompt


def load_bridge(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained bridge from checkpoint."""
    bridge = create_bridge_from_config(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bridge.load_state_dict(checkpoint['bridge_state_dict'])
    bridge = bridge.to(device)
    bridge.eval()
    
    print(f"Loaded bridge from {checkpoint_path}")
    print(f"  Training step: {checkpoint.get('state', {}).get('step', 'unknown')}")
    
    return bridge


def generate_raw(model_pair: HeLMAS_ModelPair, prompt: str, use_teacher: bool = False) -> str:
    """Generate response without He-LMAS (baseline)."""
    model = model_pair.teacher if use_teacher else model_pair.student
    tokenizer = model_pair.tokenizer
    
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def generate_with_helmas(
    model_pair: HeLMAS_ModelPair,
    bridge: torch.nn.Module,
    prompt: str,
    max_thinking_tokens: int = 100
) -> tuple:
    """
    Generate using He-LMAS pipeline.
    
    Returns:
        (teacher_thinking, student_answer, timing_info)
    """
    tokenizer = model_pair.tokenizer
    timing = {}
    
    # 1. Teacher: Generate thinking trace
    thinking_prompt = create_thinking_prompt(prompt)
    
    start = time.time()
    teacher_ids, teacher_cache = model_pair.teacher_generate(
        thinking_prompt,
        max_new_tokens=max_thinking_tokens
    )
    timing['teacher_generation'] = time.time() - start
    
    # Decode teacher's thinking
    teacher_output = tokenizer.decode(teacher_ids[0], skip_special_tokens=True)
    
    # 2. Extract and slice cache at </think>
    start = time.time()
    cache = extract_kv_cache(type('Output', (), {'past_key_values': teacher_cache})())
    sliced_cache, cut_pos = slice_cache_at_token(
        cache, teacher_ids, tokenizer, "</think>", include_end_token=True
    )
    timing['cache_extraction'] = time.time() - start
    
    # 3. Bridge: Project cache
    start = time.time()
    with torch.no_grad():
        projected_cache = bridge(sliced_cache)
    timing['bridge_projection'] = time.time() - start
    
    # Convert to tuple
    projected_cache = tuple(projected_cache)
    
    # 4. Student: Generate using injected cache
    handoff_tokens = prepare_injection_input(
        tokenizer,
        handoff_text="\nAnswer:",
        device=model_pair.student.device
    )
    
    start = time.time()
    
    # Generate continuation
    with torch.no_grad():
        outputs = model_pair.student.generate(
            input_ids=handoff_tokens,
            past_key_values=projected_cache,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    timing['student_generation'] = time.time() - start
    
    student_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return teacher_output, student_answer, timing


def parse_args():
    parser = argparse.ArgumentParser(description="He-LMAS Inference Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/bridge_final.pt",
        help="Bridge checkpoint path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (uses default if not specified)"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip raw model comparisons (faster)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("He-LMAS Inference Demo")
    print("=" * 70)
    
    # Default prompt
    prompt = args.prompt or (
        "There are five houses in a row. The Brit lives in the red house. "
        "The Swede has a dog. The Dane drinks tea. The green house is to the left "
        "of the white house. The green house owner drinks coffee. "
        "Who lives in the green house?"
    )
    
    print(f"\nPrompt: {prompt[:100]}...")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load models
    print("\nLoading models...")
    model_pair = HeLMAS_ModelPair.from_config(args.config)
    
    # Load trained bridge
    device = next(model_pair.student.parameters()).device
    bridge = load_bridge(args.checkpoint, config, device)
    
    # Run comparisons
    print("\n" + "=" * 70)
    
    if not args.no_comparison:
        # Raw 1.7B
        print("\n[1] Raw Qwen3-1.7B (Baseline)")
        print("-" * 40)
        start = time.time()
        raw_1_7b = generate_raw(model_pair, prompt, use_teacher=False)
        t_raw = time.time() - start
        print(f"Time: {t_raw:.2f}s")
        print(f"Response:\n{raw_1_7b[:500]}")
        
        # Raw 8B
        print("\n[2] Raw Qwen3-8B (Teacher)")
        print("-" * 40)
        start = time.time()
        raw_8b = generate_raw(model_pair, prompt, use_teacher=True)
        t_8b = time.time() - start
        print(f"Time: {t_8b:.2f}s")
        print(f"Response:\n{raw_8b[:500]}")
    
    # He-LMAS
    print("\n[3] He-LMAS (1.7B + Injected 8B Reasoning)")
    print("-" * 40)
    
    teacher_thinking, student_answer, timing = generate_with_helmas(
        model_pair, bridge, prompt
    )
    
    print(f"Teacher Thinking Time: {timing['teacher_generation']:.2f}s")
    print(f"Bridge Project Time: {timing['bridge_projection']:.4f}s")
    print(f"Student Answer Time: {timing['student_generation']:.2f}s")
    print(f"Total He-LMAS Time: {sum(timing.values()):.2f}s")
    
    print(f"\nTeacher's Thinking:\n{teacher_thinking[:300]}...")
    print(f"\nStudent's Answer:\n{student_answer[:500]}")
    
    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
