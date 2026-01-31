#!/usr/bin/env python3
"""
He-LMAS Bridge Training Script

Train the Heterogeneous Bridge to project KV caches from Qwen3-8B to Qwen3-1.7B.

Usage:
    python scripts/train_bridge.py --config configs/default.yaml
    python scripts/train_bridge.py --config configs/default.yaml --max-steps 1000 --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import HeLMAS_Trainer, simple_prompt_iterator, create_thinking_prompt


# Sample prompts for training (replace with proper dataset)
SAMPLE_PROMPTS = [
    "Solve for x: 3x + 5 = 20. Show your work step by step.",
    "A train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours. What is the total distance?",
    "If a rectangle has a perimeter of 24 and a length twice its width, what are the dimensions?",
    "Calculate: (15 × 8) - (12 × 4) + 25",
    "A store offers 20% off. If an item costs $45, what is the final price?",
    "Three consecutive even numbers sum to 84. What are the numbers?",
    "A car depreciates by 15% each year. If it costs $20,000 new, what is it worth after 2 years?",
    "Simplify: (x^2 + 5x + 6) / (x + 2)",
    "A pizza is cut into 8 equal slices. If you eat 3 slices and your friend eats 2, what fraction remains?",
    "Calculate the area of a triangle with base 12 and height 8.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train He-LMAS Bridge for KV cache projection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a few steps to verify setup"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to file with training prompts (one per line)"
    )
    return parser.parse_args()


def load_prompts(prompts_file: str = None) -> list:
    """Load training prompts from file or use samples."""
    if prompts_file is not None:
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {prompts_file}")
        return prompts
    
    print(f"Using {len(SAMPLE_PROMPTS)} sample prompts (provide --prompts-file for real training)")
    return SAMPLE_PROMPTS


def main():
    args = parse_args()
    
    print("=" * 60)
    print("He-LMAS Bridge Training")
    print("=" * 60)
    
    # Load prompts
    prompts = load_prompts(args.prompts_file)
    
    # Wrap prompts in thinking format
    formatted_prompts = [create_thinking_prompt(p) for p in prompts]
    
    # Create trainer
    print(f"\nLoading models from config: {args.config}")
    trainer = HeLMAS_Trainer.from_config(args.config)
    
    # Resume from checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN] Running 5 steps to verify setup...")
        max_steps = 5
    else:
        max_steps = args.max_steps
    
    # Create infinite iterator over prompts (cycle through)
    def infinite_prompts():
        while True:
            for p in formatted_prompts:
                yield p
    
    # Train
    try:
        trainer.train(infinite_prompts(), max_steps=max_steps)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint(final=True)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
