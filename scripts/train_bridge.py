#!/usr/bin/env python3
"""
He-LMAS Bridge Training Script

Train the Heterogeneous Bridge to project KV caches from Qwen3-8B to Qwen3-1.7B.

Usage:
    # Quick test with sample prompts
    python scripts/train_bridge.py --config configs/default.yaml --dry-run
    
    # Full training with GSM8K (downloads automatically)
    python scripts/train_bridge.py --config configs/h200.yaml --dataset gsm8k
    
    # Full training with OpenMathInstruct-2
    python scripts/train_bridge.py --config configs/h200.yaml --dataset openmath --max-samples 100000
    
    # Resume from checkpoint
    python scripts/train_bridge.py --config configs/h200.yaml --checkpoint checkpoints/bridge_step_5000.pt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import HeLMAS_Trainer, create_thinking_prompt
from src.data_loader import create_data_iterator, get_training_data, get_validation_data


# Sample prompts for dry-run testing
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
        help="Run 5 steps with sample prompts to verify setup"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "openmath", "metamath", "math", "sample"],
        default="sample",
        help="Dataset to train on (default: sample prompts)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size (useful for openmath which has 14M samples)"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to file with training prompts (one per line, overrides --dataset)"
    )
    
    # Validation options
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        help="Number of validation samples (0 to disable validation)"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation entirely"
    )
    
    return parser.parse_args()


def create_sample_iterator():
    """Create infinite iterator over sample prompts."""
    while True:
        for prompt in SAMPLE_PROMPTS:
            yield create_thinking_prompt(prompt)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("He-LMAS Bridge Training")
    print("=" * 60)
    
    # Determine data source
    if args.dry_run:
        print("\n[DRY RUN] Using sample prompts, 5 steps only")
        data_iterator = create_sample_iterator()
        max_steps = 5
    elif args.prompts_file:
        # Load from custom file
        print(f"\nLoading prompts from: {args.prompts_file}")
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(prompts)} prompts")
        
        def file_iterator():
            while True:
                for p in prompts:
                    yield create_thinking_prompt(p)
        
        data_iterator = file_iterator()
        max_steps = args.max_steps
        use_batched = False
    elif args.dataset == "sample":
        # Use built-in sample prompts
        print("\nUsing sample prompts (use --dataset for real training)")
        data_iterator = create_sample_iterator()
        max_steps = args.max_steps
        use_batched = False
    else:
        # Use HuggingFace dataset
        print(f"\n📊 Loading dataset: {args.dataset}")
        if args.max_samples:
            print(f"   Max samples: {args.max_samples}")
        
        data_iterator = get_training_data(
            dataset=args.dataset,
            max_samples=args.max_samples
        )
        max_steps = args.max_steps
        use_batched = False  # Will be updated below based on batch_size
    
    # Load validation data (unless disabled)
    eval_data = None
    if not args.no_validation and not args.dry_run and args.eval_samples > 0:
        # Use GSM8K test split for validation (standard benchmark)
        eval_data = get_validation_data(
            dataset="gsm8k",
            max_samples=args.eval_samples
        )
        print(f"✓ Validation enabled: {len(eval_data)} samples from GSM8K test")
    else:
        print("✗ Validation disabled")
    
    # Create trainer with validation data
    print(f"\n🔧 Loading models from config: {args.config}")
    trainer = HeLMAS_Trainer.from_config(args.config, eval_data=eval_data)
    
    # Check if we should use batched training
    batch_size = trainer.config.batch_size
    if batch_size > 1 and args.dataset not in ["sample", None] and not args.dry_run and not args.prompts_file:
        print(f"\n⚡ Using TRUE BATCHING with batch_size={batch_size}")
        # Switch to batched data iterator
        from src.data_loader import get_batched_training_data
        data_iterator = get_batched_training_data(
            dataset=args.dataset,
            batch_size=batch_size,
            max_samples=args.max_samples
        )
        use_batched = True
    else:
        print(f"\n📝 Using single-sample training (batch_size=1 or sample data)")
        use_batched = False
    
    # Resume from checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    try:
        if use_batched:
            trainer.train_batched(data_iterator, max_steps=max_steps)
        else:
            trainer.train(data_iterator, max_steps=max_steps)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint(final=True)
    
    print("\n✅ Training complete!")
    print(f"   Checkpoints saved to: {trainer.checkpoint_dir}")
    print("\n   Next steps:")
    print("   1. Run inference demo: python scripts/inference_demo.py")
    print("   2. Push to HuggingFace: ./scripts/push_to_hf.sh")


if __name__ == "__main__":
    main()
