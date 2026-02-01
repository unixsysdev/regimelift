"""
Dataset Loaders for He-LMAS Training

Provides dataset loading and formatting for training the Bridge.
Uses HuggingFace datasets library with automatic caching.

Supported datasets:
- OpenMathInstruct-2 (NVIDIA, 14M samples)
- GSM8K (grade school math, 8.5K samples)
- MetaMathQA (augmented math, 395K samples)
- MATH (competition math, 12.5K samples)
"""

import random
from typing import Iterator, Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str = "gsm8k"  # Default to smaller dataset for testing
    split: str = "train"
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42
    cache_dir: str = "data/cache"


def format_thinking_prompt(question: str, include_thinking_start: bool = True) -> str:
    """
    Format a question for Qwen3's thinking mode.
    
    Args:
        question: The math/reasoning question
        include_thinking_start: Whether to start the thinking trace
        
    Returns:
        Formatted prompt that triggers <think> mode
    """
    prompt = f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    if include_thinking_start:
        prompt += "<think>\nLet me think through this step by step.\n"
    
    return prompt


def load_gsm8k(config: DatasetConfig) -> Iterator[str]:
    """
    Load GSM8K dataset (Grade School Math 8K).
    
    Good for: Testing, smaller scale experiments
    Size: ~7.5K train, ~1.3K test
    """
    from datasets import load_dataset
    
    print(f"Loading GSM8K ({config.split})...")
    dataset = load_dataset(
        "gsm8k",
        "main",
        split=config.split,
        cache_dir=config.cache_dir
    )
    
    if config.shuffle:
        dataset = dataset.shuffle(seed=config.seed)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    print(f"  Loaded {len(dataset)} samples")
    
    for item in dataset:
        question = item["question"]
        yield format_thinking_prompt(question)


def load_openmath_instruct(config: DatasetConfig) -> Iterator[str]:
    """
    Load OpenMathInstruct-2 dataset (NVIDIA).
    
    Good for: Full training, has CoT traces already
    Size: ~14M samples
    
    Note: This is a large dataset, use max_samples for testing.
    """
    from datasets import load_dataset
    
    print(f"Loading OpenMathInstruct-2 ({config.split})...")
    
    # OpenMathInstruct-2 on HuggingFace
    dataset = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split=config.split,
        cache_dir=config.cache_dir,
        streaming=True  # Stream for large dataset
    )
    
    if config.shuffle:
        dataset = dataset.shuffle(seed=config.seed, buffer_size=10000)
    
    count = 0
    for item in dataset:
        if config.max_samples and count >= config.max_samples:
            break
        
        # OpenMathInstruct has 'problem' field
        question = item.get("problem") or item.get("question", "")
        if question:
            yield format_thinking_prompt(question)
            count += 1
    
    print(f"  Yielded {count} samples")


def load_metamath(config: DatasetConfig) -> Iterator[str]:
    """
    Load MetaMathQA dataset.
    
    Good for: Augmented math Q&A with diverse phrasings
    Size: ~395K samples
    """
    from datasets import load_dataset
    
    print(f"Loading MetaMathQA ({config.split})...")
    dataset = load_dataset(
        "meta-math/MetaMathQA",
        split=config.split,
        cache_dir=config.cache_dir
    )
    
    if config.shuffle:
        dataset = dataset.shuffle(seed=config.seed)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    print(f"  Loaded {len(dataset)} samples")
    
    for item in dataset:
        question = item.get("query") or item.get("question", "")
        yield format_thinking_prompt(question)


def load_math_dataset(config: DatasetConfig) -> Iterator[str]:
    """
    Load MATH dataset (competition math).
    
    Good for: Challenging problems, tests upper capability
    Size: ~12.5K samples
    """
    from datasets import load_dataset
    
    print(f"Loading MATH ({config.split})...")
    dataset = load_dataset(
        "hendrycks/competition_math",
        split=config.split,
        cache_dir=config.cache_dir
    )
    
    if config.shuffle:
        dataset = dataset.shuffle(seed=config.seed)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    print(f"  Loaded {len(dataset)} samples")
    
    for item in dataset:
        question = item["problem"]
        yield format_thinking_prompt(question)


def create_data_iterator(
    dataset_name: str = "gsm8k",
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    cache_dir: str = "data/cache",
    infinite: bool = True
) -> Iterator[str]:
    """
    Create a data iterator for training.
    
    Args:
        dataset_name: One of "gsm8k", "openmath", "metamath", "math"
        split: Dataset split ("train", "test")
        max_samples: Limit number of samples (None = all)
        shuffle: Whether to shuffle
        seed: Random seed
        cache_dir: Where to cache downloaded data
        infinite: If True, loop forever (for training)
        
    Returns:
        Iterator yielding formatted prompts
    """
    config = DatasetConfig(
        name=dataset_name,
        split=split,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
        cache_dir=cache_dir
    )
    
    loaders = {
        "gsm8k": load_gsm8k,
        "openmath": load_openmath_instruct,
        "metamath": load_metamath,
        "math": load_math_dataset,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(loaders.keys())}")
    
    loader = loaders[dataset_name]
    
    if infinite:
        while True:
            for prompt in loader(config):
                yield prompt
            print(f"  [Dataset epoch complete, restarting...]")
    else:
        for prompt in loader(config):
            yield prompt


def create_mixed_iterator(
    datasets: List[Dict[str, Any]],
    infinite: bool = True
) -> Iterator[str]:
    """
    Create a mixed iterator from multiple datasets.
    
    Args:
        datasets: List of dataset configs with optional weights
                  e.g., [{"name": "gsm8k", "weight": 0.3}, {"name": "metamath", "weight": 0.7}]
        infinite: If True, loop forever
        
    Returns:
        Iterator yielding mixed prompts
    """
    # Normalize weights
    total_weight = sum(d.get("weight", 1.0) for d in datasets)
    weights = [d.get("weight", 1.0) / total_weight for d in datasets]
    
    # Create iterators
    iterators = [
        create_data_iterator(
            dataset_name=d["name"],
            max_samples=d.get("max_samples"),
            shuffle=d.get("shuffle", True),
            infinite=True  # Each sub-iterator is infinite
        )
        for d in datasets
    ]
    
    print(f"Created mixed iterator with {len(datasets)} datasets:")
    for d, w in zip(datasets, weights):
        print(f"  - {d['name']}: {w:.1%}")
    
    while True:
        # Weighted random selection
        idx = random.choices(range(len(iterators)), weights=weights)[0]
        yield next(iterators[idx])
        
        if not infinite:
            break


# Convenience function for common use case
def get_training_data(
    dataset: str = "gsm8k",
    max_samples: Optional[int] = None
) -> Iterator[str]:
    """
    Get training data iterator with sensible defaults.
    
    Args:
        dataset: Dataset name
        max_samples: Limit samples (None = all)
        
    Returns:
        Infinite iterator of formatted prompts
    """
    return create_data_iterator(
        dataset_name=dataset,
        split="train",
        max_samples=max_samples,
        shuffle=True,
        infinite=True
    )


def get_validation_data(
    dataset: str = "gsm8k",
    max_samples: int = 50,
    cache_dir: str = "data/cache"
) -> List[tuple]:
    """
    Get validation data as (question, answer) tuples.
    
    Uses the test split for evaluation.
    
    Args:
        dataset: Dataset name ("gsm8k" recommended for validation)
        max_samples: Number of samples for validation
        cache_dir: Where to cache downloaded data
        
    Returns:
        List of (question, ground_truth_answer) tuples
    """
    from datasets import load_dataset
    
    print(f"📊 Loading validation data: {dataset} (test split, {max_samples} samples)")
    
    if dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
        pairs = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            # GSM8K has "question" and "answer" fields
            # answer contains the full solution with #### final_answer
            pairs.append((item["question"], item["answer"]))
        print(f"  Loaded {len(pairs)} validation samples")
        return pairs
    
    elif dataset == "math":
        ds = load_dataset("hendrycks/competition_math", split="test", cache_dir=cache_dir)
        pairs = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            # MATH has "problem" and "solution" fields
            pairs.append((item["problem"], item["solution"]))
        print(f"  Loaded {len(pairs)} validation samples")
        return pairs
    
    else:
        print(f"  Warning: {dataset} doesn't have a standard test split, using GSM8K")
        return get_validation_data("gsm8k", max_samples, cache_dir)
