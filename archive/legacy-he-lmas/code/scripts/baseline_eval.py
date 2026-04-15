#!/usr/bin/env python3
"""
He-LMAS Baseline Evaluation

Run the torture tests on raw Qwen3-1.7B and Qwen3-8B to establish:
1. Which tests the 1.7B model fails
2. Confirmation that 8B passes these tests

This creates the baseline that He-LMAS aims to improve upon.

Usage:
    python scripts/baseline_eval.py --output results/baseline.json
    python scripts/baseline_eval.py --model-only 1.7b  # Just test 1.7B
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_torture_tests(path: str = "data/torture_tests.json") -> list:
    """Load torture test cases."""
    with open(path, 'r') as f:
        return json.load(f)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 500
) -> str:
    """Generate a response from the model."""
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    
    # Use apply_chat_template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted = f"User: {prompt}\nAssistant:"
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Low temperature for consistency
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def grade_response(test: Dict[str, Any], response: str) -> Dict[str, Any]:
    """
    Grade a model's response against expected criteria.
    
    Returns grading results with explanation.
    """
    grading_type = test.get("grading", "exact_match")
    result = {
        "passed": False,
        "grading_type": grading_type,
        "explanation": ""
    }
    
    response_lower = response.lower()
    
    if grading_type == "exact_match":
        # Check if expected answer components are in response
        expected = test.get("expected_answer_contains", [])
        found = []
        missing = []
        
        for term in expected:
            if term.lower() in response_lower:
                found.append(term)
            else:
                missing.append(term)
        
        result["passed"] = len(missing) == 0
        result["found_terms"] = found
        result["missing_terms"] = missing
        result["explanation"] = f"Found {len(found)}/{len(expected)} expected terms"
    
    elif grading_type == "numeric":
        # Extract numbers from response and check against expected
        expected = test.get("expected_answer")
        tolerance = test.get("tolerance", 0)
        
        # Find all numbers in response
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        numbers = [float(n) for n in numbers]
        
        # Check if expected answer appears
        for num in numbers:
            if abs(num - expected) <= tolerance:
                result["passed"] = True
                result["found_value"] = num
                break
        
        result["expected"] = expected
        result["found_numbers"] = numbers[:10]  # Limit for readability
        result["explanation"] = f"Expected {expected}, found numbers: {numbers[:5]}"
    
    elif grading_type == "code_constraints":
        # Check for forbidden and required patterns
        forbidden = test.get("forbidden_patterns", [])
        required = test.get("required_patterns", [])
        
        violations = [p for p in forbidden if p in response]
        present_required = [p for p in required if p in response]
        
        result["passed"] = len(violations) == 0 and len(present_required) == len(required)
        result["violations"] = violations
        result["missing_required"] = [p for p in required if p not in response]
        result["explanation"] = f"Violations: {violations}, Missing: {result['missing_required']}"
    
    return result


def run_evaluation(
    model_name: str,
    tests: list,
    device: str = "cuda:0",
    quantization: str = "int4"
) -> Dict[str, Any]:
    """
    Run all torture tests on a model.
    
    Args:
        model_name: HuggingFace model ID
        tests: List of test cases
        device: Device to run on
        quantization: "int4", "int8", or None
    
    Returns:
        Evaluation results
    """
    print(f"\nLoading model: {model_name}")
    
    # Load model with quantization
    from transformers import BitsAndBytesConfig
    
    quant_config = None
    if quantization == "int4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": device
    }
    if quant_config:
        load_kwargs["quantization_config"] = quant_config
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    results = {
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": []
    }
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n  Running: {test['name']}...")
        
        start_time = time.time()
        response = generate_response(model, tokenizer, test["prompt"])
        elapsed = time.time() - start_time
        
        grading = grade_response(test, response)
        
        test_result = {
            "id": test["id"],
            "name": test["name"],
            "category": test["category"],
            "passed": grading["passed"],
            "expected_to_pass": test.get(f"expected_{'8b' if '8B' in model_name else '1_7b'}", "unknown"),
            "response": response[:500] + "..." if len(response) > 500 else response,
            "grading": grading,
            "time_seconds": elapsed
        }
        
        results["tests"].append(test_result)
        
        status = "✅ PASS" if grading["passed"] else "❌ FAIL"
        print(f"    {status} | {grading['explanation']}")
        
        if grading["passed"]:
            passed += 1
    
    results["summary"] = {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0
    }
    
    print(f"\n  Summary: {passed}/{total} passed ({results['summary']['pass_rate']:.1%})")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run He-LMAS baseline evaluation")
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--tests-file",
        type=str,
        default="data/torture_tests.json",
        help="Path to torture tests JSON"
    )
    parser.add_argument(
        "--model-only",
        type=str,
        choices=["1.7b", "8b"],
        default=None,
        help="Only test one model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int4", "int8", "none"],
        default="int4",
        help="Quantization type"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("He-LMAS Baseline Evaluation")
    print("=" * 60)
    
    # Load tests
    tests = load_torture_tests(args.tests_file)
    print(f"Loaded {len(tests)} torture tests")
    
    results = {}
    quant = None if args.quantization == "none" else args.quantization
    
    # Run evaluations
    if args.model_only != "8b":
        print("\n" + "=" * 40)
        print("Evaluating: Qwen3-1.7B (Student)")
        print("=" * 40)
        results["student_1.7b"] = run_evaluation(
            "Qwen/Qwen3-1.7B",
            tests,
            device=args.device,
            quantization=quant
        )
    
    if args.model_only != "1.7b":
        print("\n" + "=" * 40)
        print("Evaluating: Qwen3-8B (Teacher)")
        print("=" * 40)
        results["teacher_8b"] = run_evaluation(
            "Qwen/Qwen3-8B",
            tests,
            device=args.device,
            quantization=quant
        )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print comparison summary
    if "student_1.7b" in results and "teacher_8b" in results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        s_rate = results["student_1.7b"]["summary"]["pass_rate"]
        t_rate = results["teacher_8b"]["summary"]["pass_rate"]
        
        print(f"  Qwen3-1.7B (Student): {s_rate:.1%}")
        print(f"  Qwen3-8B (Teacher):   {t_rate:.1%}")
        print(f"  Gap (Potential Gain): {(t_rate - s_rate) * 100:.1f}%")
        print()
        
        # Per-test comparison
        print("  Per-Test Comparison:")
        s_tests = {t["id"]: t for t in results["student_1.7b"]["tests"]}
        t_tests = {t["id"]: t for t in results["teacher_8b"]["tests"]}
        
        for test_id in s_tests:
            s = "✅" if s_tests[test_id]["passed"] else "❌"
            t = "✅" if t_tests[test_id]["passed"] else "❌"
            name = s_tests[test_id]["name"][:35]
            transferable = "🎯" if not s_tests[test_id]["passed"] and t_tests[test_id]["passed"] else "  "
            print(f"    {transferable} {name:38} | 1.7B: {s} | 8B: {t}")
        
        print("\n  🎯 = Transferable via He-LMAS (8B passes, 1.7B fails)")


if __name__ == "__main__":
    main()
