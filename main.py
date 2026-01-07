"""
Main entry point for PrecisionProject
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
from PrecisionProject.model_dumper import ModelDumper
from PrecisionProject.precision_comparator import PrecisionComparator, ComparisonConfig


def create_torch_simple_model_and_input(model_name=None):
    """Create a simple test model"""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
        nn.Softmax(dim=1)
    )
    input_data = torch.randn(8, 10)
    return model, input_data

def create_vllm_model_and_input(model_name=None):
    """Create a simple test model"""
    from vllm import LLM, SamplingParams 
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # model_name = "facebook/opt-125m"
    # model_name = "nickypro/tinyllama-42M-fp32"
    # model_name = "/home/shenpeng/workspace/models/Qwen3-8B-GPTQ-Int4"
    llm = LLM(model=model_name, enforce_eager=True, gpu_memory_utilization=0.4)
    
    mock_input = {"prompts": prompts, "params": {"temperature": 0.0, "max_tokens": 2}}
    return llm,mock_input


def main():
    parser = argparse.ArgumentParser(description="PrecisionProject - Model Precision Testing Framework")
    parser.add_argument("--mode", choices=["dump", "compare", "demo"], default="compare",
                       help="Operation mode: dump (capture traces), compare (precision testing), or demo (run both)")
    parser.add_argument("--model-path", default="/home/shenpeng/workspace/models/Qwen3-0.6B-GPTQ-Int4",
                       help="model path")
    parser.add_argument("--framework", choices=["torch", "vllm"], default="vllm",
                       help="Framework to use")
    parser.add_argument("--golden-path", default="./data/golden",
                       help="Path to golden reference data")
    parser.add_argument("--test-path", default="./data/test",
                       help="Path to test data")
    parser.add_argument("--output-path", default="./data/output",
                       help="Path to save results")
    parser.add_argument("--abs-threshold", type=float, default=1e-2,
                       help="Absolute error threshold")
    parser.add_argument("--rel-threshold", type=float, default=1e-2,
                       help="Relative error threshold")
    parser.add_argument("--cosine-threshold", type=float, default=0.99,
                       help="Cosine similarity threshold")
    parser.add_argument("--dump-golden", type=bool, default=True,
                       help="dump mode need set dump golden")

    args = parser.parse_args()

    if args.mode == "demo":
        # Demo mode: run dump and compare

        # Create golden reference
        print("Creating golden reference...")
        model,mock_input = create_vllm_model_and_input(args.model_path)
        golden_dumper = ModelDumper(framework=args.framework, model_path=args.model_path)
        golden_dumper.dump_model_execution(
            model, mock_input, args.golden_path, args.model_path, iterations=1
        )
        del model
        del golden_dumper
        # Create test data with slight perturbation
        print("Creating test reference with perturbation...")
        test_model,test_mock_input = create_vllm_model_and_input(args.model_path)
        if args.framework == "torch":
            test_model.load_state_dict(model.state_dict())
            # Add small noise to test model
            with torch.no_grad():
                for param in test_model.parameters():
                    param += torch.randn_like(param) * 1e-6

        test_dumper = ModelDumper(framework=args.framework, model_path=args.model_path)
        test_dumper.dump_model_execution(
            test_model, test_mock_input, args.test_path, args.model_path, iterations=1
        )
        del test_model
        del test_dumper
        # Compare precision
        print("Comparing precision...")
        config = ComparisonConfig(
            abs_error_threshold=args.abs_threshold,
            rel_error_threshold=args.rel_threshold,
            cosine_similarity_threshold=args.cosine_threshold
        )
        comparator = PrecisionComparator(config)
        results, summary = comparator.compare_traces(
            args.golden_path, args.test_path, args.output_path
        )

        print(f"\nDemo completed! Results saved to {args.output_path}")
        print(f"Pass rate: {summary['pass_rate']:.2%}")

    elif args.mode == "dump":
        # Dump mode: just capture traces
        print(f"Dumping model traces to {args.test_path}")
        model, mock_input = create_vllm_model_and_input(args.model_path)
        dumper = ModelDumper(framework=args.framework, model_path=args.model_path)

        dumper.dump_model_execution(
            model, mock_input, args.golden_path if args.dump_golden else args.test_path, args.model_path, iterations=1
        )
        del model
        del dumper
    elif args.mode == "compare":
        # Compare mode: just compare existing traces
        print(f"Comparing traces from {args.golden_path} and {args.test_path}")
        config = ComparisonConfig(
            abs_error_threshold=args.abs_threshold,
            rel_error_threshold=args.rel_threshold,
            cosine_similarity_threshold=args.cosine_threshold
        )
        comparator = PrecisionComparator(config)
        results, summary = comparator.compare_traces(
            args.golden_path, args.test_path, args.output_path
        )

        print(f"Comparison completed! Results saved to {args.output_path}")
        print(f"Pass rate: {summary['pass_rate']:.2%}")


if __name__ == "__main__":
    main()