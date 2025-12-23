"""
Main entry point for PrecisionProject
"""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
from PrecisionProject.model_dumper import ModelDumper
from PrecisionProject.precision_comparator import PrecisionComparator, ComparisonConfig


def create_simple_model():
    """Create a simple test model"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
        nn.Softmax(dim=1)
    )


def main():
    parser = argparse.ArgumentParser(description="PrecisionProject - Model Precision Testing Framework")
    parser.add_argument("--mode", choices=["dump", "compare", "demo"], default="demo",
                       help="Operation mode: dump (capture traces), compare (precision testing), or demo (run both)")
    parser.add_argument("--framework", choices=["torch", "vllm"], default="torch",
                       help="Framework to use")
    parser.add_argument("--golden-path", default="./data/golden",
                       help="Path to golden reference data")
    parser.add_argument("--test-path", default="./data/test",
                       help="Path to test data")
    parser.add_argument("--output-path", default="./data/output",
                       help="Path to save results")
    parser.add_argument("--abs-threshold", type=float, default=1e-5,
                       help="Absolute error threshold")
    parser.add_argument("--rel-threshold", type=float, default=1e-5,
                       help="Relative error threshold")
    parser.add_argument("--cosine-threshold", type=float, default=0.99999,
                       help="Cosine similarity threshold")

    args = parser.parse_args()

    if args.mode == "demo":
        # Demo mode: run dump and compare
        print("Running PrecisionProject demo...")

        # Create test data
        input_data = torch.randn(8, 10)

        # Create golden reference
        print("Creating golden reference...")
        golden_dumper = ModelDumper(framework="torch")
        model = create_simple_model()
        golden_dumper.dump_model_execution(
            model, input_data, args.golden_path, "demo_model", iterations=2
        )

        # Create test data with slight perturbation
        print("Creating test reference with perturbation...")
        test_model = create_simple_model()
        test_model.load_state_dict(model.state_dict())
        # Add small noise to test model
        with torch.no_grad():
            for param in test_model.parameters():
                param += torch.randn_like(param) * 1e-6

        test_dumper = ModelDumper(framework="torch")
        test_dumper.dump_model_execution(
            test_model, input_data, args.test_path, "demo_model", iterations=2
        )

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
        print(f"Dumping model traces to {args.output_path}")
        dumper = ModelDumper(framework=args.framework)

        if args.framework == "torch":
            model = create_simple_model()
            input_data = torch.randn(8, 10)
            dumper.dump_model_execution(
                model, input_data, args.output_path, "test_model", iterations=5
            )
        else:
            print("vLLM dump mode requires specific model setup")

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