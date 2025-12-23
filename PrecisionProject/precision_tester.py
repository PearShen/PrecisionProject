"""
Main precision tester class that combines dumping and comparison functionality.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from .model_dumper import ModelDumper
from .precision_comparator import PrecisionComparator, ComparisonConfig


class PrecisionTester:
    """Main class for precision testing of models"""

    def __init__(self,
                 framework: str = "torch",
                 comparison_config: Optional[ComparisonConfig] = None):
        """
        Initialize the precision tester.

        Args:
            framework: Framework type, either "torch" or "vllm"
            comparison_config: Configuration for precision comparison
        """
        self.framework = framework
        self.dumper = ModelDumper(framework=framework)
        self.comparator = PrecisionComparator(comparison_config)

    def dump_model_execution(self,
                            model: Any,
                            input_data: Any,
                            output_path: str,
                            model_name: str = "model",
                            iterations: int = 1) -> None:
        """Dump model execution with operator traces"""
        self.dumper.dump_model_execution(
            model, input_data, output_path, model_name, iterations
        )

    def compare_precision(self,
                         golden_path: str,
                         test_path: str,
                         output_path: Optional[str] = None) -> tuple:
        """Compare precision between golden and test data"""
        return self.comparator.compare_traces(golden_path, test_path, output_path)

    def test_model_precision(self,
                           model: Any,
                           input_data: Any,
                           output_dir: str,
                           model_name: str = "model",
                           iterations: int = 1,
                           perturbation: float = 1e-6) -> Dict:
        """
        Complete precision test workflow:
        1. Create golden reference
        2. Create perturbed test version
        3. Compare precision

        Args:
            model: The model to test
            input_data: Input data for the model
            output_dir: Directory to save all results
            model_name: Name of the model
            iterations: Number of forward passes
            perturbation: Amount of perturbation to apply to test model

        Returns:
            Dictionary containing test results and summary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Paths
        golden_path = os.path.join(output_dir, "golden")
        test_path = os.path.join(output_dir, "test")
        results_path = os.path.join(output_dir, "results")

        # Step 1: Create golden reference
        print(f"Creating golden reference at {golden_path}")
        self.dump_model_execution(model, input_data, golden_path, model_name, iterations)

        # Step 2: Create perturbed test version
        print(f"Creating perturbed test version at {test_path}")
        if self.framework == "torch" and isinstance(model, nn.Module):
            test_model = self._create_perturbed_model(model, perturbation)
            self.dump_model_execution(test_model, input_data, test_path, model_name, iterations)
        else:
            # For non-PyTorch models, use the same model (no perturbation)
            self.dump_model_execution(model, input_data, test_path, model_name, iterations)

        # Step 3: Compare precision
        print(f"Comparing precision and saving results to {results_path}")
        results, summary = self.compare_precision(golden_path, test_path, results_path)

        return {
            "golden_path": golden_path,
            "test_path": test_path,
            "results_path": results_path,
            "results": results,
            "summary": summary
        }

    def _create_perturbed_model(self, model: nn.Module, perturbation: float) -> nn.Module:
        """Create a perturbed version of the model for testing"""
        if self.framework == "torch":
            import copy
            perturbed_model = copy.deepcopy(model)
            with torch.no_grad():
                for param in perturbed_model.parameters():
                    if param.requires_grad:
                        param.add_(torch.randn_like(param) * perturbation)

            return perturbed_model
        else:
            return model

    def load_and_compare_dumps(self, dump_paths: List[str], output_path: str) -> Dict:
        """
        Compare multiple dumps against a baseline (first in the list).

        Args:
            dump_paths: List of dump directories to compare
            output_path: Path to save comparison results

        Returns:
            Dictionary containing all comparison results
        """
        if len(dump_paths) < 2:
            raise ValueError("At least 2 dump paths are required for comparison")

        baseline_path = dump_paths[0]
        results = {}

        os.makedirs(output_path, exist_ok=True)

        for i, test_path in enumerate(dump_paths[1:], 1):
            test_name = f"comparison_{i}_{os.path.basename(test_path)}"
            test_output_path = os.path.join(output_path, test_name)

            comparison_results, summary = self.compare_precision(
                baseline_path, test_path, test_output_path
            )

            results[test_name] = {
                "results": comparison_results,
                "summary": summary,
                "test_path": test_path
            }

        # Save overall summary
        self._save_overall_summary(results, output_path)

        return results

    def _save_overall_summary(self, results: Dict, output_path: str):
        """Save overall summary of multiple comparisons"""
        import json

        summary_data = {
            "baseline_path": results.get("comparison_1_" + os.path.basename(list(results.values())[0]["test_path"]), {}).get("test_path", "unknown"),
            "comparisons": {}
        }

        for test_name, test_results in results.items():
            summary_data["comparisons"][test_name] = {
                "pass_rate": test_results["summary"]["pass_rate"],
                "max_absolute_error": test_results["summary"]["max_absolute_error"],
                "mean_absolute_error": test_results["summary"]["mean_absolute_error"],
                "max_relative_error": test_results["summary"]["max_relative_error"],
                "mean_relative_error": test_results["summary"]["mean_relative_error"],
                "min_cosine_similarity": test_results["summary"]["min_cosine_similarity"],
                "mean_cosine_similarity": test_results["summary"]["mean_cosine_similarity"]
            }

        with open(os.path.join(output_path, "overall_summary.json"), "w") as f:
            json.dump(summary_data, f, indent=2)

        # Print summary
        print("\nOverall Comparison Summary:")
        print("=" * 50)
        for test_name, test_results in results.items():
            print(f"\n{test_name}:")
            print(f"  Pass rate: {test_results['summary']['pass_rate']:.2%}")
            print(f"  Mean absolute error: {test_results['summary']['mean_absolute_error']:.2e}")
            print(f"  Mean relative error: {test_results['summary']['mean_relative_error']:.2e}")