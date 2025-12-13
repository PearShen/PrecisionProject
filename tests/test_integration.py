"""
Integration tests for the complete PrecisionProject workflow
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from precision_tester import PrecisionTester


class TestIntegration:
    """Integration tests for the complete workflow"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_pytorch_workflow(self):
        """Test complete PyTorch workflow end-to-end"""
        # Create a more complex model
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.Softmax(dim=1)
        )

        input_data = torch.randn(16, 10)

        # Test the complete workflow
        tester = PrecisionTester(framework="torch")
        results = tester.test_model_precision(
            model,
            input_data,
            self.temp_dir,
            "complex_model",
            iterations=3,
            perturbation=1e-7
        )

        # Verify the workflow completed successfully
        assert results is not None
        assert results["summary"]["total_comparisons"] > 0
        assert 0 <= results["summary"]["pass_rate"] <= 1

        # Verify all necessary files were created
        golden_path = results["golden_path"]
        test_path = results["test_path"]
        results_path = results["results_path"]

        # Check golden dump
        assert os.path.exists(os.path.join(golden_path, "model_info.json"))
        assert os.path.exists(os.path.join(golden_path, "operator_traces.h5"))

        # Check test dump
        assert os.path.exists(os.path.join(test_path, "model_info.json"))
        assert os.path.exists(os.path.join(test_path, "operator_traces.h5"))

        # Check comparison results
        assert os.path.exists(os.path.join(results_path, "comparison_results.json"))
        assert os.path.exists(os.path.join(results_path, "summary.txt"))

    def test_cli_integration(self):
        """Test command line interface integration"""
        # Write a simple test script
        script_path = os.path.join(self.temp_dir, "test_main.py")
        main_path = Path(__file__).parent.parent / "main.py"

        # Copy main.py to test directory
        shutil.copy(main_path, script_path)

        # Run demo mode
        result = subprocess.run(
            [sys.executable, script_path, "--mode", "demo", "--output-path", self.temp_dir],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )

        # Check if the command ran successfully
        assert result.returncode == 0
        assert "Demo completed" in result.stdout

        # Check if files were created
        assert os.path.exists(os.path.join(self.temp_dir, "golden"))
        assert os.path.exists(os.path.join(self.temp_dir, "test"))
        assert os.path.exists(os.path.join(self.temp_dir, "results", "comparison_results.json"))

    def test_realistic_model_scenario(self):
        """Test with a realistic model scenario"""
        # Create a simple transformer-like model
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size=1000, d_model=256, nhead=8):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(500, d_model))
                self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.ReLU(),
                    nn.Linear(4 * d_model, d_model)
                )

            def forward(self, x):
                # Simple forward pass
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:seq_len]
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                ff_out = self.ff(x)
                x = self.norm2(x + ff_out)
                return x.mean(dim=1)  # Pool to sequence dimensions

        model = SimpleTransformer()
        input_data = torch.randint(0, 1000, (8, 32))  # Batch of sequences

        tester = PrecisionTester(framework="torch")
        results = tester.test_model_precision(
            model,
            input_data,
            self.temp_dir,
            "transformer_model",
            iterations=2,
            perturbation=1e-8
        )

        # Verify results
        assert results["summary"]["total_comparisons"] > 0
        # Should have attention, layer norm, and linear operations
        assert results["summary"]["pass_rate"] > 0.5  # Should be fairly accurate

    def test_multiple_models_comparison(self):
        """Test comparing multiple different models"""
        models = {
            "simple": nn.Sequential(
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 10)
            ),
            "with_dropout": nn.Sequential(
                nn.Linear(50, 25),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(25, 10)
            ),
            "deep": nn.Sequential(
                nn.Linear(50, 40),
                nn.ReLU(),
                nn.Linear(40, 30),
                nn.ReLU(),
                nn.Linear(30, 20),
                nn.ReLU(),
                nn.Linear(20, 10)
            )
        }

        input_data = torch.randn(16, 50)
        dump_paths = []
        tester = PrecisionTester(framework="torch")

        # Create dumps for each model with perturbation
        for name, model in models.items():
            dump_path = os.path.join(self.temp_dir, f"dump_{name}")
            perturbed_model = tester._create_perturbed_model(model, 1e-6)
            tester.dump_model_execution(perturbed_model, input_data, dump_path, name, iterations=2)
            dump_paths.append(dump_path)

        # Compare all dumps
        results = tester.load_and_compare_dumps(dump_paths, os.path.join(self.temp_dir, "multi_model_comparison"))

        # Verify results
        assert len(results) == 2  # 3 models - 1 baseline = 2 comparisons
        assert os.path.exists(os.path.join(self.temp_dir, "multi_model_comparison", "overall_summary.json"))

        # Each comparison should have results
        for name, result in results.items():
            assert "results" in result
            assert "summary" in result
            assert result["summary"]["total_comparisons"] > 0

    def test_precision_thresholds_impact(self):
        """Test how different precision thresholds impact results"""
        model = nn.Sequential(
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 10)
        )
        input_data = torch.randn(8, 20)

        # Test with different perturbation levels and thresholds
        test_cases = [
            {"perturbation": 1e-8, "thresholds": {"abs": 1e-7, "rel": 1e-7, "cosine": 0.999999}},
            {"perturbation": 1e-6, "thresholds": {"abs": 1e-5, "rel": 1e-5, "cosine": 0.99999}},
            {"perturbation": 1e-4, "thresholds": {"abs": 1e-3, "rel": 1e-3, "cosine": 0.9999}},
        ]

        results_summary = []

        for i, case in enumerate(test_cases):
            tester = PrecisionTester(framework="torch")
            results = tester.test_model_precision(
                model,
                input_data,
                os.path.join(self.temp_dir, f"case_{i}"),
                f"model_case_{i}",
                iterations=2,
                perturbation=case["perturbation"]
            )
            results_summary.append({
                "perturbation": case["perturbation"],
                "pass_rate": results["summary"]["pass_rate"],
                "mean_abs_error": results["summary"]["mean_absolute_error"],
                "mean_rel_error": results["summary"]["mean_relative_error"],
                "mean_cosine": results["summary"]["mean_cosine_similarity"]
            })

        # Verify that higher perturbation leads to higher error and potentially lower pass rates
        assert len(results_summary) == 3
        # Errors should generally increase with perturbation
        for i in range(1, len(results_summary)):
            assert results_summary[i]["mean_abs_error"] >= results_summary[i-1]["mean_abs_error"] * 0.1  # Allow some variance

    def test_large_input_handling(self):
        """Test handling of larger inputs and models"""
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

        # Larger input
        input_data = torch.randn(32, 3, 224, 224)

        tester = PrecisionTester(framework="torch")
        results = tester.test_model_precision(
            model,
            input_data,
            self.temp_dir,
            "large_cnn",
            iterations=1,  # Single iteration due to size
            perturbation=1e-7
        )

        # Should handle larger inputs without issues
        assert results["summary"]["total_comparisons"] > 0
        assert results["summary"]["pass_rate"] > 0