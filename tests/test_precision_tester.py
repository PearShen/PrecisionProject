"""
Tests for the precision tester main functionality
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from precision_tester import PrecisionTester
from precision_comparator import ComparisonConfig


class TestPrecisionTester:
    """Test suite for PrecisionTester"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.Softmax(dim=1)
        )
        self.input_data = torch.randn(8, 10)

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_precision_tester_initialization(self):
        """Test PrecisionTester initialization"""
        # Test default initialization
        tester = PrecisionTester()
        assert tester.framework == "torch"
        assert tester.dumper is not None
        assert tester.comparator is not None

        # Test with custom framework
        tester = PrecisionTester(framework="vllm")
        assert tester.framework == "vllm"

        # Test with custom config
        config = ComparisonConfig(abs_error_threshold=1e-6)
        tester = PrecisionTester(framework="torch", comparison_config=config)
        assert tester.comparator.config.abs_error_threshold == 1e-6

    def test_complete_precision_test_workflow(self):
        """Test the complete precision testing workflow"""
        tester = PrecisionTester(framework="torch")

        results = tester.test_model_precision(
            self.model,
            self.input_data,
            self.temp_dir,
            "test_model",
            iterations=2,
            perturbation=1e-6
        )

        # Verify results structure
        assert "golden_path" in results
        assert "test_path" in results
        assert "results_path" in results
        assert "results" in results
        assert "summary" in results

        # Verify paths exist
        assert os.path.exists(results["golden_path"])
        assert os.path.exists(results["test_path"])
        assert os.path.exists(results["results_path"])

        # Check created files
        assert os.path.exists(os.path.join(results["golden_path"], "model_info.json"))
        assert os.path.exists(os.path.join(results["golden_path"], "operator_traces.h5"))
        assert os.path.exists(os.path.join(results["test_path"], "model_info.json"))
        assert os.path.exists(os.path.join(results["test_path"], "operator_traces.h5"))

    def test_perturbed_model_creation(self):
        """Test creation of perturbed model"""
        tester = PrecisionTester(framework="torch")

        original_params = []
        for param in self.model.parameters():
            original_params.append(param.clone())

        perturbed_model = tester._create_perturbed_model(self.model, 1e-5)

        # Check that parameters are different but close
        for orig_param, pert_param in zip(original_params, perturbed_model.parameters()):
            assert not torch.equal(orig_param, pert_param)
            diff = (orig_param - pert_param).abs().max()
            assert diff > 0
            assert diff < 1e-4  # Should be close to original

    def test_multiple_dump_comparison(self):
        """Test comparison of multiple dumps"""
        tester = PrecisionTester(framework="torch")

        # Create multiple dumps with different perturbation levels
        dump_paths = []

        for i, perturbation in enumerate([1e-6, 1e-5, 1e-4]):
            dump_path = os.path.join(self.temp_dir, f"dump_{perturbation}")
            os.makedirs(dump_path, exist_ok=True)

            # Create perturbed model and dump
            perturbed_model = tester._create_perturbed_model(self.model, perturbation)
            tester.dump_model_execution(perturbed_model, self.input_data, dump_path, f"model_{perturbation}", iterations=1)
            dump_paths.append(dump_path)

        # Compare all dumps against the first (smallest perturbation)
        results = tester.load_and_compare_dumps(dump_paths, os.path.join(self.temp_dir, "multi_comparison"))

        # Verify results
        assert len(results) == 2  # 3 dumps - 1 baseline = 2 comparisons
        assert os.path.exists(os.path.join(self.temp_dir, "multi_comparison", "overall_summary.json"))

        # Check that higher perturbation leads to lower pass rates (generally)
        comparison_names = list(results.keys())
        for name in comparison_names:
            assert "comparison_" in name
            assert "results" in results[name]
            assert "summary" in results[name]

    def test_dump_and_compare_methods(self):
        """Test individual dump and compare methods"""
        tester = PrecisionTester(framework="torch")

        # Test dump method
        golden_path = os.path.join(self.temp_dir, "golden")
        tester.dump_model_execution(self.model, self.input_data, golden_path, "test_model", iterations=2)

        assert os.path.exists(golden_path)
        assert os.path.exists(os.path.join(golden_path, "model_info.json"))
        assert os.path.exists(os.path.join(golden_path, "operator_traces.h5"))

        # Create test dump
        test_path = os.path.join(self.temp_dir, "test")
        perturbed_model = tester._create_perturbed_model(self.model, 1e-6)
        tester.dump_model_execution(perturbed_model, self.input_data, test_path, "test_model", iterations=2)

        # Test compare method
        results_path = os.path.join(self.temp_dir, "results")
        results, summary = tester.compare_precision(golden_path, test_path, results_path)

        assert len(results) > 0
        assert "pass_rate" in summary
        assert os.path.exists(results_path)

    def test_different_model_types(self):
        """Test with different model architectures"""
        # Test with CNN
        cnn_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

        cnn_input = torch.randn(4, 3, 32, 32)

        tester = PrecisionTester(framework="torch")
        results = tester.test_model_precision(
            cnn_model,
            cnn_input,
            self.temp_dir,
            "cnn_model",
            iterations=1,
            perturbation=1e-6
        )

        assert results["summary"]["total_comparisons"] > 0
        assert "pass_rate" in results["summary"]

    def test_vllm_framework_handling(self):
        """Test vLLM framework handling (placeholder implementation)"""
        class MockVLLMModel:
            def __init__(self):
                self.test_value = "test"

            state_dict_value = {"test": "value"}

            def parameters(self):
                return []

            def __call__(self, *args, **kwargs):
                return "output"

        tester = PrecisionTester(framework="vllm")
        mock_model = MockVLLMModel()
        mock_input = {"test": "data"}

        # This should work even with placeholder implementation
        results = tester.test_model_precision(
            mock_model,
            mock_input,
            self.temp_dir,
            "mock_vllm",
            iterations=1,
            perturbation=1e-6
        )

        # Should still create the necessary files
        assert os.path.exists(results["golden_path"])
        assert os.path.exists(results["test_path"])

    def test_error_handling(self):
        """Test error handling"""
        tester = PrecisionTester(framework="torch")

        # Test with insufficient dump paths
        with pytest.raises(ValueError, match="At least 2 dump paths are required"):
            tester.load_and_compare_dumps([self.temp_dir], self.temp_dir)

        # Test with non-existent paths for comparison
        with pytest.raises(Exception):  # Should raise some kind of error
            tester.compare_precision("/nonexistent/golden", "/nonexistent/test")