"""
Tests for the precision comparator functionality
"""

import pytest
import numpy as np
import tempfile
import os
import sys
import shutil
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from precision_comparator import PrecisionComparator, ComparisonConfig, ComparisonResult


class TestPrecisionComparator:
    """Test suite for PrecisionComparator"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.golden_dir = os.path.join(self.temp_dir, "golden")
        self.test_dir = os.path.join(self.temp_dir, "test")

        # Create test data
        self.create_test_dumps()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_dumps(self):
        """Create test dump data"""
        # Create golden data
        os.makedirs(self.golden_dir, exist_ok=True)
        self.create_hdf5_dump(self.golden_dir, "golden", np.float32)

        # Create test data with slight perturbation
        os.makedirs(self.test_dir, exist_ok=True)
        self.create_hdf5_dump(self.test_dir, "test", np.float32, perturbation=1e-6)
        
    def create_hdf5_dump(self, path: str, name: str, dtype, perturbation: float = 0):
        """Create HDF5 dump file with sample data"""
        import h5py

        # Create model info
        model_info = {
            "framework": "torch",
            "model_name": name,
            "model_type": "Sequential",
            "parameters": 1000,
            "layers": [
                {"name": "layer1", "type": "Linear", "parameters": 200},
                {"name": "layer2", "type": "ReLU", "parameters": 0}
            ]
        }

        with open(os.path.join(path, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        # Create operator traces
        with h5py.File(os.path.join(path, "operator_traces.h5"), "w") as f:
            for i in range(3):  # 3 traces
                group = f.create_group(f"trace_{i:06d}")
                group.attrs["iteration"] = i
                group.attrs["ops_idx"] = i
                group.attrs["module_name"] = ["Linear", "ReLU"][i % 2]
                group.attrs["operator_name"] = ["Linear", "ReLU"][i % 2]
                group.attrs["timestamp"] = 1234567890.0 + i

                # Create test arrays
                shapes = [[8, 10], [8, 20]]
                for j in range(2):
                    input_shape = shapes[j % len(shapes)]
                    output_shape = shapes[(j+1) % len(shapes)]

                    # Create arrays
                    input_data = np.random.randn(*input_shape).astype(dtype)
                    output_data = np.random.randn(*output_shape).astype(dtype)

                    if perturbation > 0:
                        input_data += np.random.randn(*input_shape).astype(dtype) * perturbation
                        output_data += np.random.randn(*output_shape).astype(dtype) * perturbation

                    group.create_dataset(f"input_{j}", data=input_data)
                    group.create_dataset(f"output_{j}", data=output_data)

                group.attrs["input_shapes"] = json.dumps([list(s) for s in [input_shape, output_shape]])
                group.attrs["output_shapes"] = json.dumps([list(s) for s in [output_shape, input_shape]])
                group.attrs["input_dtypes"] = json.dumps([str(input_data.dtype), str(output_data.dtype)])
                group.attrs["output_dtypes"] = json.dumps([str(output_data.dtype), str(input_data.dtype)])

    def test_tensor_comparison(self):
        """Test basic tensor comparison"""
        comparator = PrecisionComparator()

        # Identical tensors
        tensor1 = np.array([1.0, 2.0, 3.0])
        tensor2 = np.array([1.0, 2.0, 3.0])

        abs_err, rel_err, cos_sim = comparator.compare_tensors(tensor1, tensor2)
        assert abs_err == 0.0
        assert rel_err == 0.0
        assert cos_sim == 1.0

        # Different tensors
        tensor3 = np.array([1.0, 2.0, 4.0])
        abs_err, rel_err, cos_sim = comparator.compare_tensors(tensor1, tensor3)
        assert abs_err > 0.0
        assert rel_err > 0.0
        assert cos_sim < 1.0

    def test_precision_comparison(self):
        """Test full precision comparison"""
        config = ComparisonConfig(
            abs_error_threshold=1e-5,
            rel_error_threshold=1e-5,
            cosine_similarity_threshold=0.99999
        )
        comparator = PrecisionComparator(config)

        results, summary = comparator.compare_traces(
            self.golden_dir,
            self.test_dir,
            os.path.join(self.temp_dir, "results")
        )

        # Verify results structure
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(summary, dict)

        # Verify result structure
        result = results[0]
        assert isinstance(result, ComparisonResult)
        assert hasattr(result, 'operator_name')
        assert hasattr(result, 'module_name')
        assert hasattr(result, 'iteration')
        assert hasattr(result, 'absolute_error')
        assert hasattr(result, 'relative_error')
        assert hasattr(result, 'cosine_similarity')
        assert hasattr(result, 'passed')

        # Verify summary structure
        expected_keys = [
            "total_comparisons",
            "passed_comparisons",
            "failed_comparisons",
            "pass_rate",
            "max_absolute_error",
            "mean_absolute_error",
            "max_relative_error",
            "mean_relative_error",
            "min_cosine_similarity",
            "mean_cosine_similarity"
        ]
        for key in expected_keys:
            assert key in summary

    def test_comparison_different_thresholds(self):
        """Test comparison with different thresholds"""
        # High thresholds - should pass

        config = ComparisonConfig(
            abs_error_threshold=1e-3,
            rel_error_threshold=1e-3,
            cosine_similarity_threshold=0.99
        )
        comparator = PrecisionComparator(config)
        results, summary = comparator.compare_traces(self.golden_dir, self.golden_dir)
        assert summary["pass_rate"] > 0.0

        # Low thresholds - might fail
        config = ComparisonConfig(
            abs_error_threshold=1e-10,
            rel_error_threshold=1e-10,
            cosine_similarity_threshold=0.9999999
        )
        comparator = PrecisionComparator(config)
        results, summary = comparator.compare_traces(self.golden_dir, self.test_dir)
        # Might have lower pass rate due to strict thresholds

    def test_shape_mismatch_error(self):
        """Test error handling for shape mismatches"""
        comparator = PrecisionComparator()

        tensor1 = np.array([1.0, 2.0, 3.0])
        tensor2 = np.array([1.0, 2.0])  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            comparator.compare_tensors(tensor1, tensor2)

    def test_missing_file_error(self):
        """Test error handling for missing files"""
        comparator = PrecisionComparator()

        with pytest.raises(FileNotFoundError, match="Traces file not found"):
            comparator.compare_traces("/nonexistent/golden", "/nonexistent/test")

    def test_mismatched_traces(self):
        """Test error handling for mismatched number of traces"""
        # Create different number of traces
        golden_dir = os.path.join(self.temp_dir, "golden2")
        test_dir = os.path.join(self.temp_dir, "test2")

        os.makedirs(golden_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Golden with 2 traces
        self.create_hdf5_dump_with_n_traces(golden_dir, "golden", 2)
        # Test with 3 traces
        self.create_hdf5_dump_with_n_traces(test_dir, "test", 3)

        comparator = PrecisionComparator()

        with pytest.raises(ValueError, match="Number of traces mismatch"):
            comparator.compare_traces(golden_dir, test_dir)

    def create_hdf5_dump_with_n_traces(self, path: str, name: str, n_traces: int):
        """Helper to create dump with specific number of traces"""
        import h5py

        model_info = {
            "framework": "torch",
            "model_name": name,
            "model_type": "Sequential",
            "parameters": 1000,
            "layers": []
        }

        with open(os.path.join(path, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        with h5py.File(os.path.join(path, "operator_traces.h5"), "w") as f:
            for i in range(n_traces):
                group = f.create_group(f"trace_{i:06d}")
                group.attrs["iteration"] = i
                group.attrs["module_name"] = "Linear"
                group.attrs["ops_idx"] = f"layer{i}"
                group.attrs["operator_name"] = "Linear"
                group.attrs["timestamp"] = 1234567890.0 + i
                group.attrs["input_shapes"] = json.dumps([[8, 10]])
                group.attrs["output_shapes"] = json.dumps([[8, 20]])
                group.attrs["input_dtypes"] = json.dumps(["float32"])
                group.attrs["output_dtypes"] = json.dumps(["float32"])
                group.create_dataset("input_0", data=np.random.randn(8, 10))
                group.create_dataset("output_0", data=np.random.randn(8, 20))

    def test_results_saving(self):
        """Test that results are properly saved to files"""
        comparator = PrecisionComparator()
        results, summary = comparator.compare_traces(
            self.golden_dir,
            self.test_dir,
            # "/home/shenpeng/workspace/PrecisionProject/tests/temp_golden",
            # "/home/shenpeng/workspace/PrecisionProject/tests/temp",
            os.path.join(self.temp_dir, "results")
        )

        # Check that files were created
        assert os.path.exists(os.path.join(self.temp_dir, "results", "comparison_results.json"))
        assert os.path.exists(os.path.join(self.temp_dir, "results", "summary.txt"))

        # Check JSON content
        with open(os.path.join(self.temp_dir, "results", "comparison_results.json"), "r") as f:
            saved_data = json.load(f)

        assert "results" in saved_data
        assert "summary" in saved_data
        assert len(saved_data["results"]) == len(results)