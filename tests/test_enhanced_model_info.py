"""
Tests for enhanced model_info.json with multi-dimensional information
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_dumper import ModelDumper


class TestEnhancedModelInfo:
    """Test suite for enhanced model_info.json functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_enhanced_linear_model_info(self):
        """Test enhanced model info for linear models"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.Softmax(dim=1)
        )
        input_data = torch.randn(8, 10)

        dumper = ModelDumper(framework="torch")
        dumper.dump_model_execution(
            model, input_data, self.temp_dir, "linear_test_model", iterations=1
        )

        # Load and verify enhanced model info
        with open(os.path.join(self.temp_dir, "model_info.json"), "r") as f:
            model_info = json.load(f)

        # Verify basic info
        assert model_info["framework"] == "torch"
        assert model_info["model_name"] == "linear_test_model"
        assert model_info["model_type"] == "Sequential"

        # Verify enhanced input/output info
        assert "input_info" in model_info
        assert model_info["input_info"]["shape"] == [8, 10]
        assert "torch.float32" in model_info["input_info"]["dtype"]

        assert "output_info" in model_info
        assert model_info["output_info"]["shape"] == [8, 5]
        assert "torch.float32" in model_info["output_info"]["dtype"]

        # Verify enhanced layer info
        layers = model_info["layers"]
        assert len(layers) == 4

        # Check first layer (Linear)
        layer0 = layers[0]
        assert layer0["layer_index"] == 0
        assert layer0["name"] == "0"
        assert layer0["type"] == "Linear"
        assert layer0["parameters"] == 220  # 10*20 + 20
        assert layer0["trainable"] is True
        assert layer0["in_features"] == 10
        assert layer0["out_features"] == 20
        assert layer0["bias"] is True
        assert layer0["parameter_shapes"] == [[20, 10], [20]]
        assert all("float" in dtype for dtype in layer0["parameter_dtypes"])

        # Check second layer (ReLU)
        layer1 = layers[1]
        assert layer1["layer_index"] == 1
        assert layer1["name"] == "1"
        assert layer1["type"] == "ReLU"
        assert layer1["parameters"] == 0
        assert layer1["trainable"] is False
        assert layer1["inplace"] is False

    def test_enhanced_cnn_model_info(self):
        """Test enhanced model info for CNN models"""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        input_data = torch.randn(4, 3, 32, 32)

        dumper = ModelDumper(framework="torch")
        dumper.dump_model_execution(
            model, input_data, self.temp_dir, "cnn_test_model", iterations=1
        )

        # Load and verify enhanced model info
        with open(os.path.join(self.temp_dir, "model_info.json"), "r") as f:
            model_info = json.load(f)

        # Verify input/output info for CNN
        assert model_info["input_info"]["shape"] == [4, 3, 32, 32]
        assert model_info["output_info"]["shape"] == [4, 10]

        # Check Conv2d layer info
        conv_layer = next(layer for layer in model_info["layers"] if layer["type"] == "Conv2d")
        assert conv_layer["in_channels"] == 3
        assert conv_layer["out_channels"] == 16
        assert conv_layer["kernel_size"] == [3, 3]
        assert conv_layer["stride"] == [1, 1]
        assert conv_layer["padding"] == [1, 1]
        assert conv_layer["bias"] is True

        # Check BatchNorm2d layer info
        bn_layer = next(layer for layer in model_info["layers"] if layer["type"] == "BatchNorm2d")
        assert bn_layer["num_features"] == 16
        assert bn_layer["eps"] == 1e-05
        assert bn_layer["affine"] is True
        assert bn_layer["track_running_stats"] is True

        # Check AdaptiveAvgPool2d layer info
        pool_layer = next(layer for layer in model_info["layers"] if layer["type"] == "AdaptiveAvgPool2d")
        assert "output_size" in pool_layer  # This would require custom handling

        # Check Flatten layer info
        flatten_layer = next(layer for layer in model_info["layers"] if layer["type"] == "Flatten")
        assert flatten_layer["parameters"] == 0

    def test_enhanced_model_with_dropout_info(self):
        """Test enhanced model info with dropout layers"""
        model = nn.Sequential(
            nn.Linear(50, 100),
            nn.Dropout(p=0.3, inplace=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 10)
        )
        input_data = torch.randn(16, 50)

        dumper = ModelDumper(framework="torch")
        dumper.dump_model_execution(
            model, input_data, self.temp_dir, "dropout_test_model", iterations=1
        )

        # Load and verify enhanced model info
        with open(os.path.join(self.temp_dir, "model_info.json"), "r") as f:
            model_info = json.load(f)

        # Check dropout layers
        dropout_layers = [layer for layer in model_info["layers"] if layer["type"] == "Dropout"]
        assert len(dropout_layers) == 2

        # First dropout layer (p=0.3, inplace=True)
        dropout1 = dropout_layers[0]
        assert dropout1["p"] == 0.3
        assert dropout1["inplace"] is True

        # Second dropout layer (p=0.5, inplace=False)
        dropout2 = dropout_layers[1]
        assert dropout2["p"] == 0.5
        assert dropout2["inplace"] is False

    def test_sequential_model_info(self):
        """Test enhanced model info for nested Sequential models"""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 15),
                nn.ReLU(),
                nn.Linear(15, 20)
            ),
            nn.Sequential(
                nn.Linear(20, 25),
                nn.ReLU()
            ),
            nn.Linear(25, 5)
        )
        input_data = torch.randn(8, 10)

        dumper = ModelDumper(framework="torch")
        dumper.dump_model_execution(
            model, input_data, self.temp_dir, "nested_sequential_model", iterations=1
        )

        # Load and verify enhanced model info
        with open(os.path.join(self.temp_dir, "model_info.json"), "r") as f:
            model_info = json.load(f)

        # Check Sequential layers have num_layers info
        sequential_layers = [layer for layer in model_info["layers"] if layer["type"] == "Sequential"]
        assert len(sequential_layers) == 2

        for seq_layer in sequential_layers:
            assert "num_layers" in seq_layer
            assert seq_layer["num_layers"] >= 2

    def test_vllm_enhanced_model_info(self):
        """Test enhanced model info for vLLM models (placeholder)"""
        class MockVLLMModel:
            def generate(self, *args, **kwargs):
                return ["Generated text response"]

        mock_model = MockVLLMModel()
        mock_input = {"prompts": "Test prompt", "params": {"temperature": 0.8, "max_tokens": 100}}

        dumper = ModelDumper(framework="vllm")

        # This should work with the placeholder implementation
        try:
            dumper.dump_model_execution(
                mock_model, mock_input, self.temp_dir, "vllm_test_model", iterations=1
            )

            # Load and verify enhanced vLLM model info
            with open(os.path.join(self.temp_dir, "model_info.json"), "r") as f:
                model_info = json.load(f)

            # Verify vLLM specific info
            assert model_info["framework"] == "vllm"
            assert "input_info" in model_info
            assert "output_info" in model_info

            # Check enhanced input info
            assert model_info["input_info"]["type"] == "text_or_sampling_params"
            assert "sample_input" in model_info["input_info"]

            # Check enhanced output info
            assert model_info["output_info"]["type"] == "generated_text"
            assert "sample_output" in model_info["output_info"]
            assert model_info["output_info"]["num_outputs"] >= 1

            # Check enhanced layer info
            layers = model_info["layers"]
            assert len(layers) == 1
            assert layers[0]["layer_index"] == 0
            assert layers[0]["type"] == "vLLM"
            assert "description" in layers[0]
            assert "components" in layers[0]

            # Check components
            components = layers[0]["components"]
            component_names = [comp["name"] for comp in components]
            assert "tokenizer" in component_names
            assert "model_engine" in component_names
            assert "scheduler" in component_names

        except ImportError:
            # vLLM not installed, skip test
            pytest.skip("vLLM not installed")

    def test_model_info_completeness(self):
        """Test that all required fields are present in model_info"""
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 1)
        )
        input_data = torch.randn(8, 5)

        dumper = ModelDumper(framework="torch")
        dumper.dump_model_execution(
            model, input_data, self.temp_dir, "completeness_test_model", iterations=1
        )

        # Load model info
        with open(os.path.join(self.temp_dir, "model_info.json"), "r") as f:
            model_info = json.load(f)

        # Check required top-level fields
        required_fields = [
            "framework", "model_name", "model_type", "parameters",
            "input_info", "output_info", "layers"
        ]
        for field in required_fields:
            assert field in model_info, f"Missing field: {field}"

        # Check input_info fields
        input_fields = ["shape", "dtype"]
        for field in input_fields:
            assert field in model_info["input_info"], f"Missing input field: {field}"

        # Check output_info fields
        output_fields = ["shape", "dtype"]
        for field in output_fields:
            assert field in model_info["output_info"], f"Missing output field: {field}"

        # Check layer fields
        for layer in model_info["layers"]:
            required_layer_fields = [
                "layer_index", "name", "type", "parameters",
                "trainable", "parameter_shapes", "parameter_dtypes"
            ]
            for field in required_layer_fields:
                assert field in layer, f"Missing layer field: {field} in {layer['name']}"