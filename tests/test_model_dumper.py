"""
Tests for the model dumper functionality
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add src to path

from PrecisionProject.model_dumper import ModelDumper


class TestModelDumper:
    """Test suite for ModelDumper"""

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
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # def test_torch_model_dump(self):
    #     """Test PyTorch model dumping"""
    #     # import pdb
    #     # pdb.set_trace()
    #     dumper = ModelDumper(framework="torch")

    #     # Dump model
    #     dumper.dump_model_execution(
    #         self.model,
    #         self.input_data,
    #         self.temp_dir,
    #         "test_model",
    #         iterations=2
    #     )
    #     # Verify files exist
    #     assert os.path.exists(os.path.join(self.temp_dir, "model_info.json"))
    #     assert os.path.exists(os.path.join(self.temp_dir, "operator_traces.h5"))

    #     # Verify model info
    #     import json
    #     with open(os.path.join(self.temp_dir, "model_info.json"), "r") as f:
    #         model_info = json.load(f)

    #     assert model_info["framework"] == "torch"
    #     assert model_info["model_name"] == "test_model"
    #     assert model_info["model_type"] == "Sequential"
    #     assert "layers" in model_info

    # def test_operator_trace_format(self):
    #     """Test operator trace format"""
    #     dumper = ModelDumper(framework="torch")

    #     dumper.dump_model_execution(
    #         self.model,
    #         self.input_data,
    #         self.temp_dir,
    #         "test_model",
    #         iterations=1
    #     )

    #     # Load traces
    #     traces = dumper.load_traces(os.path.join(self.temp_dir, "operator_traces.h5"))

    #     # Verify trace fields
    #     assert len(traces) > 0

    #     trace = traces[0]
    #     assert trace.iteration == 0
    #     assert trace.model_name == "test_model"
    #     assert hasattr(trace, 'layer_name')
    #     assert hasattr(trace, 'operator_name')
    #     assert hasattr(trace, 'input_shapes')
    #     assert hasattr(trace, 'output_shapes')
    #     assert hasattr(trace, 'input_dtypes')
    #     assert hasattr(trace, 'output_dtypes')
    #     assert hasattr(trace, 'inputs')
    #     assert hasattr(trace, 'outputs')
    #     assert hasattr(trace, 'timestamp')

    # def test_multiple_iterations(self):
    #     """Test multiple forward pass iterations"""
    #     dumper = ModelDumper(framework="torch")

    #     dumper.dump_model_execution(
    #         self.model,
    #         self.input_data,
    #         self.temp_dir,
    #         "test_model",
    #         iterations=3
    #     )

    #     # Load traces
    #     traces = dumper.load_traces(os.path.join(self.temp_dir, "operator_traces.h5"))

    #     # Group by iteration
    #     iterations = set(trace.iteration for trace in traces)
    #     assert len(iterations) == 3
    #     assert iterations == {0, 1, 2}

    # def test_different_model_types(self):
    #     """Test with different model architectures"""
    #     # Test with CNN
    #     cnn_model = nn.Sequential(
    #         nn.Conv2d(3, 16, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.AdaptiveAvgPool2d((1, 1)),
    #         nn.Flatten(),
    #         nn.Linear(16, 10)
    #     )

    #     cnn_input = torch.randn(4, 3, 32, 32)

    #     dumper = ModelDumper(framework="torch")
    #     dumper.dump_model_execution(
    #         cnn_model,
    #         cnn_input,
    #         self.temp_dir,
    #         "cnn_model",
    #         iterations=1
    #     )

    #     # Verify traces were created
    #     traces = dumper.load_traces(os.path.join(self.temp_dir, "operator_traces.h5"))
    #     assert len(traces) > 0

    #     # Check for Conv2d operation
    #     conv_traces = [t for t in traces if t.operator_name == "Conv2d"]
    #     assert len(conv_traces) > 0

    # def test_model_structure_extraction(self):
    #     """Test model structure extraction"""
    #     dumper = ModelDumper(framework="torch")
    #     structure = dumper._get_torch_model_structure(self.model)

    #     assert isinstance(structure, list)
    #     assert len(structure) > 0

    #     # Check structure format
    #     for layer in structure:
    #         assert "name" in layer
    #         assert "type" in layer
    #         assert "parameters" in layer

    # def test_invalid_framework(self):
    #     """Test invalid framework selection"""
    #     dumper = ModelDumper(framework="invalid")

    #     with pytest.raises(ValueError, match="Unsupported framework"):
    #         dumper.dump_model_execution(
    #             self.model,
    #             self.input_data,
    #             self.temp_dir,
    #             "test_model",
    #             iterations=1
    #         )

    def test_vllm_framework_placeholder(self):
        """Test vLLM framework (placeholder implementation)"""
        # Create a mock vLLM model object
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]#*100
        from vllm import LLM, SamplingParams 
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        model_name = "facebook/opt-125m"
        # model_name = "nickypro/tinyllama-42M-fp32"
        llm = LLM(model=model_name,enforce_eager=True, gpu_memory_utilization=0.03, dtype=torch.float16)
        mock_input = {"prompts": prompts, "params": {"temperature": 0.0, "max_tokens": 3}}
        
        # mock_model = MockVLLMModel(model_name)
        

        dumper = ModelDumper(framework="vllm")
        self.dump_dir = f"/home/shenpeng/workspace/PrecisionProject/tests/temp_{model_name.replace('/','_')}/"
        
        # This should work with the placeholder implementation
        dumper.dump_model_execution(
            llm,
            mock_input,
            self.dump_dir,
            model_name,
            iterations=1
        )
        
        # Verify files were created
        assert os.path.exists(os.path.join(self.dump_dir, f"model_info.json"))
        assert os.path.exists(os.path.join(self.dump_dir, f"operator_traces.h5"))
T = TestModelDumper()
T.setup_method()
T.test_vllm_framework_placeholder()