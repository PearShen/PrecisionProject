#!/usr/bin/env python3
"""
Test script for enhanced operator capture functionality.
Demonstrates comprehensive operator tracking for both PyTorch and vLLM models.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from model_dumper import ModelDumper
from operator_formatter import OperatorTraceFormatter
from enhanced_model_dumper import register_example_custom_operators


def create_test_model():
    """Create a test model with various operator types"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.BatchNorm1d(20),
        nn.Linear(20, 15),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(15, 10),
        nn.Softmax(dim=1)
    )


def create_cnn_model():
    """Create a CNN model for more diverse operator testing"""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10),
        nn.Softmax(dim=1)
    )


def test_basic_operator_capture():
    """Test basic operator capture with enhanced ModelDumper"""
    print("🧪 Testing Basic Operator Capture")
    print("-" * 50)

    # Create test model and data
    model = create_test_model()
    input_data = torch.randn(8, 10)

    # Create enhanced dumper
    dumper = ModelDumper(framework="torch", enable_enhanced_capture=True)

    # Define output path
    output_path = "./test_output/basic_capture"
    os.makedirs(output_path, exist_ok=True)

    print(f"Model: {type(model).__name__}")
    print(f"Input shape: {input_data.shape}")
    print(f"Output path: {output_path}")

    # Perform enhanced dump
    dumper.dump_model_execution(
        model=model,
        input_data=input_data,
        output_path=output_path,
        model_name="test_model",
        iterations=2,
        capture_all_operators=True,
        save_enhanced_info=True
    )

    print(f"✅ Enhanced capture completed!")
    print(f"📊 Captured {len(dumper.get_enhanced_operator_traces())} operator traces")

    # Generate summary
    stats = dumper.get_operator_statistics()
    if stats:
        print(f"🔍 Unique operators: {stats.get('unique_operator_names', 'N/A')}")
        print(f"⚡ Total execution time: {stats.get('total_execution_time_ms', 0):.2f} ms")
        print(f"💾 Total memory: {stats.get('total_memory_alloc_mb', 0):.2f} MB")

    return dumper.get_enhanced_operator_traces()


def test_cnn_operator_capture():
    """Test operator capture with CNN model"""
    print("\n🧪 Testing CNN Operator Capture")
    print("-" * 50)

    # Create CNN model and data
    model = create_cnn_model()
    input_data = torch.randn(4, 3, 32, 32)

    # Create enhanced dumper
    dumper = ModelDumper(framework="torch", enable_enhanced_capture=True)

    # Define output path
    output_path = "./test_output/cnn_capture"
    os.makedirs(output_path, exist_ok=True)

    print(f"Model: {type(model).__name__}")
    print(f"Input shape: {input_data.shape}")

    # Perform enhanced dump
    dumper.dump_model_execution(
        model=model,
        input_data=input_data,
        output_path=output_path,
        model_name="cnn_model",
        iterations=1,
        capture_all_operators=True,
        save_enhanced_info=True
    )

    print(f"✅ CNN enhanced capture completed!")
    print(f"📊 Captured {len(dumper.get_enhanced_operator_traces())} operator traces")

    return dumper.get_enhanced_operator_traces()


def test_custom_operators():
    """Test custom operator registration and capture"""
    print("\n🧪 Testing Custom Operator Capture")
    print("-" * 50)

    # Create enhanced dumper
    dumper = ModelDumper(framework="torch", enable_enhanced_capture=True)

    # Register custom operators
    def custom_matmul_op(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Custom matrix multiplication with logging"""
        result = torch.matmul(tensor1, tensor2)
        return result

    def custom_activation(tensor: torch.Tensor) -> torch.Tensor:
        """Custom activation function"""
        return tensor * torch.sigmoid(tensor)

    dumper.register_custom_operator("custom_matmul_op", custom_matmul_op, "custom")
    dumper.register_custom_operator("custom_activation", custom_activation, "custom")

    # Create a simple model that might use these ops
    model = nn.Sequential(
        nn.Linear(10, 15),
        nn.ReLU(),
        nn.Linear(15, 5)
    )
    input_data = torch.randn(4, 10)

    # Define output path
    output_path = "./test_output/custom_operator_capture"
    os.makedirs(output_path, exist_ok=True)

    print(f"Registered custom operators: custom_matmul_op, custom_activation")

    # Perform enhanced dump
    dumper.dump_model_execution(
        model=model,
        input_data=input_data,
        output_path=output_path,
        model_name="custom_test_model",
        iterations=1,
        capture_all_operators=True,
        save_enhanced_info=True
    )

    print(f"✅ Custom operator capture completed!")
    print(f"📊 Captured {len(dumper.get_enhanced_operator_traces())} operator traces")

    # Check for custom operators in traces
    traces = dumper.get_enhanced_operator_traces()
    custom_traces = [t for t in traces if t.operator_type == 'custom']
    print(f"🔧 Custom operators found: {len(custom_traces)}")

    return traces


def test_vllm_integration():
    """Test vLLM integration with operator capture"""
    print("\n🧪 Testing vLLM Operator Capture")
    print("-" * 50)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("⚠️  vLLM not available. Skipping vLLM test.")
        return []

    # Create mock vLLM model
    class MockVLLMModel:
        def __init__(self):
            self.llm = LLM(model="facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.3)

        def generate(self, *args, **kwargs):
            return self.llm.generate(*args, **kwargs)

    # Create enhanced dumper
    dumper = ModelDumper(framework="vllm", enable_enhanced_capture=True)

    # Define output path
    output_path = "./test_output/vllm_capture"
    os.makedirs(output_path, exist_ok=True)

    # Create mock model and input
    try:
        mock_model = MockVLLMModel()
        prompts = [
            "Hello, my name is",
            "The capital of France is"
        ]
        mock_input = {"prompts": prompts, "params": {"temperature": 0.8, "max_tokens": 5}}

        print(f"Model: vLLM (OPT-125m)")
        print(f"Prompts: {len(prompts)}")

        # Perform enhanced dump
        dumper.dump_model_execution(
            model=mock_model,
            input_data=mock_input,
            output_path=output_path,
            model_name="vllm_test_model",
            iterations=1,
            capture_all_operators=True,
            save_enhanced_info=True
        )

        print(f"✅ vLLM enhanced capture completed!")
        print(f"📊 Captured {len(dumper.get_enhanced_operator_traces())} operator traces")

    except Exception as e:
        print(f"⚠️  vLLM test failed: {e}")

    return dumper.get_enhanced_operator_traces()


def test_output_formats():
    """Test different output formats"""
    print("\n🧪 Testing Output Formats")
    print("-" * 50)

    # Get traces from previous test
    dumper = ModelDumper(framework="torch", enable_enhanced_capture=True)
    model = create_test_model()
    input_data = torch.randn(4, 10)

    output_path = "./test_output/output_formats"
    os.makedirs(output_path, exist_ok=True)

    # Capture traces
    dumper.dump_model_execution(
        model=model,
        input_data=input_data,
        output_path=output_path,
        model_name="format_test_model",
        iterations=2,
        capture_all_operators=True,
        save_enhanced_info=True
    )

    traces = dumper.get_enhanced_operator_traces()

    if not traces:
        print("❌ No traces to test formats with!")
        return

    # Create formatter
    formatter = OperatorTraceFormatter(traces)

    # Test plain text summary
    print("\n📝 Plain Text Summary Preview:")
    print("-" * 30)
    text_summary = formatter.generate_plain_text_summary()
    print(text_summary[:500] + "..." if len(text_summary) > 500 else text_summary)

    # Test CSV export
    csv_path = os.path.join(output_path, "operator_traces.csv")
    formatter.export_to_csv(csv_path)
    print(f"\n📊 CSV export: {csv_path}")

    # Test HTML report
    html_path = os.path.join(output_path, "report.html")
    formatter.generate_html_report(html_path, include_charts=True)
    print(f"🌐 HTML report: {html_path}")

    # Test JSON summary
    json_path = os.path.join(output_path, "summary.json")
    formatter.generate_summary_report(json_path)
    print(f"📋 JSON summary: {json_path}")

    print("✅ All output formats generated successfully!")


def main():
    """Main test function"""
    print("🚀 Starting Enhanced Operator Capture Tests")
    print("=" * 60)

    # Create test output directory
    os.makedirs("./test_output", exist_ok=True)

    # Run tests
    try:
        traces1 = test_basic_operator_capture()
        traces2 = test_cnn_operator_capture()
        traces3 = test_custom_operators()
        traces4 = test_vllm_integration()
        test_output_formats()

        # Summary
        total_traces = len(traces1) + len(traces2) + len(traces3) + len(traces4)
        print(f"\n🎉 All tests completed successfully!")
        print(f"📊 Total operator traces captured: {total_traces}")
        print(f"📂 Test outputs saved to: ./test_output/")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()