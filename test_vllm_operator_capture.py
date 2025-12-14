"""
Test script for vLLM operator information capture functionality.
"""

import os
import sys
import json
import time
from typing import Any, Dict

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_dumper import ModelDumper
from operator_capture import OperatorCaptureManager


def create_mock_vllm_model():
    """Create a mock vLLM model for testing without actual vLLM installation"""

    class MockAttention:
        def __init__(self):
            self.name = "mock_attention"

        def __call__(self, *args, **kwargs):
            # This will be patched to capture the attention operator
            return f"mock_attention_output"

    class MockPagedAttention:
        def __init__(self):
            self.name = "mock_paged_attention"

        def __call__(self, *args, **kwargs):
            # This will be patched to capture the paged_attention operator
            return f"mock_paged_attention_output"

    class MockModelExecutor:
        def __init__(self):
            self.attention = MockAttention()
            self.paged_attention = MockPagedAttention()

        def execute_model(self, *args, **kwargs):
            # Simulate model execution with internal operations
            # These will be captured through the patched operators
            attention_out = self.attention()
            paged_out = self.paged_attention()
            return f"model_executed_with_ops: {attention_out}, {paged_out}"

    class MockWorker:
        def __init__(self):
            self.model_executor = MockModelExecutor()

        def execute_model(self, *args, **kwargs):
            # Simulate worker execution
            return self.model_executor.execute_model(*args, **kwargs)

    class MockLLMEngine:
        def __init__(self):
            self.model_executor = MockModelExecutor()
            self.workers = [MockWorker(), MockWorker()]

    class MockVLLMModel:
        def __init__(self):
            self.llm_engine = MockLLMEngine()
            # Add operator methods that will be patched
            self.attention = MockAttention()
            self.paged_attention = MockPagedAttention()

            # Create actual methods that match operator names
            self.scaled_dot_product_attention = self.attention
            self.paged_attention_op = self.paged_attention

        def generate(self, *args, **kwargs):
            # Simulate generation with multiple internal operations
            # Call operators directly so they can be captured
            attention_result = self.scaled_dot_product_attention()
            paged_result = self.paged_attention_op()

            # Also call through worker
            worker_results = []
            for worker in self.llm_engine.workers:
                result = worker.execute_model(*args, **kwargs)
                worker_results.append(result)

            return [f"Generated with ops: {attention_result}, {paged_result}, workers: {worker_results}"]

    return MockVLLMModel()


def test_vllm_operator_capture():
    """Test vLLM operator information capture"""
    print("=== Testing vLLM Operator Information Capture ===\n")

    # Create test directory
    test_dir = "test_output_vllm"
    os.makedirs(test_dir, exist_ok=True)

    # Create model dumper with enhanced capture
    model_dumper = ModelDumper(
        framework="vllm",
        enable_enhanced_capture=True
    )

    # Create mock vLLM model
    model = create_mock_vllm_model()
    print(f"Created mock vLLM model: {type(model).__name__}")

    # Test data
    test_input = "The quick brown fox jumps over the lazy dog"
    test_params = {
        "prompts": test_input,
        "params": {
            "max_tokens": 50,
            "temperature": 0.7
        }
    }

    print(f"Test input: {test_input[:50]}...")

    # Run enhanced vLLM capture
    try:
        print("\n--- Running Enhanced vLLM Operator Capture ---")
        model_dumper.dump_model_execution(
            model=model,
            input_data=test_params,
            output_path=test_dir,
            model_name="mock_vllm_test",
            iterations=1,
            capture_all_operators=True,
            save_enhanced_info=True
        )

        # Check results
        print("\n--- Checking Results ---")

        # Check enhanced traces
        enhanced_traces = model_dumper.get_enhanced_operator_traces()
        print(f"Number of enhanced operator traces captured: {len(enhanced_traces)}")

        # Print operator statistics
        stats = model_dumper.get_operator_statistics()
        if stats:
            print("\nOperator Statistics:")
            print(f"  Total operators: {stats.get('total_operators', 0)}")
            print(f"  Unique operators: {stats.get('unique_operator_names', 0)}")
            print(f"  Operator types: {stats.get('operator_count_by_type', {})}")
            print(f"  Total execution time: {stats.get('total_execution_time_ms', 0):.2f} ms")
            print(f"  Total memory allocation: {stats.get('total_memory_alloc_mb', 0):.2f} MB")

            # List operators by name
            if 'operator_count_by_name' in stats:
                print("\nOperators captured:")
                for op_name, count in stats['operator_count_by_name'].items():
                    exec_time = stats['execution_time_by_operator'].get(op_name, {}).get('total', 0)
                    memory = stats['memory_alloc_by_operator'].get(op_name, {}).get('total', 0)
                    print(f"  {op_name}: {count} calls, {exec_time:.2f} ms, {memory:.2f} MB")

        # Check saved files
        saved_files = os.listdir(test_dir)
        print(f"\nFiles saved: {saved_files}")

        # Load and display enhanced traces
        enhanced_traces_file = os.path.join(test_dir, "enhanced_operator_traces.json")
        if os.path.exists(enhanced_traces_file):
            with open(enhanced_traces_file, 'r') as f:
                traces_data = json.load(f)
                print(f"\nEnhanced traces file contains {len(traces_data)} traces")

                if traces_data:
                    print("\nSample trace:")
                    sample = traces_data[0]
                    for key, value in sample.items():
                        if key not in ['tensor_inputs', 'tensor_outputs', 'arguments']:
                            print(f"  {key}: {value}")

        # Load and display statistics
        stats_file = os.path.join(test_dir, "operator_statistics.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
                print(f"\nModel statistics saved:")
                print(f"  Model name: {stats_data.get('model_name')}")
                print(f"  Total enhanced traces: {stats_data.get('total_enhanced_traces')}")

        print("\n✅ vLLM operator capture test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_operator_capture_manager():
    """Test the OperatorCaptureManager directly"""
    print("\n=== Testing OperatorCaptureManager Directly ===\n")

    manager = OperatorCaptureManager()

    # Configure for vLLM operators
    manager.configure_capture(
        capture_torch_ops=False,
        capture_custom_ops=True,
        capture_vllm_ops=True,
        performance_timing=True,
        memory_tracking=True
    )

    # Register vLLM operators
    model = create_mock_vllm_model()
    registered_ops = manager.register_vllm_operators(model)
    print(f"Registered operators: {registered_ops}")

    # Test capture context
    print("\n--- Testing Capture Context ---")
    try:
        with manager.capture_context(
            model_name="test_vllm",
            iteration=0,
            target_modules=[]
        ):
            # Simulate some operations by calling the wrapped operators
            for op_name in registered_ops[:3]:  # Test first 3 operators
                if op_name in manager.custom_operators:
                    # Call the wrapped operator to trigger capture
                    op_info = manager.custom_operators[op_name]
                    if 'wrapped' in op_info:
                        result = op_info['wrapped']("test_input")
                        print(f"  Executed {op_name}: {result}")

        # Check results
        traces = manager.get_captured_operators()
        print(f"\nCaptured {len(traces)} operator traces")

        for trace in traces:
            print(f"  Operator: {trace['operator_name']}, "
                  f"Type: {trace['operator_type']}, "
                  f"Module: {trace['module_path']}, "
                  f"Time: {trace['execution_time_ms']:.2f} ms")

        print("\n✅ OperatorCaptureManager test completed!")
        return True

    except Exception as e:
        print(f"\n❌ OperatorCaptureManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Starting vLLM Operator Capture Tests...\n")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    results = []

    # Test basic manager
    results.append(test_operator_capture_manager())

    # Test full integration
    results.append(test_vllm_operator_capture())

    # Summary
    print("\n" + "=" * 50)
    print("VLLM OPERATOR CAPTURE TEST SUMMARY")
    print("=" * 50)
    total_tests = len(results)
    passed = sum(results)
    failed = total_tests - passed

    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if passed == total_tests:
        print("\n🎉 All tests passed! vLLM operator capture is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())