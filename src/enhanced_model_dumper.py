"""
Enhanced model dumper with comprehensive operator capture capabilities.
Extends the original ModelDumper to support detailed operator tracking.
"""

import torch
import torch.nn as nn
import h5py
import json
import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from model_dumper import ModelDumper, OperatorInfo
from operator_capture import OperatorCaptureManager, EnhancedOperatorInfo


class EnhancedModelDumper(ModelDumper):
    """
    Enhanced model dumper with comprehensive operator tracking.

    This class extends ModelDumper to include:
    - All torch operator tracking
    - Custom operator registration
    - Performance timing
    - Memory usage tracking
    - Detailed execution context
    """

    def __init__(self, framework: str = "torch"):
        super().__init__(framework)
        self.operator_capture_manager = OperatorCaptureManager()
        self.enhanced_traces: List[EnhancedOperatorInfo] = []

    def configure_operator_capture(self,
                                 capture_torch_ops: bool = True,
                                 capture_custom_ops: bool = True,
                                 include_autograd: bool = False,
                                 performance_timing: bool = True,
                                 memory_tracking: bool = True,
                                 custom_operators: Optional[Dict[str, Callable]] = None,
                                 filter_operators: Optional[List[str]] = None):
        """
        Configure comprehensive operator capture settings.

        Args:
            capture_torch_ops: Whether to capture PyTorch native operators
            capture_custom_ops: Whether to capture custom operators
            include_autograd: Whether to include autograd operations
            performance_timing: Whether to track execution time
            memory_tracking: Whether to track memory allocation
            custom_operators: Dictionary of custom operators to register
            filter_operators: List of specific operators to capture (None for default set)
        """
        self.operator_capture_manager.configure_capture(
            capture_torch_ops=capture_torch_ops,
            capture_custom_ops=capture_custom_ops,
            include_autograd=include_autograd,
            performance_timing=performance_timing,
            memory_tracking=memory_tracking,
            filter_operators=filter_operators
        )

        # Register custom operators
        if custom_operators:
            for name, func in custom_operators.items():
                self.register_custom_operator(name, func)

    def register_custom_operator(self,
                                name: str,
                                func: Callable,
                                operator_type: str = 'custom',
                                module_path: str = None):
        """Register a custom operator for capture"""
        self.operator_capture_manager.register_custom_operator(name, func, operator_type, module_path)

    def dump_model_execution_with_operators(self,
                                          model: Any,
                                          input_data: Any,
                                          output_path: str,
                                          model_name: str = "model",
                                          iterations: int = 1,
                                          save_enhanced_info: bool = True,
                                          save_operator_stats: bool = True) -> None:
        """
        Dump model execution with comprehensive operator tracking.

        This method performs both the original model dumping and comprehensive operator capture.

        Args:
            model: The model to dump
            input_data: Input data for the model
            output_path: Path to save the dump
            model_name: Name of the model
            iterations: Number of forward passes to record
            save_enhanced_info: Whether to save enhanced operator information
            save_operator_stats: Whether to save operator statistics
        """
        if self.framework == "torch":
            self._dump_torch_model_with_operators(
                model, input_data, output_path, model_name, iterations,
                save_enhanced_info, save_operator_stats
            )
        elif self.framework == "vllm":
            self._dump_vllm_model_with_operators(
                model, input_data, output_path, model_name, iterations,
                save_enhanced_info, save_operator_stats
            )
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def _dump_torch_model_with_operators(self,
                                       model: nn.Module,
                                       input_data: torch.Tensor,
                                       output_path: str,
                                       model_name: str,
                                       iterations: int,
                                       save_enhanced_info: bool,
                                       save_operator_stats: bool) -> None:
        """Dump PyTorch model with comprehensive operator tracking"""
        os.makedirs(output_path, exist_ok=True)

        # First, perform the original model dump
        self._dump_torch_model(model, input_data, output_path, model_name, iterations)

        # Then perform enhanced operator capture
        target_modules = [name for name, _ in model.named_modules() if name]

        for iteration in range(iterations):
            with self.operator_capture_manager.capture_context(
                model_name=model_name,
                iteration=iteration,
                target_modules=target_modules
            ):
                # Run forward pass with operator capture
                with torch.no_grad():
                    _ = model(input_data)

        # Collect enhanced traces
        self.enhanced_traces = self.operator_capture_manager.captured_operators

        # Save enhanced information
        if save_enhanced_info:
            self._save_enhanced_operator_traces(output_path, model_name)

        # Save operator statistics
        if save_operator_stats:
            self._save_operator_statistics(output_path, model_name)

        # Save legacy compatibility traces
        self._enhanced_to_legacy_traces()
        self._save_traces_hdf5(output_path)

    def _dump_vllm_model_with_operators(self,
                                       model: Any,
                                       input_data: Any,
                                       output_path: str,
                                       model_name: str,
                                       iterations: int,
                                       save_enhanced_info: bool,
                                       save_operator_stats: bool) -> None:
        """Dump vLLM model with comprehensive operator tracking"""
        os.makedirs(output_path, exist_ok=True)

        # First, perform original vLLM dump
        self._dump_vllm_model(model, input_data, output_path, model_name, iterations)

        # For vLLM, we need to patch the generate method for operator tracking
        original_generate = None

        if hasattr(model, 'generate'):
            original_generate = model.generate

            def enhanced_generate(*args, **kwargs):
                with self.operator_capture_manager.capture_context(
                    model_name=model_name,
                    iteration=0,  # vLLM typically runs once
                    target_modules=[]
                ):
                    return original_generate(*args, **kwargs)

            # Temporarily replace generate method
            model.generate = enhanced_generate

        try:
            # Execute VLLM for operator capture
            if isinstance(input_data, str):
                outputs = model.generate(input_data)
            else:
                class DummySamplingParams:
                    def __init__(self, *args, **kwargs):
                        pass
                SamplingParams = DummySamplingParams
                try:
                    from vllm import SamplingParams
                except ImportError:
                    pass

                sampling_params = SamplingParams(**input_data.get("params", {}))
                outputs = model.generate(input_data["prompts"], sampling_params)

            # Collect enhanced traces
            self.enhanced_traces = self.operator_capture_manager.captured_operators

            # Save enhanced information
            if save_enhanced_info:
                self._save_enhanced_operator_traces(output_path, model_name)

            # Save operator statistics
            if save_operator_stats:
                self._save_operator_statistics(output_path, model_name)

            # Save legacy compatibility traces
            self._enhanced_to_legacy_traces()
            self._save_traces_hdf5(output_path)

        finally:
            # Restore original generate method
            if original_generate and hasattr(model, 'generate'):
                model.generate = original_generate

    def _save_enhanced_operator_traces(self, output_path: str, model_name: str) -> None:
        """Save enhanced operator traces to JSON"""
        enhanced_traces_path = os.path.join(output_path, "enhanced_operator_traces.json")

        # Convert enhanced traces to dictionaries for JSON serialization
        enhanced_traces_dicts = []
        for trace in self.enhanced_traces:
            trace_dict = {
                'iteration': trace.iteration,
                'model_name': trace.model_name,
                'operator_name': trace.operator_name,
                'operator_type': trace.operator_type,
                'module_path': trace.module_path,
                'layer_name': trace.layer_name,
                'call_site': trace.call_site,
                'input_shapes': trace.input_shapes,
                'output_shapes': trace.output_shapes,
                'input_dtypes': trace.input_dtypes,
                'output_dtypes': trace.output_dtypes,
                'execution_time_ms': trace.execution_time_ms,
                'memory_alloc_mb': trace.memory_alloc_mb,
                'arguments': trace.arguments,
                'tensor_input_shapes': [list(t.shape) for t in trace.tensor_inputs],
                'tensor_input_dtypes': [str(t.dtype) for t in trace.tensor_inputs],
                'tensor_output_shapes': [list(t.shape) for t in trace.tensor_outputs],
                'tensor_output_dtypes': [str(t.dtype) for t in trace.tensor_outputs],
                'timestamp': trace.timestamp,
                'thread_id': trace.thread_id,
                'context_info': trace.context_info
            }
            enhanced_traces_dicts.append(trace_dict)

        with open(enhanced_traces_path, 'w') as f:
            json.dump(enhanced_traces_dicts, f, indent=2, default=str)

    def _save_operator_statistics(self, output_path: str, model_name: str) -> None:
        """Save operator statistics to JSON"""
        stats_path = os.path.join(output_path, "operator_statistics.json")

        if self.enhanced_traces:
            stats = self.operator_capture_manager.get_operator_statistics()

            # Add additional information
            stats['model_name'] = model_name
            stats['capture_timestamp'] = time.time()
            stats['total_enhanced_traces'] = len(self.enhanced_traces)

            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)

    def _enhanced_to_legacy_traces(self) -> None:
        """Convert enhanced traces to legacy OperatorInfo format for compatibility"""
        self.operator_traces.clear()

        for enhanced_trace in self.enhanced_traces:
            # Convert enhanced tensor inputs/outputs to numpy arrays
            inputs_np = []
            for tensor in enhanced_trace.tensor_inputs:
                inputs_np.append(tensor.detach().cpu().numpy())

            outputs_np = []
            for tensor in enhanced_trace.tensor_outputs:
                outputs_np.append(tensor.detach().cpu().numpy())

            # Create legacy format trace
            legacy_trace = OperatorInfo(
                iteration=enhanced_trace.iteration,
                model_name=enhanced_trace.model_name,
                layer_name=enhanced_trace.layer_name,
                operator_name=enhanced_trace.operator_name,
                input_shapes=enhanced_trace.input_shapes,
                output_shapes=enhanced_trace.output_shapes,
                input_dtypes=enhanced_trace.input_dtypes,
                output_dtypes=enhanced_trace.output_dtypes,
                inputs=inputs_np,
                outputs=outputs_np,
                timestamp=enhanced_trace.timestamp
            )

            self.operator_traces.append(legacy_trace)

    def get_enhanced_operator_traces(self) -> List[EnhancedOperatorInfo]:
        """Get the enhanced operator traces"""
        return self.enhanced_traces

    def export_enhanced_traces_to_json(self, filepath: str) -> None:
        """Export enhanced traces to JSON file"""
        self.operator_capture_manager.export_to_json(filepath)

    def get_operator_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of captured operators"""
        stats = self.operator_capture_manager.get_operator_statistics()

        # Add information about different operator types
        if self.enhanced_traces:
            operator_types = {}
            for trace in self.enhanced_traces:
                op_type = trace.operator_type
                if op_type not in operator_types:
                    operator_types[op_type] = {
                        'count': 0,
                        'operators': set()
                    }
                operator_types[op_type]['count'] += 1
                operator_types[op_type]['operators'].add(trace.operator_name)

            # Convert sets to lists for JSON serialization
            for op_type in operator_types:
                operator_types[op_type]['operators'] = list(operator_types[op_type]['operators'])

            stats['operator_types'] = operator_types

        return stats

    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks from captured operators"""
        if not self.enhanced_traces:
            return {}

        # Sort operators by execution time
        sorted_by_time = sorted(self.enhanced_traces,
                              key=lambda x: x.execution_time_ms,
                              reverse=True)

        # Sort operators by memory allocation
        sorted_by_memory = sorted(self.enhanced_traces,
                                key=lambda x: x.memory_alloc_mb,
                                reverse=True)

        # Group by operator name for aggregate analysis
        time_by_operator = {}
        memory_by_operator = {}

        for trace in self.enhanced_traces:
            name = trace.operator_name
            time_by_operator[name] = time_by_operator.get(name, 0) + trace.execution_time_ms
            memory_by_operator[name] = memory_by_operator.get(name, 0) + trace.memory_alloc_mb

        # Sort aggregated data
        top_time_operators = sorted(time_by_operator.items(),
                                  key=lambda x: x[1],
                                  reverse=True)[:10]
        top_memory_operators = sorted(memory_by_operator.items(),
                                    key=lambda x: x[1],
                                    reverse=True)[:10]

        return {
            'slowest_individual_operations': [
                {
                    'operator': trace.operator_name,
                    'layer': trace.layer_name,
                    'execution_time_ms': trace.execution_time_ms,
                    'memory_alloc_mb': trace.memory_alloc_mb,
                    'call_site': trace.call_site
                }
                for trace in sorted_by_time[:10]
            ],
            'memory_heavy_operations': [
                {
                    'operator': trace.operator_name,
                    'layer': trace.layer_name,
                    'memory_alloc_mb': trace.memory_alloc_mb,
                    'execution_time_ms': trace.execution_time_ms,
                    'call_site': trace.call_site
                }
                for trace in sorted_by_memory[:10]
            ],
            'top_time_consumers_by_operator': top_time_operators,
            'top_memory_consumers_by_operator': top_memory_operators,
            'total_execution_time_ms': sum(t.execution_time_ms for t in self.enhanced_traces),
            'total_memory_alloc_mb': sum(t.memory_alloc_mb for t in self.enhanced_traces)
        }


# Example usage and custom operator registration
def register_example_custom_operators(dumper: EnhancedModelDumper):
    """Example of registering custom operators"""

    def custom_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Custom attention operation"""
        # Simplified attention implementation
        scores = torch.matmul(query, key.transpose(-2, -1))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)

    def custom_gating(input_tensor: torch.Tensor, gate: torch.Tensor):
        """Custom gating operation"""
        return input_tensor * torch.sigmoid(gate)

    # Register custom operators
    dumper.register_custom_operator("custom_attention", custom_attention, "custom")
    dumper.register_custom_operator("custom_gating", custom_gating, "custom")

    return custom_attention, custom_gating