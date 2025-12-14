"""
Enhanced operator capture system for comprehensive model operation tracking.
Supports both PyTorch native operators and custom operators.
"""

import torch
import torch.nn as nn
import inspect
import functools
import time
import json
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import weakref


@dataclass
class EnhancedOperatorInfo:
    """Enhanced information about an operator execution"""
    # Basic info
    iteration: int
    model_name: str
    operator_name: str
    operator_type: str  # 'torch', 'custom', 'autograd'

    # Location info
    module_path: str  # Full module path
    layer_name: str   # Layer/module name
    call_site: str    # File and line where called

    # Execution info
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    input_dtypes: List[str]
    output_dtypes: List[str]

    # Performance metrics
    execution_time_ms: float
    memory_alloc_mb: float

    # Arguments
    arguments: Dict[str, Any]  # Non-tensor arguments
    tensor_inputs: List[torch.Tensor]
    tensor_outputs: List[torch.Tensor]

    # Context
    timestamp: float
    thread_id: int
    context_info: Dict[str, Any]  # Additional context


class OperatorCaptureManager:
    """
    Centralized manager for capturing all operator calls during model execution.

    Features:
    - Automatic PyTorch operator registration
    - Custom operator registration
    - Performance timing
    - Memory tracking
    - Thread-safe operation
    - Context filtering
    """

    def __init__(self):
        self.captured_operators: List[EnhancedOperatorInfo] = []
        self.is_capturing = False
        self.capture_config = {}
        self.torch_patches = {}
        self.custom_operators = {}
        self.iteration_count = 0
        self.model_name = ""
        self.lock = threading.Lock()
        self.active_contexts = set()

        # Default operators to capture
        self.default_torch_operators = [
            # Math operations
            'add', 'sub', 'mul', 'div', 'pow', 'exp', 'log', 'sqrt', 'sum',
            'mean', 'max', 'min', 'matmul', 'linear',
            # Neural operations
            'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d',
            'batch_norm', 'layer_norm', 'group_norm',
            'relu', 'gelu', 'sigmoid', 'tanh', 'softmax',
            # Tensor operations
            'view', 'reshape', 'permute', 'transpose', 'squeeze', 'unsqueeze',
            'concat', 'stack', 'split', 'chunk', 'gather', 'scatter',
            # Pooling
            'max_pool1d', 'max_pool2d', 'max_pool3d',
            'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
            'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_avg_pool1d', 'adaptive_avg_pool2d',
            # Dropout
            'dropout', 'dropout1d', 'dropout2d', 'dropout3d',
            # Embedding
            'embedding', 'embedding_bag',
            # RNN
            'lstm', 'gru', 'rnn',
        ]

        # vLLM-specific operators to capture
        self.default_vllm_operators = [
            # vLLM attention mechanisms
            'attention',
            'scaled_dot_product_attention',
            'multi_head_attention',
            # vLLM transformer blocks
            'transformer_block',
            'decoder_layer',
            'rms_norm',
            'apply_rotary_emb',
            # vLLM KV cache operations
            'kv_cache',
            'cache_ops',
            'copy_blocks',
            'free_blocks',
            # vLLM modeling operations
            'model_forward',
            'compute_logits',
            'sample',
            'greedy_sample',
            'top_k_sample',
            'top_p_sample',
            'temperature_sample',
            # vLLM tensor parallel operations
            'all_reduce',
            'all_gather',
            'broadcast',
            # vLLM custom kernels
            'paged_attention',
            'flash_attn',
            'xformers_attn',
        ]

    def configure_capture(self,
                         capture_torch_ops: bool = True,
                         capture_custom_ops: bool = True,
                         include_autograd: bool = False,
                         performance_timing: bool = True,
                         memory_tracking: bool = True,
                         capture_vllm_ops: bool = False,
                         filter_operators: Optional[List[str]] = None):
        """Configure what to capture during execution"""
        # Determine default operators based on capture settings
        default_ops = []
        if capture_torch_ops:
            default_ops.extend(self.default_torch_operators)
        if capture_vllm_ops:
            default_ops.extend(self.default_vllm_operators)

        self.capture_config = {
            'capture_torch_ops': capture_torch_ops,
            'capture_custom_ops': capture_custom_ops,
            'capture_vllm_ops': capture_vllm_ops,
            'include_autograd': include_autograd,
            'performance_timing': performance_timing,
            'memory_tracking': memory_tracking,
            'filter_operators': filter_operators or default_ops
        }

    @contextmanager
    def capture_context(self,
                       model_name: str,
                       iteration: int = 0,
                       target_modules: Optional[List[str]] = None):
        """Context manager for capturing operator calls"""
        if self.is_capturing:
            raise RuntimeError("Capture already in progress")

        self.is_capturing = True
        self.model_name = model_name
        self.iteration_count = iteration

        # Store original functions
        patched_functions = {}

        try:
            # Patch torch operations
            if self.capture_config.get('capture_torch_ops', True):
                patched_functions.update(self._patch_torch_operations())

            # Add module context tracking
            self._setup_module_context_tracking(target_modules)

            yield self

        finally:
            # Restore original functions
            self._restore_patched_functions(patched_functions)
            self._cleanup_module_context_tracking()
            self.is_capturing = False

    def register_custom_operator(self,
                               name: str,
                               func: Callable,
                               operator_type: str = 'custom',
                               module_path: str = None):
        """Register a custom operator for capture"""
        if module_path is None:
            module_path = inspect.getmodule(func).__name__

        self.custom_operators[name] = {
            'func': func,
            'type': operator_type,
            'module_path': module_path
        }

        # Patch the custom operator
        wrapped_func = self._wrap_operator(func, name, operator_type, module_path)
        self.custom_operators[name]['wrapped'] = wrapped_func

    def register_vllm_operators(self, vllm_model: Any):
        """Register vLLM-specific operators for capture"""
        vllm_operators_found = []

        # Try to import vLLM
        try:
            import vllm
            from vllm import attention
            from vllm.model_executor.layers.attention import Attention
            from vllm.model_executor.layers.transformer_layer import TransformerLayer
            from vllm.worker.worker import Worker

            # Register vLLM attention operations
            if hasattr(attention, 'ScaledDotProductAttention'):
                self.register_custom_operator(
                    'scaled_dot_product_attention',
                    attention.ScaledDotProductAttention.forward,
                    'vllm_attention',
                    'vllm.attention'
                )
                vllm_operators_found.append('scaled_dot_product_attention')

            # Register transformer layer operations
            if hasattr(TransformerLayer, 'forward'):
                self.register_custom_operator(
                    'transformer_layer',
                    TransformerLayer.forward,
                    'vllm_transformer',
                    'vllm.model_executor.layers.transformer_layer'
                )
                vllm_operators_found.append('transformer_layer')

            # Try to register KV cache operations
            try:
                from vllm.attention.backends.paged_attn import PagedAttention
                if hasattr(PagedAttention, 'forward'):
                    self.register_custom_operator(
                        'paged_attention',
                        PagedAttention.forward,
                        'vllm_kv_cache',
                        'vllm.attention.backends.paged_attn'
                    )
                    vllm_operators_found.append('paged_attention')
            except ImportError:
                pass

            # Register worker operations
            if hasattr(Worker, 'execute_model'):
                self.register_custom_operator(
                    'execute_model',
                    Worker.execute_model,
                    'vllm_worker',
                    'vllm.worker'
                )
                vllm_operators_found.append('execute_model')

        except ImportError:
            # vLLM not installed, register mock operators for testing
            class MockOperator:
                def __init__(self, name):
                    self.name = name
                def __call__(self, *args, **kwargs):
                    # Simulate operator execution with timing and memory
                    import time
                    import random
                    start_time = time.perf_counter()
                    # Simulate some work
                    time.sleep(random.uniform(0.001, 0.005))
                    end_time = time.perf_counter()
                    return f"mock_{self.name}_result", (end_time - start_time) * 1000

            for op_name in self.default_vllm_operators:
                mock_op = MockOperator(op_name)
                self.register_custom_operator(
                    op_name,
                    mock_op,
                    'vllm_mock',
                    'vllm.mock'
                )
                vllm_operators_found.append(op_name)

        return vllm_operators_found

    def _patch_torch_operations(self) -> Dict[str, Callable]:
        """Patch torch operations for capture"""
        patched = {}

        for op_name in self.capture_config.get('filter_operators', self.default_torch_operators):
            if hasattr(torch, op_name):
                original_func = getattr(torch, op_name)
                wrapped_func = self._wrap_operator(
                    original_func,
                    op_name,
                    'torch',
                    f'torch.{op_name}'
                )
                setattr(torch, op_name, wrapped_func)
                self.torch_patches[op_name] = original_func
                patched[op_name] = original_func

            # Also check torch.nn.functional
            if hasattr(torch.nn.functional, op_name):
                original_func = getattr(torch.nn.functional, op_name)
                wrapped_func = self._wrap_operator(
                    original_func,
                    op_name,
                    'torch',
                    f'torch.nn.functional.{op_name}'
                )
                setattr(torch.nn.functional, op_name, wrapped_func)
                self.torch_patches[f'nn.functional.{op_name}'] = original_func
                patched[f'nn.functional.{op_name}'] = original_func

        # Patch custom operators if enabled
        if self.capture_config.get('capture_custom_ops', True):
            for op_name, op_info in self.custom_operators.items():
                if 'wrapped' in op_info:
                    # We need to replace the original in its module
                    module_path = op_info['module_path']
                    if '.' in module_path:
                        parts = module_path.split('.')
                        try:
                            module = __import__(module_path)
                            for part in parts[1:]:
                                module = getattr(module, part)
                        except ImportError:
                            # For modules that can't be imported, skip patching
                            continue

                        if hasattr(module, op_name):
                            original_func = getattr(module, op_name)
                            setattr(module, op_name, op_info['wrapped'])
                            self.torch_patches[f'{module_path}.{op_name}'] = original_func
                            patched[f'{module_path}.{op_name}'] = original_func

        return patched

    def _wrap_operator(self,
                      original_func: Callable,
                      op_name: str,
                      op_type: str,
                      module_path: str) -> Callable:
        """Wrap an operator function for capture"""
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            if not self.is_capturing:
                return original_func(*args, **kwargs)

            # Get call site information
            frame = inspect.currentframe().f_back
            call_site = self._get_call_site(frame)

            # Pre-execution tracking
            start_time = time.perf_counter() if self.capture_config.get('performance_timing', True) else None
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() and self.capture_config.get('memory_tracking', True) else 0

            # Extract inputs
            tensor_inputs, input_shapes, input_dtypes = self._extract_tensor_info(args, kwargs)

            # Extract non-tensor arguments
            arguments = self._extract_non_tensor_args(args, kwargs)

            try:
                # Execute original function
                result = original_func(*args, **kwargs)

                # Post-execution tracking
                end_time = time.perf_counter() if self.capture_config.get('performance_timing', True) else None
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() and self.capture_config.get('memory_tracking', True) else 0

                # Extract outputs
                tensor_outputs, output_shapes, output_dtypes = self._extract_tensor_info([result] if not isinstance(result, (list, tuple)) else result, {})

                # Calculate metrics
                execution_time_ms = (end_time - start_time) * 1000 if end_time else 0
                memory_alloc_mb = (end_memory - start_memory) / (1024 * 1024) if start_memory >= 0 else 0

                # Determine context info
                layer_name = self._get_current_layer_name()

                # Create operator info
                operator_info = EnhancedOperatorInfo(
                    iteration=self.iteration_count,
                    model_name=self.model_name,
                    operator_name=op_name,
                    operator_type=op_type,
                    module_path=module_path,
                    layer_name=layer_name,
                    call_site=call_site,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    input_dtypes=input_dtypes,
                    output_dtypes=output_dtypes,
                    execution_time_ms=execution_time_ms,
                    memory_alloc_mb=memory_alloc_mb,
                    arguments=arguments,
                    tensor_inputs=tensor_inputs,
                    tensor_outputs=tensor_outputs,
                    timestamp=time.time(),
                    thread_id=threading.get_ident(),
                    context_info={}
                )

                # Thread-safe addition
                with self.lock:
                    self.captured_operators.append(operator_info)

                return result

            except Exception as e:
                # Log error and re-raise
                print(f"Error capturing operator {op_name}: {e}")
                raise

        return wrapper

    def _extract_tensor_info(self,
                           args: Union[tuple, list],
                           kwargs: dict) -> Tuple[List[torch.Tensor], List[List[int]], List[str]]:
        """Extract tensor information from arguments"""
        tensors = []
        shapes = []
        dtypes = []

        def process_item(item):
            if isinstance(item, torch.Tensor):
                tensors.append(item)
                shapes.append(list(item.shape))
                dtypes.append(str(item.dtype))
            elif isinstance(item, (list, tuple)):
                for subitem in item:
                    process_item(subitem)

        # Process positional arguments
        for arg in args:
            process_item(arg)

        # Process keyword arguments
        for kwarg in kwargs.values():
            process_item(kwarg)

        return tensors, shapes, dtypes

    def _extract_non_tensor_args(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract non-tensor arguments for logging"""
        def serialize_if_needed(obj):
            if isinstance(obj, torch.Tensor):
                return f"<Tensor: shape={list(obj.shape)}, dtype={obj.dtype}>"
            elif isinstance(obj, (list, tuple)):
                return [serialize_if_needed(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize_if_needed(v) for k, v in obj.items()}
            else:
                try:
                    json.dumps(obj)  # Test if serializable
                    return obj
                except:
                    return str(obj)

        args_serializable = serialize_if_needed(args)
        kwargs_serializable = serialize_if_needed(kwargs)

        return {
            'args': args_serializable,
            'kwargs': kwargs_serializable
        }

    def _get_call_site(self, frame) -> str:
        """Get call site information"""
        try:
            return f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"
        except:
            return "unknown"

    def _get_current_layer_name(self) -> str:
        """Get current active layer name from context"""
        # This can be enhanced with proper context tracking
        return "unknown"

    def _setup_module_context_tracking(self, target_modules: Optional[List[str]]):
        """Setup context tracking for better layer identification"""
        # Placeholder for module context tracking
        # Could use hooks or monkey patching to track active modules
        pass

    def _cleanup_module_context_tracking(self):
        """Cleanup module context tracking"""
        pass

    def _restore_patched_functions(self, patched_functions: Dict[str, Callable]):
        """Restore original patched functions"""
        for key, original_func in self.torch_patches.items():
            if '.' in key:
                # Handle nested attributes like torch.nn.functional
                parts = key.split('.')
                module = torch
                for part in parts[:-1]:
                    module = getattr(module, part)
                setattr(module, parts[-1], original_func)
            else:
                setattr(torch, key, original_func)

        self.torch_patches.clear()

    def get_captured_operators(self) -> List[Dict[str, Any]]:
        """Get captured operators as dictionaries"""
        with self.lock:
            return [asdict(op) for op in self.captured_operators]

    def clear_captured_operators(self):
        """Clear captured operators"""
        with self.lock:
            self.captured_operators.clear()

    def export_to_json(self, filepath: str):
        """Export captured operators to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_captured_operators(), f, indent=2, default=str)

    def get_operator_statistics(self) -> Dict[str, Any]:
        """Get statistics about captured operators"""
        if not self.captured_operators:
            return {}

        operators = [asdict(op) for op in self.captured_operators]

        stats = {
            'total_operators': len(operators),
            'unique_operator_names': len(set(op['operator_name'] for op in operators)),
            'operator_count_by_name': {},
            'operator_count_by_type': {},
            'total_execution_time_ms': sum(op['execution_time_ms'] for op in operators),
            'total_memory_alloc_mb': sum(op['memory_alloc_mb'] for op in operators),
            'execution_time_by_operator': {},
            'memory_alloc_by_operator': {}
        }

        for op in operators:
            name = op['operator_name']
            op_type = op['operator_type']
            exec_time = op['execution_time_ms']
            memory = op['memory_alloc_mb']

            # Count by name
            stats['operator_count_by_name'][name] = stats['operator_count_by_name'].get(name, 0) + 1

            # Count by type
            stats['operator_count_by_type'][op_type] = stats['operator_count_by_type'].get(op_type, 0) + 1

            # Execution time by operator
            if name not in stats['execution_time_by_operator']:
                stats['execution_time_by_operator'][name] = {'count': 0, 'total': 0, 'avg': 0}
            stats['execution_time_by_operator'][name]['count'] += 1
            stats['execution_time_by_operator'][name]['total'] += exec_time

            # Memory allocation by operator
            if name not in stats['memory_alloc_by_operator']:
                stats['memory_alloc_by_operator'][name] = {'count': 0, 'total': 0, 'avg': 0}
            stats['memory_alloc_by_operator'][name]['count'] += 1
            stats['memory_alloc_by_operator'][name]['total'] += memory

        # Calculate averages
        for name, data in stats['execution_time_by_operator'].items():
            data['avg'] = data['total'] / data['count']

        for name, data in stats['memory_alloc_by_operator'].items():
            data['avg'] = data['total'] / data['count']

        return stats