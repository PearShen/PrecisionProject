# vLLM Operator Information Capture Guide

This guide explains how to use the enhanced vLLM operator information capture functionality in PrecisionProject.

## Overview

The vLLM operator capture system allows you to:
- Capture all execution operators during vLLM inference
- Track performance metrics (execution time, memory usage)
- Analyze operator-level execution patterns
- Export traces in multiple formats (JSON, HDF5, HTML, CSV)

## Supported vLLM Operators

### Attention Mechanisms
- `attention` - General attention operations
- `scaled_dot_product_attention` - Scaled dot-product attention
- `multi_head_attention` - Multi-head attention
- `paged_attention` - vLLM's paged attention kernel
- `flash_attn` - Flash attention implementation
- `xformers_attn` - xFormers attention backend

### Transformer Components
- `transformer_block` - Transformer block operations
- `decoder_layer` - Decoder layer forward pass
- `rms_norm` - RMS normalization
- `apply_rotary_emb` - Rotary positional embeddings

### Memory & Cache Operations
- `kv_cache` - Key-value cache operations
- `cache_ops` - Cache management operations
- `copy_blocks` - Block copying for KV cache
- `free_blocks` - Block deallocation

### Model Operations
- `model_forward` - Model forward pass
- `compute_logits` - Logits computation
- `sample` - Token sampling operations
- `greedy_sample` - Greedy sampling
- `top_k_sample` - Top-k sampling
- `top_p_sample` - Top-p (nucleus) sampling
- `temperature_sample` - Temperature-based sampling

### Distributed Operations
- `all_reduce` - All-reduce for tensor parallelism
- `all_gather` - All-gather operations
- `broadcast` - Broadcast operations

## Usage Examples

### Basic vLLM Operator Capture

```python
from model_dumper import ModelDumper

# Initialize model dumper with enhanced capture
model_dumper = ModelDumper(
    framework="vllm",
    enable_enhanced_capture=True
)

# vLLM model (can be actual vLLM LLM or mock for testing)
from vllm import LLM
model = LLM("meta-llama/Llama-2-7b-hf")

# Input configuration
input_data = {
    "prompts": "The quick brown fox jumps over the lazy dog",
    "params": {
        "max_tokens": 50,
        "temperature": 0.7
    }
}

# Run enhanced operator capture
model_dumper.dump_model_execution(
    model=model,
    input_data=input_data,
    output_path="./vllm_operator_trace",
    model_name="llama2_7b",
    iterations=3,
    capture_all_operators=True,
    save_enhanced_info=True
)
```

### Direct Operator Capture Manager Usage

```python
from operator_capture import OperatorCaptureManager
from vllm import LLM

# Initialize capture manager
manager = OperatorCaptureManager()

# Configure for vLLM operators
manager.configure_capture(
    capture_torch_ops=False,  # Disable torch ops
    capture_custom_ops=True,
    capture_vllm_ops=True,
    performance_timing=True,
    memory_tracking=True
)

# Register vLLM operators
model = LLM("meta-llama/Llama-2-7b-hf")
registered_ops = manager.register_vllm_operators(model)
print(f"Registered {len(registered_ops)} vLLM operators")

# Capture operators during inference
with manager.capture_context(
    model_name="llama2_7b_test",
    iteration=0,
    target_modules=[]
):
    outputs = model.generate("Hello, world!")

# Get captured traces
traces = manager.get_captured_operators()
print(f"Captured {len(traces)} operator traces")

# Export traces
manager.export_to_json("vllm_traces.json")

# Get statistics
stats = manager.get_operator_statistics()
print(f"Total operators: {stats['total_operators']}")
print(f"Total execution time: {stats['total_execution_time_ms']:.2f} ms")
```

## Output Files

When you run enhanced vLLM capture, the following files are generated:

### `model_info.json`
Basic model structure and metadata:
```json
{
    "framework": "vllm",
    "model_name": "llama2_7b",
    "model_type": "vLLM",
    "layers": [...]
}
```

### `enhanced_operator_traces.json`
Detailed operator traces with full metadata:
```json
[
    {
        "iteration": 0,
        "model_name": "llama2_7b",
        "operator_name": "paged_attention",
        "operator_type": "vllm_attention",
        "module_path": "vllm.attention.backends.paged_attn",
        "execution_time_ms": 15.23,
        "memory_alloc_mb": 2.45,
        "call_site": "llama_layers.py:123 in forward",
        "timestamp": 1703140925.123
    }
]
```

### `operator_statistics.json`
Comprehensive statistics:
```json
{
    "total_operators": 1523,
    "unique_operator_names": 23,
    "operator_count_by_name": {
        "paged_attention": 128,
        "rms_norm": 256,
        "scaled_dot_product_attention": 128
    },
    "execution_time_by_operator": {
        "paged_attention": {"count": 128, "total": 1950.4, "avg": 15.23}
    },
    "memory_alloc_by_operator": {
        "paged_attention": {"count": 128, "total": 313.6, "avg": 2.45}
    }
}
```

### `operator_traces.h5`
Legacy HDF5 format for compatibility with existing code.

## Advanced Configuration

### Selective Operator Capture

```python
# Capture only specific operators
manager.configure_capture(
    capture_vllm_ops=True,
    filter_operators=[
        'paged_attention',
        'scaled_dot_product_attention',
        'kv_cache'
    ]
)
```

### Performance Optimization

```python
# Disable expensive tracking for production
manager.configure_capture(
    performance_timing=False,  # Disable timing for performance
    memory_tracking=False      # Disable memory tracking
)
```

## Analyzing Captured Data

### Python Analysis Script

```python
import json
import matplotlib.pyplot as plt

# Load operator traces
with open("enhanced_operator_traces.json", "r") as f:
    traces = json.load(f)

# Analyze execution times
operator_times = {}
for trace in traces:
    op = trace["operator_name"]
    time = trace["execution_time_ms"]
    if op not in operator_times:
        operator_times[op] = []
    operator_times[op].append(time)

# Plot execution time distribution
plt.figure(figsize=(12, 6))
plt.boxplot([operator_times[op] for op in operator_times.keys()],
            labels=operator_times.keys())
plt.title("vLLM Operator Execution Time Distribution")
plt.ylabel("Time (ms)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Performance Bottleneck Identification

```python
# Find bottleneck operators
stats = manager.get_operator_statistics()
total_ops = stats['total_operators']

print("Top 5 operators by total execution time:")
exec_times = stats['execution_time_by_operator']
sorted_ops = sorted(exec_times.items(),
                   key=lambda x: x[1]['total'],
                   reverse=True)

for i, (op, data) in enumerate(sorted_ops[:5]):
    percentage = (data['total'] / stats['total_execution_time_ms']) * 100
    print(f"{i+1}. {op}: {data['total']:.2f}ms ({percentage:.1f}%)")
```

## Troubleshooting

### Common Issues

1. **No operators captured**
   - Ensure vLLM operators are properly registered
   - Check if model methods are correctly hooked to operators
   - Verify capture context is active during execution

2. **Import errors**
   - The system works without vLLM installed (uses mock operators)
   - For real vLLM capture, install vLLM: `pip install vllm`

3. **Performance impact**
   - Capture adds overhead (~10-20% slower)
   - Disable timing/memory tracking in production if needed

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check registered operators
manager = OperatorCaptureManager()
registered = manager.register_vllm_operators(model)
print("Operators:", registered)
print("Custom operators:", list(manager.custom_operators.keys()))
```

## Real vLLM Integration

When using with actual vLLM (not mocks), the system automatically detects and hooks into:
- Attention mechanisms in `vllm.attention.backends.*`
- Transformer layers in `vllm.model_executor.layers.*`
- Worker execution in `vllm.worker.*`

Example with real vLLM:
```python
from vllm import LLM, SamplingParams

# Real vLLM model
model = LLM("meta-llama/Llama-2-7b-hf")

# Configure sampling params
sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.8,
    top_p=0.95
)

# Run capture
model_dumper.dump_model_execution(
    model=model,
    input_data={"prompts": "Explain quantum computing", "params": sampling_params},
    output_path="./real_vllm_trace",
    capture_all_operators=True
)
```

## Summary

The enhanced vLLM operator capture system provides:
✅ **Comprehensive operator tracking** - All vLLM internal operations
✅ **Performance metrics** - Execution time and memory usage
✅ **Multiple export formats** - JSON, HDF5, statistics
✅ **Backwards compatibility** - Works with existing model dumper
✅ **Mock support** - Works without vLLM installation for testing
✅ **Production ready** - Configurable overhead and selective capture

This enables detailed analysis of vLLM inference performance and operator-level debugging for LLM deployment optimization.