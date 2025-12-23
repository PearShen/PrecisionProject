# PrecisionProject

A comprehensive precision testing framework for PyTorch and vLLM models.

## Features

-Dump model structure and operator execution traces with comprehensive multi-dimensional information
- Capture input/output data for each operator during forward pass with shapes and data types
- Compare results using absolute error, relative error, and cosine similarity
- Support for PyTorch and vLLM frameworks with detailed layer-specific information
- Enhanced model info with input/output shapes, dtypes, layer indices, trainable parameters, and configurations
- Organized data storage with iteration, model, layer, and operator metadata

## Installation

```bash
pip install -r requirements.txt
python setup.py sdist bdist_wheel
```

## Usage

```python
from PrecisionProject import PrecisionTester

# Initialize tester
tester = PrecisionTester()

# Dump model execution
tester.dump_model_execution(model, input_data, output_path="./dumps")

# Compare precision
results = tester.compare_precision(golden_path="./golden", test_path="./test")
```