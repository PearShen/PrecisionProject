#!/usr/bin/env python3

import sys
import os
import torch
import torch.nn as nn
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from model_dumper import ModelDumper

def test_enhanced_model_info():
    """Test enhanced model info functionality"""

    # Create test model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(20, 5),
        nn.Softmax(dim=1)
    )

    input_data = torch.randn(8, 10)

    # Create dumper
    dumper = ModelDumper(framework="torch")

    # Create output directory
    output_dir = "./test_enhanced_output"
    os.makedirs(output_dir, exist_ok=True)

    print("Dumping model execution with enhanced info...")
    dumper.dump_model_execution(
        model, input_data, output_dir, "enhanced_test_model", iterations=1
    )

    # Load and display model info
    with open(os.path.join(output_dir, "model_info.json"), "r") as f:
        model_info = json.load(f)

    print("\n=== Enhanced Model Info ===")
    print(json.dumps(model_info, indent=2))

    # Verify enhanced features
    print("\n=== Verification ===")
    print(f"Framework: {model_info['framework']}")
    print(f"Model Name: {model_info['model_name']}")
    print(f"Input Shape: {model_info['input_info']['shape']}")
    print(f"Input Dtype: {model_info['input_info']['dtype']}")
    print(f"Output Shape: {model_info['output_info']['shape']}")
    print(f"Output Dtype: {model_info['output_info']['dtype']}")

    print(f"\nLayer Details:")
    for i, layer in enumerate(model_info['layers']):
        print(f"  Layer {i}: {layer['name']} ({layer['type']})")
        print(f"    - Parameters: {layer['parameters']}")
        print(f"    - Trainable: {layer['trainable']}")
        print(f"    - Layer Index: {layer.get('layer_index', 'N/A')}")

        if layer['type'] == 'Linear':
            print(f"    - In Features: {layer.get('in_features', 'N/A')}")
            print(f"    - Out Features: {layer.get('out_features', 'N/A')}")
            print(f"    - Bias: {layer.get('bias', 'N/A')}")
        elif layer['type'] == 'Dropout':
            print(f"    - Dropout Rate: {layer.get('p', 'N/A')}")
            print(f"    - Inplace: {layer.get('inplace', 'N/A')}")
        elif layer['type'] == 'ReLU':
            print(f"    - Inplace: {layer.get('inplace', 'N/A')}")

    print(f"\nParameter Details for Linear Layer:")
    linear_layer = model_info['layers'][0]  # First linear layer
    print(f"  - Parameter Shapes: {linear_layer['parameter_shapes']}")
    print(f"  - Parameter Dtypes: {linear_layer['parameter_dtypes']}")

    print(f"\n✅ Enhanced model info test completed successfully!")
    print(f"📁 Results saved in: {output_dir}")

    return True

if __name__ == "__main__":
    test_enhanced_model_info()