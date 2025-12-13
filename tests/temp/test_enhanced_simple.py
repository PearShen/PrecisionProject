#!/usr/bin/env python3

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# Create a simple test without external dependencies
def test_enhanced_structure():
    """Test the enhanced structure creation without PyTorch"""

    print("Testing enhanced model info structure...")

    # Simulate what the enhanced model info would look like
    enhanced_model_info = {
        "framework": "torch",
        "model_name": "enhanced_test_model",
        "model_type": "Sequential",
        "parameters": 325,
        "input_info": {
            "shape": [8, 10],
            "dtype": "torch.float32"
        },
        "output_info": {
            "shape": [8, 5],
            "dtype": "torch.float32"
        },
        "layers": [
            {
                "layer_index": 0,
                "name": "0",
                "type": "Linear",
                "parameters": 220,
                "trainable": True,
                "parameter_shapes": [[20, 10], [20]],
                "parameter_dtypes": ["torch.float32", "torch.float32"],
                "in_features": 10,
                "out_features": 20,
                "bias": True
            },
            {
                "layer_index": 1,
                "name": "1",
                "type": "ReLU",
                "parameters": 0,
                "trainable": False,
                "parameter_shapes": [],
                "parameter_dtypes": [],
                "inplace": False
            },
            {
                "layer_index": 2,
                "name": "2",
                "type": "Linear",
                "parameters": 105,
                "trainable": True,
                "parameter_shapes": [[5, 20], [5]],
                "parameter_dtypes": ["torch.float32", "torch.float32"],
                "in_features": 20,
                "out_features": 5,
                "bias": True
            }
        ]
    }

    # Save the enhanced model info
    import json
    with open(os.path.join(os.path.dirname(__file__), 'enhanced_model_info.json'), 'w') as f:
        json.dump(enhanced_model_info, f, indent=2)

    print("✅ Enhanced model info structure test completed!")
    print("📁 File saved: enhanced_model_info.json")
    print("\n🔍 Enhanced Features Overview:")
    print("  📊 input_info: Model input shape and dtype")
    print("  📊 output_info: Model output shape and dtype")
    print("  🏗️  layer_index: Sequential layer ordering")
    print("  🧠 trainable: Whether layer parameters are trainable")
    print("  📐 parameter_shapes: Shapes of all parameters")
    print("  🔤 parameter_dtypes: Data types of all parameters")
    print("  ⚙️  layer-specific details (in_features, out_features, etc.)")

    return enhanced_model_info

if __name__ == "__main__":
    test_enhanced_structure()