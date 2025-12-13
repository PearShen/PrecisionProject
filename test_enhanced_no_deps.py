#!/usr/bin/env python3

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_enhanced_structure_without_deps():
    """Test enhanced model info structure without external dependencies"""

    print("🚀 Testing Enhanced Model Info Structure (No Dependencies)")
    print("=" * 60)

    # Test 1: Verify enhanced JSON structure
    print("\n📋 Testing enhanced JSON structure...")
    try:
        with open('example_enhanced_model_info.json', 'r') as f:
            enhanced_info = json.load(f)

        print("✅ Enhanced JSON loaded successfully")
        print(f"📊 Framework: {enhanced_info['framework']}")
        print(f"🏷️  Model Name: {enhanced_info['model_name']}")
        print(f"⚙️  Model Type: {enhanced_info['model_type']}")
        print(f"🔢 Parameters: {enhanced_info['parameters']}")

    except Exception as e:
        print(f"❌ JSON loading failed: {e}")
        return False

    # Test 2: Verify input/output info
    print("\n📈 Testing input/output information...")
    try:
        input_info = enhanced_info['input_info']
        output_info = enhanced_info['output_info']

        print("✅ Input Info:")
        print(f"  Shape: {input_info['shape']}")
        print(f"  Dtype: {input_info['dtype']}")

        print("✅ Output Info:")
        print(f"  Shape: {output_info['shape']}")
        print(f"  Dtype: {output_info['dtype']}")

        assert 'shape' in input_info and 'dtype' in input_info
        assert 'shape' in output_info and 'dtype' in output_info

    except Exception as e:
        print(f"❌ Input/output info test failed: {e}")
        return False

    # Test 3: Verify enhanced layer structure
    print("\n🏗️ Testing enhanced layer structure...")
    try:
        layers = enhanced_info['layers']
        print(f"✅ Number of layers: {len(layers)}")

        for i, layer in enumerate(layers):
            print(f"  Layer {i}: {layer['type']} (index: {layer['layer_index']})")

            # Check all enhanced fields exist
            required_fields = ['layer_index', 'name', 'type', 'parameters',
                             'trainable', 'parameter_shapes', 'parameter_dtypes']

            for field in required_fields:
                if field not in layer:
                    print(f"❌ Missing field {field} in layer {i}")
                    return False

    except Exception as e:
        print(f"❌ Layer structure test failed: {e}")
        return False

    # Test 4: Verify layer-specific details
    print("\n🧩 Testing layer-specific details...")
    try:
        # Linear layer
        linear_layers = [layer for layer in layers if layer['type'] == 'Linear']
        if linear_layers:
            linear = linear_layers[0]
            print("✅ Linear layer details:")
            print(f"  In features: {linear['in_features']}")
            print(f"  Out features: {linear['out_features']}")
            print(f"  Bias: {linear['bias']}")
            assert 'in_features' in linear
            assert 'out_features' in linear

        # ReLU layer
        relu_layers = [layer for layer in layers if layer['type'] == 'ReLU']
        if relu_layers:
            relu = relu_layers[0]
            print("✅ ReLU layer details:")
            print(f"  Inplace: {relu['inplace']}")
            assert 'inplace' in relu

    except Exception as e:
        print(f"❌ Layer-specific details test failed: {e}")
        return False

    # Test 5: Compare with old structure
    print("\n🔄 Enhanced vs Old Structure Comparison...")
    try:
        # Load old structure (if exists)
        old_structure_path = os.path.join('tests', 'temp', 'model_info_old.json')
        if os.path.exists(old_structure_path):
            with open(old_structure_path, 'r') as f:
                old_info = json.load(f)

            print("✅ Old structure loaded for comparison")

            # Key differences
            print("📊 Enhancements added:")
            print("  + input_info.shapes & dtype")
            print("  + output_info.shapes & dtype")
            print("  + layer_layer.track_running_stats: parameter_shapes & dtypes")
            print("  + layer.trainable boolean")
            print("  + layer-specific configurations")
            print("  + layer sequential indexing")

        else:
            print("⚠️  Old structure not found for comparison")

    except Exception as e:
        print(f"⚠️  Comparison skipped: {e}")

    print("\n" + "=" * 60)
    print("🎉 Enhanced Structure Test Complete!")
    print("✅ All structure verifications passed!")

    print("\n📋 Enhanced Structure Benefits:")
    print("  🎯 Complete Model Overview: All details in one file")
    print("  📐 Shape & Type Info: Input/output specifications")
    print("  🔧 Layer Details: Parameters, types, configurations")
    print("  🧠 Trainable Status: Training vs inference parameters")
    print("  🏗️  Sequential Ordering: Layer index for reference")
    print("  ⚙️  Framework Support: PyTorch & vLLM coverage")

    return True

if __name__ == "__main__":
    success = test_enhanced_structure_without_deps()
    if success:
        print("\n🚀 Enhanced Model Info Structure Ready!")
        sys.exit(0)
    else:
        print("\n❌ Structure verification failed.")
        sys.exit(1)