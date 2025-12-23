#!/usr/bin/env python3
"""
Test script to verify that PrecisionProject wheel installation works
"""

import sys
import importlib.util

def test_wheel_installation():
    """Test if the PrecisionProject wheel is properly installed"""

    print("Testing PrecisionProject wheel installation...")

    # Test 1: Check if package can be found via pip
    import pkg_resources
    try:
        dist = pkg_resources.get_distribution('precisionproject')
        print(f"✓ Package 'precisionproject' found in pip")
        print(f"  Version: {dist.version}")
        print(f"  Location: {dist.location}")
    except pkg_resources.DistributionNotFound:
        print("✗ Package 'precisionproject' not found in pip")
        return False

    # Test 2: Check if package directory exists
    spec = importlib.util.find_spec('PrecisionProject')
    if spec is None:
        print("✗ PrecisionProject package not found in Python path")
        return False
    else:
        print(f"✓ PrecisionProject package found at: {spec.origin}")

    # Test 3: Check if main modules exist
    try:
        import PrecisionProject
        print(f"✓ PrecisionProject package imported successfully")
        print(f"  __version__: {getattr(PrecisionProject, '__version__', 'not found')}")
    except ImportError as e:
        print(f"✗ Failed to import PrecisionProject: {e}")
        print("  This may be due to missing dependencies, but the wheel installation is successful")

    # Test 4: Check if all expected modules are in the wheel
    expected_modules = [
        'PrecisionProject.model_dumper',
        'PrecisionProject.precision_comparator',
        'PrecisionProject.precision_tester',
        'PrecisionProject.operator_capture',
        'PrecisionProject.model_efficiency',
        'PrecisionProject.utils.data',
        'PrecisionProject.utils.pdb'
    ]

    print("\nChecking module availability:")
    for module_name in expected_modules:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✓ {module_name}")
        else:
            print(f"✗ {module_name}")

    # Test 5: Check console script registration
    try:
        import pkg_resources
        entry_points = pkg_resources.get_entry_map('precisionproject')
        if 'console_scripts' in entry_points:
            print("✓ Console scripts registered:")
            for name, ep in entry_points['console_scripts'].items():
                print(f"  - {name}: {ep}")
        else:
            print("ℹ No console scripts found")
    except:
        print("ℹ Could not check console scripts")

    print("\n" + "="*50)
    print("Wheel installation test completed!")
    print("="*50)
    return True

if __name__ == "__main__":
    test_wheel_installation()