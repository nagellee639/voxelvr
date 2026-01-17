
import numpy as np
import sys
try:
    from voxelvr.calibration.calibration_cpp import batch_detect_charuco
    print("Extension loaded successfully")
except ImportError as e:
    print(f"Failed to load extension: {e}")
    sys.exit(1)

def noop(*args): pass

print("Trying empty list...")
try:
    batch_detect_charuco([], 7, 5, 0.04, 0.03, 'DICT_6X6_250', noop)
    print("Empty list passed!")
except Exception as e:
    print(f"Empty list failed: {e}")

print("Trying list of one uint8 array...")
arr = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    batch_detect_charuco([arr], 7, 5, 0.04, 0.03, 'DICT_6X6_250', noop)
    print("One array passed!")
except Exception as e:
    print(f"One array failed: {e}")

print("Trying without callback (None)...")
try:
    batch_detect_charuco([arr], 7, 5, 0.04, 0.03, 'DICT_6X6_250', None)
    print("None callback passed!")
except Exception as e:
    print(f"None callback failed: {e}")
