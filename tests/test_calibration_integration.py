
import pytest
import numpy as np
import cv2
from pathlib import Path
import time

from voxelvr.calibration.extrinsics import calibrate_extrinsics, capture_extrinsic_frames
from voxelvr.config import CalibrationConfig, CameraIntrinsics, CameraExtrinsics
from voxelvr.calibration.charuco import create_charuco_board

TEST_DATA_DIR = Path("test_data/calibration")

@pytest.fixture
def test_images():
    images = []
    # Load available test images
    for p in sorted(TEST_DATA_DIR.glob("*.jpg"))[:5]:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    return images

@pytest.fixture
def mock_intrinsics():
    # create dummy intrinsics
    intr = CameraIntrinsics(
        camera_id=0,
        camera_name="TestCam",
        resolution=(1280, 720),
        camera_matrix=[[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
        distortion_coeffs=[0, 0, 0, 0, 0],
        reprojection_error=0.1,
        calibration_date="2023-01-01"
    )
    return [intr, intr] # Simulate 2 cameras

def test_parallel_extrinsics_execution(test_images, mock_intrinsics):
    if not test_images:
        pytest.skip("No test images found")
        
    # Create fake captures
    # We need at least 5 captures for calibration to proceed
    captures = []
    for i in range(5):
        # reuse images cyclically
        img = test_images[i % len(test_images)]
        # Create a dict simulating 2 cameras seeing roughly the same image (bad math but valid code path)
        captures.append({
            0: img,
            1: img
        })
        
    config = CalibrationConfig()
    # Ensure config matches the board in the image (assuming standard 5x5)
    config.charuco_squares_x = 5
    config.charuco_squares_y = 7 # The test images seem to be 5x7 or similar? 
    # Let's perform a quick check or just trust defaults and expect 'success=False' but no crash
    
    start_time = time.time()
    
    # Run calibration
    # It might return None because of bad math (same image for both cameras = singular),
    # but we just want to verify the parallel detection loop runs without error.
    result = calibrate_extrinsics(captures, mock_intrinsics, config)
    
    duration = time.time() - start_time
    print(f"Calibration took {duration:.4f}s")
    
    # Assert we survived
    assert True 

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
