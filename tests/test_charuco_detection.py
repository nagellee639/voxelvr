
import pytest
import cv2
import numpy as np
import sys
from voxelvr.calibration.charuco import detect_charuco, create_charuco_board, generate_charuco_img

class TestCharucoDetection:
    """Integration tests for Charuco detection pipeline."""

    def test_detect_charuco_pipeline(self):
        """
        Simulate the entire detection pipeline:
        1. Generate a board image
        2. Detect corners in it
        3. Verify it doesn't crash and finds corners
        """
        # 1. Generate a valid Charuco board image
        board_img = generate_charuco_img(
            squares_x=5,
            squares_y=5, 
            square_length=0.04,
            marker_length=0.03,
            dictionary="DICT_6X6_250"
        )
        
        # 2. Setup board object (needed for detection)
        board, aruco_dict = create_charuco_board(
            squares_x=5,
            squares_y=5,
            square_length=0.04,
            marker_length=0.03,
            dictionary="DICT_6X6_250"
        )
        
        # 3. Detect (This is where it crashed)
        try:
            result = detect_charuco(board_img, board, aruco_dict)
        except AttributeError as e:
            pytest.fail(f"Detection failed with AttributeError: {e}")
        except Exception as e:
            pytest.fail(f"Detection failed with Exception: {e}")
            
        # 4. Verification
        assert result['success'] == True, "Should detect corners in generated perfect image"
        assert result['corners'] is not None
        assert len(result['corners']) > 0

    def test_opencv_version_check(self):
        """Check and print OpenCV version for debug context."""
        print(f"OpenCV Version: {cv2.__version__}")
        # We don't assert here, just useful info if other tests fail
