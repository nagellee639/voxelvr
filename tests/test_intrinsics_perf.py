
import pytest
import numpy as np
import cv2
import time
from pathlib import Path
from voxelvr.config import CalibrationConfig
from voxelvr.calibration.intrinsics import calibrate_intrinsics
from voxelvr.calibration.charuco import create_charuco_board
import shutil

class TestIntrinsicsPerformance:
    def setup_class(self):
        self.config = CalibrationConfig()
        # Ensure we have a sufficient number of frames for a valid test
        self.n_frames = 20
        self.frames = []
        
        # Create a synthetic board image
        board, _ = create_charuco_board(
            self.config.charuco_squares_x,
            self.config.charuco_squares_y,
            self.config.charuco_square_length,
            self.config.charuco_marker_length,
            self.config.charuco_dict,
        )
        img = board.generateImage((1000, 1000), marginSize=0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Create variations (rotations/noise not strictly needed for perf, just valid detection)
        # But we want to simulate some workload.
        for i in range(self.n_frames):
            # Just copy for now, maybe rotate slightly if needed for validity check?
            # calibrateCamera needs different views.
            # Rotating the image 90 degrees or simple warps.
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), i * 5, 1)
            warped = cv2.warpAffine(img, M, (cols, rows))
            self.frames.append(warped)


    def test_calibration_performance(self):
        print("\nStarting performance test...")
        
        # Measure time
        start_time = time.time()
        
        intrinsics = calibrate_intrinsics(self.frames, self.config, camera_id=0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nCalibration took {duration:.4f} seconds for {self.n_frames} frames")
        
        assert intrinsics is not None
        assert intrinsics.reprojection_error < 1.0  # Should be very low for synthetic data
        
        # We can't easily assert the progress bar printed, but we can verify it ran without error
        # and checking logs if we capture stdout (pytest does).

if __name__ == "__main__":
    t = TestIntrinsicsPerformance()
    t.setup_class()
    t.test_calibration_performance()
