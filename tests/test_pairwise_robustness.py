
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from voxelvr.calibration.pairwise_extrinsics import calibrate_camera_pair, is_valid_transform
from voxelvr.config import CalibrationConfig, CameraIntrinsics

class TestPairwiseRobustness:
    """Tests for robustness of pairwise calibration."""

    @pytest.fixture
    def mock_config(self):
        return CalibrationConfig(
            charuco_squares_x=5,
            charuco_squares_y=5,
            charuco_square_length=0.04,
            charuco_marker_length=0.03,
            charuco_dict="DICT_6X6_250",
        )

    @pytest.fixture
    def mock_intrinsics(self):
        intrinsics_a = CameraIntrinsics(
            camera_id=0,
            camera_name="Camera 0",
            resolution=(640, 480),
            camera_matrix=[[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            distortion_coeffs=[0, 0, 0, 0, 0],
            reprojection_error=0.5,
            calibration_date="now",
        )
        intrinsics_b = CameraIntrinsics(
            camera_id=1,
            camera_name="Camera 1",
            resolution=(640, 480),
            camera_matrix=[[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            distortion_coeffs=[0, 0, 0, 0, 0],
            reprojection_error=0.5,
            calibration_date="now",
        )
        return intrinsics_a, intrinsics_b

    def test_reject_high_reprojection_error(self, mock_config, mock_intrinsics):
        """Test that calibration with high reprojection error is rejected."""
        intrinsics_a, intrinsics_b = mock_intrinsics
        
        # We need to mock the internal functions that compute transforms
        # so we can inject a scenario that produces high variance (error)
        
        with patch('voxelvr.calibration.pairwise_extrinsics.detect_charuco') as mock_detect, \
             patch('voxelvr.calibration.pairwise_extrinsics.estimate_pose') as mock_estimate:
             
            # Setup successful detection
            mock_detect.return_value = {'success': True, 'corners': np.zeros((10, 1, 2)), 'ids': np.zeros((10, 1))}
            
            # Setup poses that are inconsistent (high variance)
            # Frame 1: Identity
            # Frame 2: Identity
            # Frame 3: Identity + large translation
            
            # estimate_pose returns (success, rvec, tvec)
            # We need to return values for:
            # Frame 1: Cam A, Cam B
            # Frame 2: Cam A, Cam B
            # Frame 3: Cam A, Cam B
            
            # Cam A always sees board at origin
            rvec_a = np.zeros(3)
            tvec_a = np.zeros(3)
            
            # Cam B sees board differently each time (simulating noise/error)
            # Frame 1: B is at [1, 0, 0] relative to board
            rvec_b1 = np.zeros(3)
            tvec_b1 = np.array([1.0, 0.0, 0.0])
            
            # Frame 2: B is at [1, 0, 0]
            rvec_b2 = np.zeros(3)
            tvec_b2 = np.array([1.0, 0.0, 0.0])
            
            # Frame 3: B is at [2, 0, 0] (Large jump -> high variance/error)
            rvec_b3 = np.zeros(3)
            tvec_b3 = np.array([2.0, 0.0, 0.0])
            
            mock_estimate.side_effect = [
                # Frame 1
                (True, rvec_a, tvec_a), (True, rvec_b1, tvec_b1),
                # Frame 2
                (True, rvec_a, tvec_a), (True, rvec_b2, tvec_b2),
                # Frame 3
                (True, rvec_a, tvec_a), (True, rvec_b3, tvec_b3),
            ]
            
            captures = [
                {0: np.zeros((480, 640, 3)), 1: np.zeros((480, 640, 3))},
                {0: np.zeros((480, 640, 3)), 1: np.zeros((480, 640, 3))},
                {0: np.zeros((480, 640, 3)), 1: np.zeros((480, 640, 3))},
            ]
            
            result = calibrate_camera_pair(captures, intrinsics_a, intrinsics_b, mock_config)
            
            # Should be rejected due to high error (the large jump creates high variance)
            assert result is None

    def test_accept_low_reprojection_error(self, mock_config, mock_intrinsics):
        """Test that calibration with low reprojection error is accepted."""
        intrinsics_a, intrinsics_b = mock_intrinsics
        
        with patch('voxelvr.calibration.pairwise_extrinsics.detect_charuco') as mock_detect, \
             patch('voxelvr.calibration.pairwise_extrinsics.estimate_pose') as mock_estimate:
             
            mock_detect.return_value = {'success': True, 'corners': np.zeros((10, 1, 2)), 'ids': np.zeros((10, 1))}
            
            rvec_a = np.zeros(3)
            tvec_a = np.zeros(3)
            
            # Consistent poses
            rvec_b = np.zeros(3)
            tvec_b = np.array([1.0, 0.0, 0.0])
            
            mock_estimate.side_effect = [
                (True, rvec_a, tvec_a), (True, rvec_b, tvec_b),
                (True, rvec_a, tvec_a), (True, rvec_b, tvec_b),
                (True, rvec_a, tvec_a), (True, rvec_b, tvec_b),
            ]
            
            captures = [
                {0: np.zeros((480, 640, 3)), 1: np.zeros((480, 640, 3))},
                {0: np.zeros((480, 640, 3)), 1: np.zeros((480, 640, 3))},
                {0: np.zeros((480, 640, 3)), 1: np.zeros((480, 640, 3))},
            ]
            
            result = calibrate_camera_pair(captures, intrinsics_a, intrinsics_b, mock_config)
            
            assert result is not None
            assert result.reprojection_error < 5.0
