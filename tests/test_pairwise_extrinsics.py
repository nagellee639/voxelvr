"""
Tests for pairwise extrinsics calibration module.

Tests cover:Comprehensive unit tests for the pairwise extrinsics module.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from voxelvr.calibration.pairwise_extrinsics import (
    PairwiseCalibrationResult,
    calibrate_camera_pair,
    chain_pairwise_transforms,
    calibrate_pairwise_extrinsics,
    get_required_pairs,
    check_connectivity,
)
from voxelvr.calibration.extrinsics import create_transform_matrix, invert_transform


class TestPairwiseCalibrationResult:
    """Tests for PairwiseCalibrationResult dataclass."""
    
    def test_creation(self):
        """Test creating a pairwise result."""
        T = np.eye(4)
        result = PairwiseCalibrationResult(
            camera_a=0,
            camera_b=2,
            transform_a_to_b=T,
            reprojection_error=0.5,
            num_frames_used=10,
        )
        
        assert result.camera_a == 0
        assert result.camera_b == 2
        assert np.array_equal(result.transform_a_to_b, T)
        assert result.reprojection_error == 0.5
        assert result.num_frames_used == 10


class TestCheckConnectivity:
    """Tests for the connectivity check function."""
    
    def test_empty_results(self):
        """Empty results should show all cameras disconnected."""
        is_connected, disconnected = check_connectivity([], [0, 2, 4])
        assert not is_connected
        assert disconnected == {0, 2, 4}
    
    def test_single_pair(self):
        """Single pair connects only 2 cameras."""
        results = [
            PairwiseCalibrationResult(0, 2, np.eye(4), 0.1, 10),
        ]
        
        is_connected, disconnected = check_connectivity(results, [0, 2, 4])
        assert not is_connected
        assert 4 in disconnected
        assert 0 not in disconnected
        assert 2 not in disconnected
    
    def test_chain_connected(self):
        """Chain of pairs should connect all cameras."""
        results = [
            PairwiseCalibrationResult(0, 2, np.eye(4), 0.1, 10),
            PairwiseCalibrationResult(2, 4, np.eye(4), 0.1, 10),
            PairwiseCalibrationResult(4, 6, np.eye(4), 0.1, 10),
        ]
        
        is_connected, disconnected = check_connectivity(results, [0, 2, 4, 6])
        assert is_connected
        assert len(disconnected) == 0
    
    def test_star_topology(self):
        """Star topology (all connected to center) should work."""
        results = [
            PairwiseCalibrationResult(0, 2, np.eye(4), 0.1, 10),
            PairwiseCalibrationResult(0, 4, np.eye(4), 0.1, 10),
            PairwiseCalibrationResult(0, 6, np.eye(4), 0.1, 10),
        ]
        
        is_connected, disconnected = check_connectivity(results, [0, 2, 4, 6])
        assert is_connected
        assert len(disconnected) == 0
    
    def test_disconnected_subgraph(self):
        """Disconnected subgraph should be detected."""
        results = [
            PairwiseCalibrationResult(0, 2, np.eye(4), 0.1, 10),
            PairwiseCalibrationResult(4, 6, np.eye(4), 0.1, 10),  # Disconnected from 0,2
        ]
        
        is_connected, disconnected = check_connectivity(results, [0, 2, 4, 6])
        assert not is_connected
        # Either 4,6 or 0,2 will be in disconnected depending on start point
        assert len(disconnected) == 2


class TestChainPairwiseTransforms:
    """Tests for the transform chaining function."""
    
    def test_empty_results(self):
        """Empty results should return empty dict."""
        transforms = chain_pairwise_transforms([])
        assert transforms == {}
    
    def test_single_pair(self):
        """Single pair should give two transforms."""
        T_0_to_2 = np.eye(4)
        T_0_to_2[0, 3] = 1.0  # 1m offset in X
        
        results = [
            PairwiseCalibrationResult(0, 2, T_0_to_2, 0.1, 10),
        ]
        
        transforms = chain_pairwise_transforms(results, reference_camera=0)
        
        assert 0 in transforms
        assert 2 in transforms
        
        # Camera 0 should be at origin
        assert np.allclose(transforms[0][:3, 3], [0, 0, 0])
    
    def test_chain_positions(self):
        """Chain of transforms should produce correct positions."""
        # Create transforms that place cameras 1m apart in X
        T_0_to_2 = np.eye(4)
        T_0_to_2[0, 3] = 1.0
        
        T_2_to_4 = np.eye(4)
        T_2_to_4[0, 3] = 1.0
        
        results = [
            PairwiseCalibrationResult(0, 2, T_0_to_2, 0.1, 10),
            PairwiseCalibrationResult(2, 4, T_2_to_4, 0.1, 10),
        ]
        
        transforms = chain_pairwise_transforms(results, reference_camera=0)
        
        # Camera 0 at origin
        assert np.allclose(transforms[0][:3, 3], [0, 0, 0])
        
        # Cameras 2 and 4 should be offset
        # The exact positions depend on transform direction
        assert not np.allclose(transforms[2][:3, 3], [0, 0, 0])
        assert not np.allclose(transforms[4][:3, 3], transforms[2][:3, 3])
    
    def test_reference_camera_selection(self):
        """Reference camera should be at origin."""
        T = np.eye(4)
        T[0, 3] = 1.0
        
        results = [
            PairwiseCalibrationResult(0, 2, T, 0.1, 10),
        ]
        
        # With camera 2 as reference
        transforms = chain_pairwise_transforms(results, reference_camera=2)
        
        assert np.allclose(transforms[2][:3, 3], [0, 0, 0])
        assert not np.allclose(transforms[0][:3, 3], [0, 0, 0])
    
    def test_shortest_path_selection(self):
        """Should use shortest path (lowest error) when multiple paths exist."""
        # Create a triangle with different errors
        T = np.eye(4)
        T[0, 3] = 1.0
        
        results = [
            PairwiseCalibrationResult(0, 2, T, 0.1, 10),  # Low error
            PairwiseCalibrationResult(2, 4, T, 0.1, 10),  # Low error
            PairwiseCalibrationResult(0, 4, T, 10.0, 10),  # High error - should not be used
        ]
        
        transforms = chain_pairwise_transforms(results, reference_camera=0)
        
        # All cameras should be reachable
        assert 0 in transforms
        assert 2 in transforms
        assert 4 in transforms


class TestGetRequiredPairs:
    """Tests for the required pairs function."""
    
    def test_empty_cameras(self):
        """Empty camera list should return empty pairs."""
        pairs = get_required_pairs([])
        assert pairs == []
    
    def test_single_camera(self):
        """Single camera should return no pairs."""
        pairs = get_required_pairs([0])
        assert pairs == []
    
    def test_two_cameras(self):
        """Two cameras should return one pair."""
        pairs = get_required_pairs([0, 2])
        assert len(pairs) == 1
        assert (0, 2) in pairs
    
    def test_four_cameras(self):
        """Four cameras should return 6 pairs (4 choose 2)."""
        pairs = get_required_pairs([0, 2, 4, 6])
        assert len(pairs) == 6
        
        # Check all pairs present
        expected = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 6), (4, 6)]
        for pair in expected:
            assert pair in pairs


class TestCalibrateCameraPair:
    """Tests for the camera pair calibration function."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock CalibrationConfig."""
        from voxelvr.config import CalibrationConfig
        return CalibrationConfig(
            charuco_squares_x=5,
            charuco_squares_y=5,
            charuco_square_length=0.04,
            charuco_marker_length=0.03,
            charuco_dict="DICT_6X6_250",
        )
    
    @pytest.fixture
    def mock_intrinsics(self):
        """Create mock CameraIntrinsics objects."""
        from voxelvr.config import CameraIntrinsics
        from datetime import datetime
        
        intrinsics_a = CameraIntrinsics(
            camera_id=0,
            camera_name="Camera 0",
            resolution=(640, 480),
            camera_matrix=[[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            distortion_coeffs=[0, 0, 0, 0, 0],
            reprojection_error=0.5,
            calibration_date=datetime.now().isoformat(),
        )
        
        intrinsics_b = CameraIntrinsics(
            camera_id=2,
            camera_name="Camera 2",
            resolution=(640, 480),
            camera_matrix=[[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            distortion_coeffs=[0, 0, 0, 0, 0],
            reprojection_error=0.5,
            calibration_date=datetime.now().isoformat(),
        )
        
        return intrinsics_a, intrinsics_b
    
    def test_insufficient_captures(self, mock_config, mock_intrinsics):
        """Should return None with less than 3 valid frames."""
        intrinsics_a, intrinsics_b = mock_intrinsics
        
        # Only 2 captures
        captures = [
            {0: np.zeros((480, 640, 3), dtype=np.uint8),
             2: np.zeros((480, 640, 3), dtype=np.uint8)},
            {0: np.zeros((480, 640, 3), dtype=np.uint8),
             2: np.zeros((480, 640, 3), dtype=np.uint8)},
        ]
        
        result = calibrate_camera_pair(captures, intrinsics_a, intrinsics_b, mock_config)
        
        # Will return None because no charuco board will be detected in blank images
        assert result is None


class TestCalibrationPanelPairwiseIntegration:
    """Tests for CalibrationPanel pairwise logic."""
    
    def test_pairwise_captures_storage(self):
        """Test that pairwise captures can be stored and retrieved correctly."""
        from voxelvr.gui.calibration_panel import CalibrationPanel, CalibrationStep
        
        panel = CalibrationPanel()
        panel.set_cameras([0, 2, 4])
        
        # Verify pairwise_captures dict exists and is empty
        assert hasattr(panel, '_pairwise_captures')
        assert panel._pairwise_captures == {}
        
        # Manually add a pairwise capture (simulating what process_frame_detections does)
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        panel._pairwise_captures[(0, 2)] = [{
            'frames': {0: fake_frame.copy(), 2: fake_frame.copy()},
            'detections': {0: {'success': True}, 2: {'success': True}},
            'timestamp': 1234567890,
        }]
        
        # Verify storage
        assert (0, 2) in panel._pairwise_captures
        assert len(panel._pairwise_captures[(0, 2)]) == 1
        assert panel._pairwise_captures[(0, 2)][0]['frames'][0] is not None
    
    def test_progress_summary_includes_pairwise(self):
        """Test that progress summary includes pairwise data."""
        from voxelvr.gui.calibration_panel import CalibrationPanel
        
        panel = CalibrationPanel()
        panel.set_cameras([0, 2])
        
        progress = panel.get_progress_summary()
        
        assert 'pairwise' in progress
        assert 'pairwise_connected' in progress
        assert 'pairwise_disconnected_cameras' in progress
    
    def test_connectivity_detection(self):
        """Test that connectivity is correctly detected."""
        from voxelvr.gui.calibration_panel import CalibrationPanel, CalibrationStep
        
        panel = CalibrationPanel()
        panel.set_cameras([0, 2, 4])
        
        # Initially not connected
        progress = panel.get_progress_summary()
        assert not progress['pairwise_connected']
        
        # Manually add enough pairwise captures to connect all
        panel._pairwise_captures[(0, 2)] = [{'frames': {}, 'detections': {}} for _ in range(10)]
        panel._pairwise_captures[(2, 4)] = [{'frames': {}, 'detections': {}} for _ in range(10)]
        
        progress = panel.get_progress_summary()
        assert progress['pairwise_connected']
        assert len(progress['pairwise_disconnected_cameras']) == 0
