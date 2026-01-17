import pytest
import numpy as np
from voxelvr.pose.confidence_filter import ConfidenceFilter, ViewConfidenceState, create_tpose, TPOSE_3D
from voxelvr.pose.detector_2d import Keypoints2D

class TestConfidenceFilter:
    @pytest.fixture
    def filter(self):
        return ConfidenceFilter(
            num_joints=3,  # Simplified for testing
            confidence_threshold=0.5,
            grace_period_frames=3,
            reactivation_frames=2
        )

    def create_mock_keypoints(self, camera_id, confidences, positions=None):
        if positions is None:
            positions = np.zeros((len(confidences), 2))
        return Keypoints2D(
            positions=positions,
            confidences=np.array(confidences, dtype=np.float32),
            image_width=100,
            image_height=100,
            camera_id=camera_id,
            threshold=0.5
        )

    def test_tpose_initialization(self, filter):
        """Test that system generates T-pose at startup"""
        assert np.array_equal(filter.last_confident_positions, create_tpose())
        assert not filter.has_tracking_history

    def test_basic_filtering(self, filter):
        """Test that unconfident views are filtered out initially"""
        # All views start as INACTIVE_DEEP
        
        # Frame 1: Cam 0 confident, Cam 1 unconfident
        kp0 = self.create_mock_keypoints(0, [0.9, 0.9, 0.9])
        kp1 = self.create_mock_keypoints(1, [0.1, 0.1, 0.1])
        
        # Should not be active yet (need 2 frames for reactivation)
        filtered, _ = filter.update([kp0, kp1])
        assert len(filtered) == 0
        
        # Frame 2: Same
        filtered, _ = filter.update([kp0, kp1])
        
        # Now Cam 0 should be active (2 consecutive confident frames)
        # Cam 1 still inactive
        assert len(filtered) == 1
        assert filtered[0].camera_id == 0

    def test_grace_period_recovery(self, filter):
        """Test instant reactivation within grace period"""
        # Activate Cam 0 first
        kp_good = self.create_mock_keypoints(0, [0.9, 0.9, 0.9])
        kp_bad = self.create_mock_keypoints(0, [0.1, 0.1, 0.1])
        
        filter.update([kp_good])
        filter.update([kp_good])
        
        # Now active. Send bad frame.
        filtered, _ = filter.update([kp_bad])
        
        # Should be inactive now
        assert len(filtered) == 0
        assert filter.view_states[0][0].state == ViewConfidenceState.INACTIVE_GRACE
        
        # Send good frame immediately (within 3 frame grace period)
        filtered, _ = filter.update([kp_good])
        
        # Should be INSTANTLY active again
        assert len(filtered) == 1
        assert filter.view_states[0][0].state == ViewConfidenceState.ACTIVE

    def test_deep_inactivity(self, filter):
        """Test that prolonged inactivity requires proof to reactivate"""
        kp_good = self.create_mock_keypoints(0, [0.9, 0.9, 0.9])
        kp_bad = self.create_mock_keypoints(0, [0.1, 0.1, 0.1])
        
        # Activate
        filter.update([kp_good])
        filter.update([kp_good])
        
        # Go bad for grace_period + 1 frames
        for _ in range(filter.grace_period_frames + 1):
             filter.update([kp_bad])
             
        assert filter.view_states[0][0].state == ViewConfidenceState.INACTIVE_DEEP
        
        # Now good frame - should NOT reactivate instantly
        filtered, _ = filter.update([kp_good])
        assert len(filtered) == 0
        
        # Need one more (reactivation_frames = 2)
        filtered, _ = filter.update([kp_good])
        assert len(filtered) == 1

    def test_joint_freezing(self, filter):
        """Test that unconfident joints are filled with last known position"""
        # Set a known "last position" manually for testing
        test_pos = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ])
        # Only use first 3 joints since we initialized filter with num_joints=3
        test_pos = test_pos[:3]
        
        filter.last_confident_positions[:3] = test_pos
        filter.has_tracking_history = True
        
        # Simulate triangulation result where middle joint is invalid
        measured_pos = np.zeros((3, 3))
        valid_mask = np.array([True, False, True], dtype=bool)
        
        result = filter.apply_freezing(measured_pos, valid_mask)
        
        # Valid joints should be unchanged (0.0)
        assert np.array_equal(result[0], [0.0, 0.0, 0.0])
        assert np.array_equal(result[2], [0.0, 0.0, 0.0])
        
        # Invalid joint should be frozen to last known position
        assert np.array_equal(result[1], [2.0, 2.0, 2.0])
        
        # Internal state should update for valid joints
        assert np.array_equal(filter.last_confident_positions[0], [0.0, 0.0, 0.0])
        assert np.array_equal(filter.last_confident_positions[1], [2.0, 2.0, 2.0])  # Unchanged

    def test_tpose_fallback(self, filter):
        """Test fallback to T-pose if no tracking history"""
        # No history yet
        assert not filter.has_tracking_history
        
        measured_pos = np.zeros((3, 3))
        valid_mask = np.array([False, False, False], dtype=bool)
        
        result = filter.apply_freezing(measured_pos, valid_mask)
        
        # Should be T-pose values
        assert np.array_equal(result, TPOSE_3D[:3])
