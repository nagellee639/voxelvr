
import pytest
import numpy as np
from voxelvr.pose.confidence_filter import ConfidenceFilter, ViewConfidenceState
from voxelvr.pose.detector_2d import Keypoints2D

class TestFreezingLogic:
    def create_mock_keypoints(self, camera_id: int, confidence: float) -> Keypoints2D:
        """Create mock keypoints with uniform confidence."""
        num_joints = 17
        positions = np.zeros((num_joints, 2))
        confidences = np.full(num_joints, confidence)
        return Keypoints2D(
            camera_id=camera_id,
            positions=positions,
            confidences=confidences,
            image_width=1280,
            image_height=720,
        )
        
    def warmup_filter(self, f: ConfidenceFilter, camera_id: int):
        """Feed enough confident frames to activate the view."""
        for _ in range(f.reactivation_frames):
             f.update([self.create_mock_keypoints(camera_id, 0.9)])

    def test_strict_exclusion_of_low_confidence(self):
        """
        Verify that a view is excluded IMMEDIATELY upon dropping below threshold.
        It should NOT be used even during the grace period.
        """
        f = ConfidenceFilter(
            confidence_threshold=0.5,
            grace_period_frames=7,
            reactivation_frames=3
        )
        
        # Frame 1: Establish active state
        self.warmup_filter(f, 1)
        
        # Now verify it IS active
        kp = self.create_mock_keypoints(1, 0.9)
        filtered, _ = f.update([kp])
        assert len(filtered) == 1
        
        # Frame 2: Low confidence -> Should be EXCLUDED (even if in grace period)
        kp_low = self.create_mock_keypoints(1, 0.1)
        filtered, _ = f.update([kp_low])
        assert len(filtered) == 0
        
        # Verify internal state is INACTIVE_GRACE
        state = f.view_states[0][1] # Joint 0, Cam 1
        assert state.state == ViewConfidenceState.INACTIVE_GRACE

    def test_grace_period_recovery(self):
        """
        Verify fast recovery within grace period (7 frames).
        """
        f = ConfidenceFilter(grace_period_frames=7, reactivation_frames=3)
        
        # Establish Active
        self.warmup_filter(f, 1)
        
        # Check active
        filtered, _ = f.update([self.create_mock_keypoints(1, 0.9)])
        assert len(filtered) == 1
        
        # Drop for 5 frames (within grace)
        for _ in range(5):
            filtered, _ = f.update([self.create_mock_keypoints(1, 0.1)])
            assert len(filtered) == 0
            
        # Resume -> Should recover IMMEDIATELY
        filtered, _ = f.update([self.create_mock_keypoints(1, 0.9)])
        assert len(filtered) == 1
        # Should not need 3 frames to recover

    def test_long_drop_requires_reactivation(self):
        """
        Verify that after >7 frames of low confidence, we need 3 good frames to recover.
        """
        f = ConfidenceFilter(grace_period_frames=7, reactivation_frames=3)
        
        # Establish Active
        self.warmup_filter(f, 1)
        
        # Drop for 10 frames (exceeds grace)
        for _ in range(10):
            f.update([self.create_mock_keypoints(1, 0.1)])
            
        # Verify state is DEEP inactive
        assert f.view_states[0][1].state == ViewConfidenceState.INACTIVE_DEEP
        
        # Frame 1 of resumption: should still be excluded (needs proof)
        filtered, _ = f.update([self.create_mock_keypoints(1, 0.9)])
        assert len(filtered) == 0
        
        # Frame 2
        filtered, _ = f.update([self.create_mock_keypoints(1, 0.9)])
        assert len(filtered) == 0
        
        # Frame 3 (Reactivation met) -> Active
        filtered, _ = f.update([self.create_mock_keypoints(1, 0.9)])
        assert len(filtered) == 1

    def test_freezing_behavior(self):
        """
        Verify apply_freezing uses last known good position when valid_mask is False.
        """
        f = ConfidenceFilter()
        
        # 1. Provide valid tracking
        pos_good = np.ones((17, 3)) * 1.0
        mask_good = np.ones(17, dtype=bool)
        f.apply_freezing(pos_good, mask_good)
        
        assert np.array_equal(f.last_confident_positions, pos_good)
        
        # 2. Tracking lost (valid_mask False)
        pos_junk = np.zeros((17, 3)) # Input might be zeros or junk
        mask_bad = np.zeros(17, dtype=bool)
        
        result = f.apply_freezing(pos_junk, mask_bad)
        
        # Result should be the GOOD positions, not the junk
        assert np.array_equal(result, pos_good)
        
    def test_per_joint_behavior(self):
        """
        Verify logic works independently per joint.
        """
        f = ConfidenceFilter(reactivation_frames=3)
        self.warmup_filter(f, 1) # Warm up all joints
        
        # Joint 0 confident, Joint 1 not
        kp = self.create_mock_keypoints(1, 0.0)
        kp.confidences[0] = 0.9
        kp.confidences[1] = 0.1
        
        filtered, _ = f.update([kp])
        # The keypoint object is returned because it has at least one active joint
        assert len(filtered) == 1
        
        # Check active states
        # Joint 0 should be active
        assert f.view_states[0][1].is_active() == True
        
        # Joint 1 should have transitioned to INACTIVE_GRACE
        assert f.view_states[1][1].state == ViewConfidenceState.INACTIVE_GRACE
        
    def test_initial_startup_delay(self):
        """
        Verify that we need reactivation_frames to start tracking initially.
        """
        f = ConfidenceFilter(reactivation_frames=3)
        kp = self.create_mock_keypoints(1, 0.9)
        
        # Frame 1
        filtered, _ = f.update([kp])
        assert len(filtered) == 0 # Not yet active
        
        # Frame 2
        filtered, _ = f.update([kp])
        assert len(filtered) == 0
        
        # Frame 3
        filtered, _ = f.update([kp])
        assert len(filtered) == 1 # Now active
