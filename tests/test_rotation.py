"""
Rotation Estimation Tests

Tests the tracker rotation estimation logic.
"""

import pytest
import numpy as np
from conftest import generate_t_pose, generate_walking_pose


class TestRotationBasics:
    """Basic rotation estimation tests."""
    
    def test_normalize(self):
        """Test vector normalization."""
        from voxelvr.pose.rotation import normalize
        
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        
        assert np.isclose(np.linalg.norm(result), 1.0)
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0])
    
    def test_normalize_zero_vector(self):
        """Test normalization of zero vector."""
        from voxelvr.pose.rotation import normalize
        
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])
    
    def test_safe_cross(self):
        """Test safe cross product."""
        from voxelvr.pose.rotation import safe_cross
        
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        
        result = safe_cross(a, b)
        expected = np.array([0.0, 0.0, 1.0])
        
        np.testing.assert_allclose(result, expected, atol=1e-6)
    
    def test_build_rotation_matrix(self):
        """Test rotation matrix construction."""
        from voxelvr.pose.rotation import build_rotation_matrix
        
        forward = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 1.0, 0.0])
        
        R = build_rotation_matrix(forward, up)
        
        # Should be identity rotation for these inputs
        assert R.shape == (3, 3)
        
        # Should be orthonormal
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


class TestTrackerRotations:
    """Test individual tracker rotation estimation."""
    
    def test_hip_rotation_t_pose(self):
        """Test hip rotation for T-pose (should face forward)."""
        from voxelvr.pose.rotation import estimate_hip_rotation
        
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        rotation = estimate_hip_rotation(pose, valid_mask)
        
        assert rotation is not None
        assert len(rotation) == 3
        
        # T-pose facing forward should have small rotations
        assert all(abs(r) < 45 for r in rotation), f"Unexpected rotation: {rotation}"
    
    def test_chest_rotation_t_pose(self):
        """Test chest rotation for T-pose."""
        from voxelvr.pose.rotation import estimate_chest_rotation
        
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        rotation = estimate_chest_rotation(pose, valid_mask)
        
        assert rotation is not None
        assert len(rotation) == 3
    
    def test_foot_rotation(self):
        """Test foot rotation estimation."""
        from voxelvr.pose.rotation import estimate_foot_rotation
        
        pose = generate_t_pose()
        
        # Left foot
        left_rotation = estimate_foot_rotation(
            pose[15],  # left_ankle
            pose[13],  # left_knee
            is_left=True,
        )
        
        # Right foot
        right_rotation = estimate_foot_rotation(
            pose[16],  # right_ankle
            pose[14],  # right_knee
            is_left=False,
        )
        
        assert left_rotation is not None
        assert right_rotation is not None
        
        # For symmetric T-pose, rotations should be similar (mirrored)
        # but not identical due to left/right handling
    
    def test_knee_rotation(self):
        """Test knee rotation estimation."""
        from voxelvr.pose.rotation import estimate_knee_rotation
        
        pose = generate_t_pose()
        
        left_rotation = estimate_knee_rotation(
            pose[11],  # left_hip
            pose[13],  # left_knee
            pose[15],  # left_ankle
            is_left=True,
        )
        
        assert left_rotation is not None
    
    def test_elbow_rotation(self):
        """Test elbow rotation estimation."""
        from voxelvr.pose.rotation import estimate_elbow_rotation
        
        pose = generate_t_pose()
        
        left_rotation = estimate_elbow_rotation(
            pose[5],   # left_shoulder
            pose[7],   # left_elbow
            pose[9],   # left_wrist
            is_left=True,
        )
        
        assert left_rotation is not None


class TestFullRotationEstimation:
    """Test complete rotation estimation for all trackers."""
    
    def test_estimate_all_rotations(self):
        """Test estimating all tracker rotations."""
        from voxelvr.pose.rotation import estimate_all_rotations
        
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        rotations = estimate_all_rotations(pose, valid_mask)
        
        assert rotations is not None
        assert rotations.hip is not None
        assert rotations.chest is not None
        assert rotations.left_foot is not None
        assert rotations.right_foot is not None
        assert rotations.left_knee is not None
        assert rotations.right_knee is not None
        assert rotations.left_elbow is not None
        assert rotations.right_elbow is not None
    
    def test_rotations_to_dict(self):
        """Test converting rotations to dictionary format."""
        from voxelvr.pose.rotation import estimate_all_rotations
        
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        rotations = estimate_all_rotations(pose, valid_mask)
        rotation_dict = rotations.to_dict()
        
        assert 'hip' in rotation_dict
        assert 'chest' in rotation_dict
        assert 'left_foot' in rotation_dict
        assert 'right_foot' in rotation_dict
        
        # Each should be a tuple of 3 floats
        for name, rot in rotation_dict.items():
            assert len(rot) == 3
            assert all(isinstance(r, (int, float, np.floating)) for r in rot)
    
    def test_rotations_with_missing_joints(self):
        """Test rotation estimation with missing joints."""
        from voxelvr.pose.rotation import estimate_all_rotations
        
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        # Invalidate left leg
        valid_mask[11] = False  # left_hip
        valid_mask[13] = False  # left_knee
        valid_mask[15] = False  # left_ankle
        
        rotations = estimate_all_rotations(pose, valid_mask)
        
        # Left leg rotations should be None
        assert rotations.left_knee is None
        assert rotations.left_foot is None
        
        # Right leg should still work
        assert rotations.right_knee is not None
        assert rotations.right_foot is not None


class TestRotationFilter:
    """Test rotation temporal filtering."""
    
    def test_rotation_filter_basic(self):
        """Test basic rotation filtering."""
        from voxelvr.pose.rotation import RotationFilter, TrackerRotations
        
        filter = RotationFilter(alpha=0.3)
        
        # First frame
        rot1 = TrackerRotations(
            hip=np.array([0.0, 0.0, 0.0]),
            chest=np.array([0.0, 0.0, 0.0]),
        )
        
        filtered1 = filter.filter(rot1)
        np.testing.assert_allclose(filtered1.hip, [0.0, 0.0, 0.0], atol=1e-6)
        
        # Second frame with small change
        rot2 = TrackerRotations(
            hip=np.array([10.0, 0.0, 0.0]),
            chest=np.array([5.0, 0.0, 0.0]),
        )
        
        filtered2 = filter.filter(rot2)
        
        # Should be smoothed (not full 10 degrees)
        assert filtered2.hip[0] < 10.0
        assert filtered2.hip[0] > 0.0
    
    def test_rotation_filter_wraparound(self):
        """Test filter handles angle wraparound correctly."""
        from voxelvr.pose.rotation import RotationFilter, TrackerRotations
        
        filter = RotationFilter(alpha=0.5)
        
        # Start at 170 degrees
        rot1 = TrackerRotations(hip=np.array([170.0, 0.0, 0.0]))
        filter.filter(rot1)
        
        # Jump to -170 degrees (should be treated as +20 degree change, not -340)
        rot2 = TrackerRotations(hip=np.array([-170.0, 0.0, 0.0]))
        filtered = filter.filter(rot2)
        
        # Result should be near 180 or -180, not near 0
        assert abs(abs(filtered.hip[0]) - 180) < 30, \
            f"Wraparound handling failed: {filtered.hip[0]}"
