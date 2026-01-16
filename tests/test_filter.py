"""
Filter Module Tests

Tests for temporal smoothing filters.
"""

import pytest
import numpy as np
from conftest import generate_walking_pose


class TestOneEuroFilter:
    """Test One-Euro filter implementation."""
    
    def test_filter_initialization(self):
        """Test filter initializes correctly."""
        from voxelvr.pose.filter import OneEuroFilter
        
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        assert f is not None
    
    def test_filter_first_value(self):
        """Test first value passes through unfiltered."""
        from voxelvr.pose.filter import OneEuroFilter
        
        f = OneEuroFilter()
        result = f.filter(5.0)
        
        assert result == 5.0
    
    def test_filter_smooth_signal(self):
        """Test filter smooths noisy signal."""
        from voxelvr.pose.filter import OneEuroFilter
        
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)  # No speed adaptation
        
        # Generate noisy signal
        clean = 10.0
        noisy = [clean + np.random.normal(0, 2) for _ in range(100)]
        
        filtered = [f.filter(v) for v in noisy]
        
        # Filtered signal should have lower variance
        noisy_std = np.std(noisy)
        filtered_std = np.std(filtered[-50:])  # Skip warmup
        
        assert filtered_std < noisy_std, \
            f"Filter didn't reduce noise: {filtered_std} >= {noisy_std}"
    
    def test_filter_follows_fast_changes(self):
        """Test filter follows fast changes with high beta."""
        from voxelvr.pose.filter import OneEuroFilter
        
        f = OneEuroFilter(min_cutoff=1.0, beta=1.0)  # High speed adaptation
        
        # Warmup at 0.0 with timestamps
        for i in range(10):
            f.filter(0.0, t=i * 0.033)  # 30fps timing
        
        # Rapid change - filter should follow quickly with high beta
        results = []
        for i in range(20):
            val = f.filter(10.0, t=(10 + i) * 0.033)
            results.append(val)
        
        # Should approach 10.0 after several frames
        assert results[-1] > 8.0, f"Filter too slow: {results[-1]}"


class TestPoseFilter:
    """Test full pose temporal filter."""
    
    def test_pose_filter_initialization(self):
        """Test pose filter initializes for all joints."""
        from voxelvr.pose.filter import PoseFilter
        
        f = PoseFilter(num_joints=17)
        
        assert len(f.joint_filters) == 17
    
    def test_pose_filter_shape(self):
        """Test pose filter maintains shape."""
        from voxelvr.pose.filter import PoseFilter
        
        f = PoseFilter(num_joints=17)
        
        pose = np.random.rand(17, 3)
        valid_mask = np.ones(17, dtype=bool)
        
        result = f.filter(pose, valid_mask)
        
        assert result.shape == (17, 3)
    
    def test_pose_filter_with_invalid_joints(self):
        """Test filter handles invalid joints."""
        from voxelvr.pose.filter import PoseFilter
        
        f = PoseFilter(num_joints=17)
        
        pose = np.random.rand(17, 3)
        valid_mask = np.zeros(17, dtype=bool)
        valid_mask[0:5] = True  # Only first 5 joints valid
        
        result = f.filter(pose, valid_mask)
        
        # Invalid joints should be unchanged
        np.testing.assert_array_equal(result[5:], pose[5:])
    
    def test_pose_filter_reduces_jitter(self):
        """Test filter reduces jitter in walking animation."""
        from voxelvr.pose.filter import PoseFilter
        
        f = PoseFilter(num_joints=17)
        
        # Generate walking sequence with added noise
        poses = []
        filtered_poses = []
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(100):
            t = i * 0.033
            pose = generate_walking_pose(t)
            
            # Add noise
            noisy_pose = pose + np.random.normal(0, 0.01, pose.shape)
            poses.append(noisy_pose)
            
            filtered = f.filter(noisy_pose, valid_mask)
            filtered_poses.append(filtered)
        
        # Calculate frame-to-frame jitter
        def calculate_jitter(pose_list):
            jitters = []
            for i in range(1, len(pose_list)):
                diff = np.linalg.norm(pose_list[i] - pose_list[i-1], axis=1)
                jitters.append(np.mean(diff))
            return np.std(jitters)
        
        original_jitter = calculate_jitter(poses)
        filtered_jitter = calculate_jitter(filtered_poses[-50:])  # Skip warmup
        
        print(f"\nJitter reduction:")
        print(f"  Original: {original_jitter:.4f}")
        print(f"  Filtered: {filtered_jitter:.4f}")
        print(f"  Reduction: {(1 - filtered_jitter/original_jitter)*100:.1f}%")
        
        assert filtered_jitter < original_jitter


class TestExponentialMovingAverage:
    """Test EMA filter."""
    
    def test_ema_initialization(self):
        """Test EMA filter initializes."""
        from voxelvr.pose.filter import ExponentialMovingAverage
        
        ema = ExponentialMovingAverage(alpha=0.3)
        assert ema is not None
    
    def test_ema_smoothing(self):
        """Test EMA smooths data."""
        from voxelvr.pose.filter import ExponentialMovingAverage
        
        ema = ExponentialMovingAverage(alpha=0.3)
        
        # Apply to sequence
        values = [10.0] * 5 + [20.0] * 5
        results = [ema.filter(v) for v in values]
        
        # Should gradually transition from 10 to 20
        assert results[0] == 10.0  # First value unchanged
        assert results[5] < 20.0   # Not immediately at 20
        assert results[-1] > 18.0  # Close to 20 by end
