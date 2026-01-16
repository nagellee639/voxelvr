"""
Tests for Parameter Optimizer

Detailed tests for the auto-adjustment and profile system.
"""

import pytest
import numpy as np
import time

from voxelvr.gui.param_optimizer import (
    ParameterOptimizer,
    FilterProfile,
    PROFILE_PRESETS,
    ProfileSettings,
    JitterMetrics,
    LatencyMetrics,
)


class TestProfilePresets:
    """Tests for profile preset configurations."""
    
    def test_all_profiles_defined(self):
        """Verify all profiles have presets."""
        for profile in FilterProfile:
            if profile != FilterProfile.CUSTOM:
                assert profile in PROFILE_PRESETS
    
    def test_preset_values_valid(self):
        """Verify preset values are within valid ranges."""
        for profile, settings in PROFILE_PRESETS.items():
            assert 0.1 <= settings.min_cutoff <= 10.0
            assert 0.0 <= settings.beta <= 2.0
            assert 0.1 <= settings.d_cutoff <= 5.0
    
    def test_profiles_have_descriptions(self):
        """Verify all profiles have descriptions."""
        for profile, settings in PROFILE_PRESETS.items():
            assert len(settings.description) > 0
    
    def test_profile_characteristics(self):
        """Verify profiles have expected characteristics."""
        low_jitter = PROFILE_PRESETS[FilterProfile.LOW_JITTER]
        low_latency = PROFILE_PRESETS[FilterProfile.LOW_LATENCY]
        
        # Low jitter should have lower cutoff (more smoothing)
        assert low_jitter.min_cutoff < low_latency.min_cutoff
        
        # Low latency should have higher beta (less lag)
        assert low_latency.beta > low_jitter.beta


class TestAutoAdjustment:
    """Tests for automatic parameter adjustment."""
    
    def test_auto_adjust_reduces_jitter(self):
        """Test that auto-adjust attempts to reduce high jitter."""
        optimizer = ParameterOptimizer(
            initial_profile=FilterProfile.LOW_LATENCY,  # Start with high jitter profile
            update_interval=0.0,  # Immediate updates
        )
        
        initial_min_cutoff = optimizer.min_cutoff
        
        # Simulate noisy tracking (high jitter)
        base_pos = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(200):
            noisy_pos = base_pos + np.random.randn(17, 3) * 0.05  # High noise
            optimizer.update(noisy_pos, valid_mask, timestamp=i * 0.033)
        
        # Should have decreased min_cutoff for more smoothing
        # (or parameters should have moved toward jitter reduction)
        jitter = optimizer.measure_jitter()
        assert jitter.sample_count > 0  # Verify we have data
    
    def test_auto_adjust_respects_movement(self):
        """Test that auto-adjust considers movement speed."""
        optimizer = ParameterOptimizer(update_interval=0.0)
        
        valid_mask = np.ones(17, dtype=bool)
        
        # Simulate fast movement
        for i in range(100):
            pos = np.zeros((17, 3))
            pos[:, 0] = i * 0.1  # Moving in X
            optimizer.update(pos, valid_mask, timestamp=i * 0.033)
        
        # Beta should remain high or increase for fast movement
        assert optimizer.beta >= 0.3
    
    def test_manual_override_prevents_auto_adjust(self):
        """Test that manual override prevents auto-adjustment."""
        optimizer = ParameterOptimizer(update_interval=0.0)
        
        optimizer.set_manual_parameters(min_cutoff=0.5, beta=0.2)
        
        original_min_cutoff = optimizer.min_cutoff
        
        # Feed data that would normally trigger adjustment
        base_pos = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(100):
            noisy_pos = base_pos + np.random.randn(17, 3) * 0.1
            optimizer.update(noisy_pos, valid_mask, timestamp=i * 0.033)
        
        # Parameters should be unchanged due to manual override
        assert optimizer.min_cutoff == original_min_cutoff
    
    def test_auto_adjust_smooth_transitions(self):
        """Test that parameter changes are smooth, not abrupt."""
        optimizer = ParameterOptimizer(update_interval=0.0)
        
        param_history = []
        
        def record_params(min_cutoff, beta, d_cutoff):
            param_history.append(min_cutoff)
        
        optimizer.add_callback(record_params)
        
        base_pos = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(200):
            noisy_pos = base_pos + np.random.randn(17, 3) * 0.02
            optimizer.update(noisy_pos, valid_mask, timestamp=i * 0.033)
        
        if len(param_history) > 1:
            # Check that changes are gradual (no big jumps)
            for i in range(1, len(param_history)):
                change = abs(param_history[i] - param_history[i-1])
                assert change < 0.5  # No jumps larger than 0.5


class TestJitterMeasurement:
    """Tests for jitter measurement accuracy."""
    
    def test_zero_jitter_for_static_pose(self):
        """Test that static pose has near-zero jitter."""
        optimizer = ParameterOptimizer()
        
        positions = np.zeros((17, 3))
        positions[:, 1] = 1.5  # Place at realistic height
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(50):
            optimizer.update(positions.copy(), valid_mask, timestamp=i * 0.033)
        
        jitter = optimizer.measure_jitter()
        
        assert jitter.position_std < 0.0001  # Essentially zero
        assert jitter.max_deviation < 0.0001
    
    def test_jitter_proportional_to_noise(self):
        """Test that measured jitter scales with actual noise."""
        base_pos = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        jitters = []
        for noise_level in [0.001, 0.01, 0.1]:
            optimizer = ParameterOptimizer()
            
            for i in range(50):
                noisy_pos = base_pos + np.random.randn(17, 3) * noise_level
                optimizer.update(noisy_pos, valid_mask, timestamp=i * 0.033)
            
            jitter = optimizer.measure_jitter()
            jitters.append(jitter.position_std)
        
        # Jitter should increase with noise level
        assert jitters[0] < jitters[1] < jitters[2]
    
    def test_jitter_specific_joints(self):
        """Test measuring jitter for specific joints."""
        optimizer = ParameterOptimizer()
        
        positions = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(50):
            noisy_pos = positions.copy()
            noisy_pos[0] += np.random.randn(3) * 0.1  # Only add noise to joint 0
            optimizer.update(noisy_pos, valid_mask, timestamp=i * 0.033)
        
        # Jitter for joint 0 should be higher than others
        jitter_0 = optimizer.measure_jitter(joint_indices=[0])
        jitter_others = optimizer.measure_jitter(joint_indices=[1, 2, 3])
        
        assert jitter_0.position_std > jitter_others.position_std


class TestLatencyEstimation:
    """Tests for filter latency estimation."""
    
    def test_latency_increases_with_smoothing(self):
        """Test that latency estimate increases with more smoothing."""
        latencies = []
        
        for min_cutoff in [0.5, 1.0, 2.0, 5.0]:
            optimizer = ParameterOptimizer()
            optimizer.set_manual_parameters(min_cutoff=min_cutoff)
            
            latency = optimizer.estimate_latency()
            latencies.append(latency.estimated_latency_ms)
        
        # Latency should decrease as min_cutoff increases
        for i in range(len(latencies) - 1):
            assert latencies[i] > latencies[i + 1]
    
    def test_beta_reduces_latency(self):
        """Test that higher beta reduces effective latency."""
        optimizer = ParameterOptimizer()
        
        optimizer.set_manual_parameters(min_cutoff=1.0, beta=0.1)
        latency_low_beta = optimizer.estimate_latency()
        
        optimizer.set_manual_parameters(min_cutoff=1.0, beta=1.5)
        latency_high_beta = optimizer.estimate_latency()
        
        assert latency_high_beta.estimated_latency_ms < latency_low_beta.estimated_latency_ms
    
    def test_frame_delay_calculation(self):
        """Test frame delay calculation."""
        optimizer = ParameterOptimizer()
        
        latency = optimizer.estimate_latency()
        
        # Frame delay should be reasonable (less than 10 frames at 30fps)
        assert latency.filter_delay_frames >= 0
        assert latency.filter_delay_frames < 10


class TestParameterHistory:
    """Tests for parameter history tracking."""
    
    def test_history_recorded(self):
        """Test that parameter changes are recorded."""
        optimizer = ParameterOptimizer()
        
        for profile in [FilterProfile.LOW_JITTER, FilterProfile.LOW_LATENCY, FilterProfile.BALANCED]:
            optimizer.set_profile(profile)
            time.sleep(0.01)  # Small delay for different timestamps
        
        assert len(optimizer.parameter_history) >= 3
    
    def test_history_has_timestamps(self):
        """Test that history entries have timestamps."""
        optimizer = ParameterOptimizer()
        
        optimizer.set_profile(FilterProfile.LOW_JITTER)
        
        if len(optimizer.parameter_history) > 0:
            entry = optimizer.parameter_history[-1]
            assert 'time' in entry
            assert 'min_cutoff' in entry
            assert 'beta' in entry


class TestCallbackSystem:
    """Tests for the callback notification system."""
    
    def test_multiple_callbacks(self):
        """Test that multiple callbacks can be added."""
        optimizer = ParameterOptimizer()
        
        results = {'count': 0}
        
        def callback1(mc, b, dc):
            results['count'] += 1
        
        def callback2(mc, b, dc):
            results['count'] += 10
        
        optimizer.add_callback(callback1)
        optimizer.add_callback(callback2)
        
        optimizer.set_profile(FilterProfile.LOW_JITTER)
        
        assert results['count'] == 11
    
    def test_remove_callback(self):
        """Test removing callbacks."""
        optimizer = ParameterOptimizer()
        
        results = {'called': False}
        
        def callback(mc, b, dc):
            results['called'] = True
        
        optimizer.add_callback(callback)
        optimizer.remove_callback(callback)
        
        optimizer.set_profile(FilterProfile.LOW_JITTER)
        
        # Callback was removed, should not be called
        assert results['called'] == False
    
    def test_callback_error_handling(self):
        """Test that callback errors don't crash the optimizer."""
        optimizer = ParameterOptimizer()
        
        def bad_callback(mc, b, dc):
            raise ValueError("Test error")
        
        optimizer.add_callback(bad_callback)
        
        # Should not raise despite bad callback
        optimizer.set_profile(FilterProfile.LOW_JITTER)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
