"""
Tests for VoxelVR GUI Components

Tests the GUI logic components without requiring actual display.
"""

import pytest
import numpy as np
import time
from typing import List

# Import GUI components
from voxelvr.gui.param_optimizer import (
    ParameterOptimizer, 
    FilterProfile, 
    PROFILE_PRESETS,
    ProfileSettings,
    JitterMetrics,
    LatencyMetrics,
)
from voxelvr.gui.osc_status import (
    OSCStatusIndicator, 
    ConnectionState,
    OSCStats,
)
from voxelvr.gui.camera_panel import (
    CameraPanel, 
    CameraFeedInfo,
)
from voxelvr.gui.calibration_panel import (
    CalibrationPanel, 
    CalibrationStep,
    CalibrationState,
    CameraCalibrationStatus,
)
from voxelvr.gui.tracking_panel import (
    TrackingPanel, 
    TrackingState,
    TrackerConfig,
    TrackingStatus,
)
from voxelvr.gui.performance_panel import (
    PerformancePanel, 
    PerformanceMetrics,
)
from voxelvr.gui.debug_panel import (
    DebugPanel, 
    DebugMetrics,
)


# ============================================================================
# Parameter Optimizer Tests
# ============================================================================

class TestParameterOptimizer:
    """Tests for the parameter optimization system."""
    
    def test_initialization(self):
        """Test optimizer initializes with correct defaults."""
        optimizer = ParameterOptimizer()
        
        assert optimizer.current_profile == FilterProfile.BALANCED
        assert optimizer.auto_adjust_enabled == True
        assert optimizer.manual_override == False
        
        # Check default parameters match balanced profile
        balanced = PROFILE_PRESETS[FilterProfile.BALANCED]
        assert optimizer.min_cutoff == balanced.min_cutoff
        assert optimizer.beta == balanced.beta
    
    def test_profile_switching(self):
        """Test switching between profiles."""
        optimizer = ParameterOptimizer()
        
        for profile in FilterProfile:
            if profile == FilterProfile.CUSTOM:
                continue
            
            optimizer.set_profile(profile)
            settings = PROFILE_PRESETS[profile]
            
            assert optimizer.current_profile == profile
            assert optimizer.min_cutoff == settings.min_cutoff
            assert optimizer.beta == settings.beta
    
    def test_manual_parameters(self):
        """Test setting manual parameters."""
        optimizer = ParameterOptimizer()
        
        optimizer.set_manual_parameters(min_cutoff=0.5, beta=1.0, d_cutoff=0.8)
        
        assert optimizer.manual_override == True
        assert optimizer.current_profile == FilterProfile.CUSTOM
        assert optimizer.min_cutoff == 0.5
        assert optimizer.beta == 1.0
        assert optimizer.d_cutoff == 0.8
    
    def test_parameter_clamping(self):
        """Test that parameters are clamped to valid ranges."""
        optimizer = ParameterOptimizer()
        
        # Set out-of-range values
        optimizer.set_manual_parameters(min_cutoff=-1.0, beta=5.0, d_cutoff=10.0)
        
        assert optimizer.min_cutoff >= 0.1
        assert optimizer.beta <= 2.0
        assert optimizer.d_cutoff <= 5.0
    
    def test_callback_notification(self):
        """Test callback notification on parameter change."""
        optimizer = ParameterOptimizer()
        
        received_params = []
        
        def callback(min_cutoff, beta, d_cutoff):
            received_params.append((min_cutoff, beta, d_cutoff))
        
        optimizer.add_callback(callback)
        optimizer.set_profile(FilterProfile.LOW_JITTER)
        
        assert len(received_params) == 1
        assert received_params[0][0] == PROFILE_PRESETS[FilterProfile.LOW_JITTER].min_cutoff
    
    def test_auto_adjust(self):
        """Test auto-adjustment with pose updates."""
        optimizer = ParameterOptimizer(update_interval=0.0)  # Immediate updates
        
        # Generate synthetic pose data
        positions = np.random.randn(17, 3) * 0.01  # Small random positions
        valid_mask = np.ones(17, dtype=bool)
        
        initial_min_cutoff = optimizer.min_cutoff
        
        # Feed many updates to trigger adjustment
        for i in range(100):
            optimizer.update(positions + np.random.randn(17, 3) * 0.001, valid_mask)
        
        # Parameters should have changed due to auto-adjustment
        # (or stayed the same if within targets - just verify no crash)
        assert optimizer.min_cutoff >= 0.1
    
    def test_jitter_measurement(self):
        """Test jitter measurement from position history."""
        optimizer = ParameterOptimizer()
        
        positions = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        # Feed identical positions (no jitter)
        for i in range(30):
            optimizer.update(positions.copy(), valid_mask, timestamp=i * 0.033)
        
        jitter = optimizer.measure_jitter()
        assert jitter.position_std < 0.001  # Very low jitter
    
    def test_jitter_measurement_with_noise(self):
        """Test jitter measurement detects noise."""
        optimizer = ParameterOptimizer()
        
        base_positions = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        # Feed noisy positions
        for i in range(30):
            noisy = base_positions + np.random.randn(17, 3) * 0.01
            optimizer.update(noisy, valid_mask, timestamp=i * 0.033)
        
        jitter = optimizer.measure_jitter()
        assert jitter.position_std > 0.005  # Significant jitter
    
    def test_latency_estimation(self):
        """Test latency estimation based on parameters."""
        optimizer = ParameterOptimizer()
        
        # Low cutoff = high latency
        optimizer.set_profile(FilterProfile.LOW_JITTER)
        latency_low = optimizer.estimate_latency()
        
        # High cutoff = low latency
        optimizer.set_profile(FilterProfile.LOW_LATENCY)
        latency_high = optimizer.estimate_latency()
        
        assert latency_low.estimated_latency_ms > latency_high.estimated_latency_ms
    
    def test_manual_override_disable(self):
        """Test disabling manual override returns to auto."""
        optimizer = ParameterOptimizer()
        
        optimizer.set_manual_parameters(min_cutoff=0.5)
        assert optimizer.manual_override == True
        
        optimizer.disable_manual_override()
        assert optimizer.manual_override == False
        assert optimizer.auto_adjust_enabled == True
    
    def test_history_reset(self):
        """Test resetting position history."""
        optimizer = ParameterOptimizer()
        
        positions = np.random.randn(17, 3)
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(10):
            optimizer.update(positions, valid_mask)
        
        optimizer.reset_history()
        
        jitter = optimizer.measure_jitter()
        assert jitter.sample_count == 0


# ============================================================================
# OSC Status Tests
# ============================================================================

class TestOSCStatus:
    """Tests for OSC connection status indicator."""
    
    def test_initialization(self):
        """Test status initializes as disconnected."""
        status = OSCStatusIndicator()
        
        assert status.state == ConnectionState.DISCONNECTED
    
    def test_message_send_updates_state(self):
        """Test sending message updates connection state."""
        status = OSCStatusIndicator()
        
        status.on_message_sent(100)
        
        assert status.state == ConnectionState.CONNECTED
    
    def test_state_callbacks(self):
        """Test state change callbacks."""
        status = OSCStatusIndicator()
        
        states_received = []
        
        def callback(state):
            states_received.append(state)
        
        status.add_state_callback(callback)
        
        status.on_connect()
        status.on_message_sent()
        status.on_disconnect()
        
        assert ConnectionState.CONNECTING in states_received
        assert ConnectionState.CONNECTED in states_received
        assert ConnectionState.DISCONNECTED in states_received
    
    def test_stats_tracking(self):
        """Test message statistics tracking."""
        status = OSCStatusIndicator()
        
        for i in range(10):
            status.on_message_sent(50)
        
        stats = status.get_stats()
        
        assert stats.messages_sent == 10
        assert stats.bytes_sent == 500
    
    def test_error_handling(self):
        """Test error state handling."""
        status = OSCStatusIndicator()
        
        status.on_error("Test error")
        
        assert status.state == ConnectionState.ERROR
        assert status.get_stats().errors == 1
    
    def test_activity_timeout(self):
        """Test that inactivity causes disconnect."""
        status = OSCStatusIndicator(activity_timeout=0.1)
        
        status.on_message_sent()
        assert status.state == ConnectionState.CONNECTED
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should now be disconnected
        assert status.state == ConnectionState.DISCONNECTED
    
    def test_state_colors(self):
        """Test state color mapping."""
        status = OSCStatusIndicator()
        
        status.on_message_sent()
        color = status.get_state_color()
        assert color == (0, 200, 100)  # Green for connected
        
        status.on_error()
        color = status.get_state_color()
        assert color == (255, 80, 80)  # Red for error


# ============================================================================
# Camera Panel Tests
# ============================================================================

class TestCameraPanel:
    """Tests for camera preview panel."""
    
    def test_add_camera(self):
        """Test adding cameras."""
        panel = CameraPanel()
        
        panel.add_camera(0, "Front")
        panel.add_camera(1, "Left")
        panel.add_camera(2, "Right")
        
        assert len(panel.get_camera_ids()) == 3
        assert 0 in panel.get_camera_ids()
    
    def test_grid_layout_calculation(self):
        """Test grid layout updates correctly."""
        panel = CameraPanel(max_columns=3)
        
        # 1 camera = 1x1
        panel.add_camera(0)
        assert panel.get_grid_size() == (1, 1)
        
        # 2 cameras = 2x1
        panel.add_camera(1)
        assert panel.get_grid_size() == (2, 1)
        
        # 4 cameras = 3x2
        panel.add_camera(2)
        panel.add_camera(3)
        assert panel.get_grid_size() == (3, 2)
    
    def test_frame_update(self):
        """Test updating camera frames."""
        panel = CameraPanel()
        panel.add_camera(0)
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (100, 150, 200)
        
        panel.update_frame(0, frame)
        
        info = panel.get_camera_info(0)
        assert info.is_connected == True
        assert info.frame_count == 1
    
    def test_remove_camera(self):
        """Test removing cameras."""
        panel = CameraPanel()
        
        panel.add_camera(0)
        panel.add_camera(1)
        
        panel.remove_camera(0)
        
        assert len(panel.get_camera_ids()) == 1
        assert 0 not in panel.get_camera_ids()
    
    def test_fps_calculation(self):
        """Test FPS calculation from frame updates."""
        panel = CameraPanel()
        panel.add_camera(0)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Simulate 30fps updates
        for i in range(30):
            panel.update_frame(0, frame, timestamp=i * 0.033)
        
        info = panel.get_camera_info(0)
        assert info.fps > 25 and info.fps < 35  # Approximately 30fps
    
    def test_expand_toggle(self):
        """Test camera expand toggle."""
        panel = CameraPanel()
        panel.add_camera(0)
        panel.add_camera(1)
        
        assert panel.expanded_camera is None
        
        panel.toggle_expand(0)
        assert panel.expanded_camera == 0
        
        panel.toggle_expand(0)
        assert panel.expanded_camera is None


# ============================================================================
# Calibration Panel Tests
# ============================================================================

class TestCalibrationPanel:
    """Tests for calibration wizard."""
    
    def test_initialization(self):
        """Test calibration starts in idle state."""
        panel = CalibrationPanel()
        
        assert panel.current_step == CalibrationStep.IDLE
        assert panel.state.is_running == False
    
    def test_set_cameras(self):
        """Test setting cameras for calibration."""
        panel = CalibrationPanel()
        
        panel.set_cameras([0, 1, 2])
        
        assert len(panel.state.cameras) == 3
        assert 0 in panel.state.cameras
    
    def test_start_calibration(self):
        """Test starting calibration."""
        panel = CalibrationPanel()
        panel.set_cameras([0, 1])
        
        panel.start_calibration()
        
        assert panel.current_step == CalibrationStep.EXPORT_BOARD
        assert panel.state.is_running == True
    
    def test_step_progression(self):
        """Test progressing through calibration steps."""
        panel = CalibrationPanel(intrinsic_frames_required=5, extrinsic_frames_required=5)
        panel.set_cameras([0])
        
        # Start
        panel.start_calibration()
        assert panel.current_step == CalibrationStep.EXPORT_BOARD
        
        # Begin: to calibration phase
        panel.begin_calibration()
        assert panel.current_step == CalibrationStep.CALIBRATION
        
        # Finish
        panel.finish_calibration()
        assert panel.current_step == CalibrationStep.COMPLETE
    
    def test_calibration_cancel(self):
        """Test canceling calibration."""
        panel = CalibrationPanel()
        panel.set_cameras([0, 1])
        
        panel.start_calibration()
        panel.cancel_calibration()
        
        assert panel.current_step == CalibrationStep.IDLE
        assert panel.state.is_running == False
    
    def test_board_detection_update(self):
        """Test board detection status update."""
        panel = CalibrationPanel()
        panel.set_cameras([0, 1])
        panel.start_calibration()
        
        panel.update_board_detection(0, True)
        
        assert panel.state.cameras[0].board_visible == True
        assert panel.state.all_cameras_visible == False  # Camera 1 not visible yet
        
        panel.update_board_detection(1, True)
        
        assert panel.state.all_cameras_visible == True
    
    def test_step_callbacks(self):
        """Test step change callbacks."""
        panel = CalibrationPanel()
        panel.set_cameras([0])
        
        steps_received = []
        
        def callback(step):
            steps_received.append(step)
        
        panel.add_step_callback(callback)
        
        panel.start_calibration()
        panel.cancel_calibration()
        
        assert CalibrationStep.EXPORT_BOARD in steps_received
        assert CalibrationStep.IDLE in steps_received
    
    def test_get_instructions(self):
        """Test getting step instructions."""
        panel = CalibrationPanel()
        panel.set_cameras([0])
        
        instructions = panel.get_step_instructions()
        assert "Start Calibration" in instructions or "Click" in instructions
        
        panel.start_calibration()
        instructions = panel.get_step_instructions()
        assert "Export" in instructions or "Print" in instructions


# ============================================================================
# Tracking Panel Tests
# ============================================================================

class TestTrackingPanel:
    """Tests for tracking controls panel."""
    
    def test_initialization(self):
        """Test tracking starts stopped."""
        panel = TrackingPanel()
        
        assert panel.is_running == False
        assert panel.status.state == TrackingState.STOPPED
    
    def test_osc_config(self):
        """Test OSC configuration."""
        panel = TrackingPanel(osc_ip="192.168.1.100", osc_port=9001)
        
        ip, port = panel.get_osc_config()
        assert ip == "192.168.1.100"
        assert port == 9001
        
        panel.set_osc_config("localhost", 9000)
        ip, port = panel.get_osc_config()
        assert ip == "localhost"
    
    def test_tracker_toggle(self):
        """Test enabling/disabling trackers."""
        panel = TrackingPanel()
        
        assert panel.get_tracker_enabled("hip") == True  # Default enabled
        
        panel.set_tracker_enabled("hip", False)
        assert panel.get_tracker_enabled("hip") == False
        
        enabled = panel.get_enabled_trackers()
        assert "hip" not in enabled
    
    def test_state_transitions(self):
        """Test tracking state transitions."""
        panel = TrackingPanel()
        
        panel.request_start()
        assert panel.status.state == TrackingState.STARTING
        
        panel.on_tracking_started()
        assert panel.status.state == TrackingState.RUNNING
        assert panel.is_running == True
        
        panel.request_stop()
        assert panel.status.state == TrackingState.STOPPING
        
        panel.on_tracking_stopped()
        assert panel.status.state == TrackingState.STOPPED
    
    def test_error_handling(self):
        """Test error state."""
        panel = TrackingPanel()
        
        panel.on_tracking_error("Test error")
        
        assert panel.status.state == TrackingState.ERROR
        assert "Test error" in panel.status.error_message
    
    def test_pose_update(self):
        """Test updating pose data."""
        panel = TrackingPanel()
        
        positions = np.random.randn(17, 3)
        valid_mask = np.ones(17, dtype=bool)
        valid_mask[0] = False
        
        panel.update_pose(positions, valid_mask)
        
        pose = panel.get_current_pose()
        assert pose is not None
        assert pose['valid'][0] == False
    
    def test_joint_info(self):
        """Test getting joint information."""
        panel = TrackingPanel()
        
        positions = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        confidences = np.full(17, 0.9)
        
        panel.update_pose(positions, valid_mask, confidences)
        
        joints = panel.get_joint_info()
        assert len(joints) == 17
        assert joints[0]['name'] == "Nose"
        assert joints[0]['confidence'] == 0.9


# ============================================================================
# Performance Panel Tests
# ============================================================================

class TestPerformancePanel:
    """Tests for performance monitoring panel."""
    
    def test_initialization(self):
        """Test initial metrics are zero."""
        panel = PerformancePanel()
        
        metrics = panel.current_metrics
        assert metrics.total_fps == 0.0
    
    def test_metrics_update(self):
        """Test updating metrics."""
        panel = PerformancePanel()
        
        metrics = PerformanceMetrics(
            total_fps=30.0,
            total_latency_ms=33.3,
            num_valid_joints=15,
        )
        
        panel.update(metrics)
        
        assert panel.current_metrics.total_fps == 30.0
    
    def test_history_tracking(self):
        """Test that history is tracked."""
        panel = PerformancePanel()
        
        for i in range(10):
            panel.update(PerformanceMetrics(total_fps=25.0 + i))
        
        history = panel.get_fps_history()
        assert len(history['total']) == 10
    
    def test_average_fps(self):
        """Test average FPS calculation."""
        panel = PerformancePanel()
        
        for i in range(30):
            panel.update(PerformanceMetrics(total_fps=30.0))
            time.sleep(0.01)  # Small delay so times are different
        
        avg = panel.get_average_fps(window_seconds=1.0)
        assert avg > 25 and avg < 35
    
    def test_statistics(self):
        """Test statistics calculation."""
        panel = PerformancePanel()
        
        for i in range(50):
            panel.update(PerformanceMetrics(total_fps=30.0 + np.random.randn()))
        
        stats = panel.get_statistics()
        assert 'fps' in stats
        assert 'mean' in stats['fps']
        assert stats['fps']['mean'] > 25
    
    def test_timing_histogram(self):
        """Test frame timing histogram."""
        panel = PerformancePanel()
        
        for latency in [10, 20, 30, 40, 50]:
            panel.update(PerformanceMetrics(total_latency_ms=latency))
        
        histogram = panel.get_timing_histogram()
        assert len(histogram) > 0
    
    def test_reset(self):
        """Test resetting statistics."""
        panel = PerformancePanel()
        
        panel.update(PerformanceMetrics(total_fps=30.0))
        panel.reset()
        
        assert panel.get_total_frames() == 0


# ============================================================================
# Debug Panel Tests
# ============================================================================

class TestDebugPanel:
    """Tests for debug and tuning panel."""
    
    def test_initialization(self):
        """Test debug panel initializes with optimizer."""
        panel = DebugPanel()
        
        assert panel.optimizer is not None
        assert panel.metrics.current_min_cutoff == PROFILE_PRESETS[FilterProfile.BALANCED].min_cutoff
    
    def test_profile_change(self):
        """Test changing profiles through debug panel."""
        panel = DebugPanel()
        
        panel.set_profile("low_jitter")
        
        assert panel.metrics.current_profile == "low_jitter"
        assert panel.optimizer.current_profile == FilterProfile.LOW_JITTER
    
    def test_slider_values(self):
        """Test setting individual parameters."""
        panel = DebugPanel()
        
        panel.set_min_cutoff(0.5)
        panel.set_beta(1.0)
        
        values = panel.get_current_values()
        assert values['min_cutoff'] == 0.5
        assert values['beta'] == 1.0
    
    def test_auto_adjust_toggle(self):
        """Test toggling auto-adjustment."""
        panel = DebugPanel()
        
        panel.set_auto_adjust(False)
        assert panel.metrics.auto_adjust_enabled == False
        
        panel.set_auto_adjust(True)
        assert panel.metrics.auto_adjust_enabled == True
    
    def test_manual_override(self):
        """Test manual override."""
        panel = DebugPanel()
        
        panel.enable_manual_override()
        assert panel.metrics.manual_override == True
        
        panel.disable_manual_override()
        assert panel.metrics.manual_override == False
    
    def test_pose_update_measures_jitter(self):
        """Test that pose updates trigger jitter measurement."""
        panel = DebugPanel()
        
        positions = np.random.randn(17, 3)
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(30):
            panel.update(positions + np.random.randn(17, 3) * 0.01, valid_mask, timestamp=i * 0.033)
        
        assert panel.metrics.jitter_position_mm > 0
    
    def test_slider_ranges(self):
        """Test getting slider ranges."""
        panel = DebugPanel()
        
        ranges = panel.get_slider_ranges()
        
        assert 'min_cutoff' in ranges
        assert ranges['min_cutoff'][0] < ranges['min_cutoff'][1]  # min < max
    
    def test_reset_defaults(self):
        """Test resetting to defaults."""
        panel = DebugPanel()
        
        panel.set_profile("low_jitter")
        panel.reset_to_defaults()
        
        assert panel.metrics.current_profile == "balanced" or panel.optimizer.current_profile == FilterProfile.BALANCED
    
    def test_graph_data(self):
        """Test getting graph data."""
        panel = DebugPanel()
        
        positions = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        for i in range(50):
            panel.update(positions + np.random.randn(17, 3) * 0.01, valid_mask, timestamp=i * 0.033)
        
        times, values = panel.get_graph_data('jitter')
        assert len(times) == len(values)
        assert len(times) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestGUIIntegration:
    """Integration tests for GUI components working together."""
    
    def test_optimizer_debug_panel_sync(self):
        """Test optimizer and debug panel stay in sync."""
        optimizer = ParameterOptimizer()
        panel = DebugPanel(optimizer=optimizer)
        
        # Change via optimizer
        optimizer.set_profile(FilterProfile.LOW_LATENCY)
        
        # Should be reflected in panel
        metrics = panel.metrics
        assert metrics.current_min_cutoff == PROFILE_PRESETS[FilterProfile.LOW_LATENCY].min_cutoff
    
    def test_tracking_performance_sync(self):
        """Test tracking and performance panels work together."""
        tracking = TrackingPanel()
        perf = PerformancePanel()
        
        # Simulate tracking updates
        tracking.request_start()
        tracking.on_tracking_started()
        
        for i in range(10):
            tracking.update_status(fps=30.0, valid_joints=15)
            perf.update(PerformanceMetrics(
                total_fps=30.0,
                num_valid_joints=15,
            ))
        
        assert tracking.status.fps == 30.0
        assert perf.get_average_fps() > 25.0
    
    def test_calibration_camera_sync(self):
        """Test calibration and camera panels work together."""
        camera = CameraPanel()
        calib = CalibrationPanel()
        
        # Add cameras to both
        for i in range(3):
            camera.add_camera(i)
        
        calib.set_cameras([0, 1, 2])
        
        # Verify sync
        assert len(camera.get_camera_ids()) == 3
        assert len(calib.state.cameras) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
