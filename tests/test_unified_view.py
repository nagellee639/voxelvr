"""
Tests for Unified View Module

Tests unified view logic, state management, and callbacks.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List

from voxelvr.gui.unified_view import (
    UnifiedView,
    UnifiedViewState,
    CalibrationMode,
    TrackingMode,
    CameraFeedInfo,
    CalibrationStatus,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def unified_view() -> UnifiedView:
    """Create a UnifiedView instance."""
    return UnifiedView()


@pytest.fixture
def view_with_cameras(unified_view) -> UnifiedView:
    """Create a UnifiedView with cameras set up."""
    unified_view.set_cameras([0, 1, 2, 3])
    return unified_view


# ============================================================================
# Initialization Tests
# ============================================================================

class TestInitialization:
    """Tests for view initialization."""
    
    def test_default_state(self, unified_view):
        """Test default state after initialization."""
        state = unified_view.state
        
        assert len(state.cameras) == 0
        assert state.calibration.mode == CalibrationMode.CHARUCO
        assert state.tracking_mode == TrackingMode.STOPPED
        assert state.apriltags_enabled is False
        assert state.osc_connected is False
        assert state.debug_window_open is False
    
    def test_custom_osc_config(self):
        """Test custom OSC configuration."""
        view = UnifiedView(osc_ip="192.168.1.100", osc_port=9001)
        
        assert view.state.osc_ip == "192.168.1.100"
        assert view.state.osc_port == 9001
    
    def test_custom_calibration_requirements(self):
        """Test custom calibration frame requirements."""
        view = UnifiedView(charuco_frames_required=20, skeleton_poses_required=5)
        
        assert view.state.calibration.charuco_required == 20
        assert view.state.calibration.skeleton_required == 5


# ============================================================================
# Camera Grid Tests
# ============================================================================

class TestCameraGrid:
    """Tests for camera grid functionality."""
    
    def test_camera_grid_auto_detect(self, unified_view):
        """Test camera detection callback."""
        detected = []
        
        def detect_callback() -> List[int]:
            detected.append(True)
            return [0, 2, 4]
        
        unified_view.set_detect_cameras_callback(detect_callback)
        result = unified_view.detect_cameras()
        
        assert len(detected) == 1
        assert result == [0, 2, 4]
        assert len(unified_view.state.cameras) == 3
    
    def test_set_cameras(self, unified_view):
        """Test setting cameras directly."""
        unified_view.set_cameras([0, 1, 2, 3])
        
        assert len(unified_view.state.cameras) == 4
        assert 0 in unified_view.state.cameras
        assert 3 in unified_view.state.cameras
    
    def test_camera_grid_layout(self, view_with_cameras):
        """Test grid layout calculation."""
        rows, cols = view_with_cameras.get_camera_grid_layout()
        
        # 4 cameras should be 2x2
        assert rows == 2
        assert cols == 2
    
    def test_camera_grid_layout_single(self, unified_view):
        """Test grid layout for single camera."""
        unified_view.set_cameras([0])
        rows, cols = unified_view.get_camera_grid_layout()
        
        assert rows == 1
        assert cols == 1
    
    def test_camera_grid_layout_empty(self, unified_view):
        """Test grid layout for no cameras."""
        rows, cols = unified_view.get_camera_grid_layout()
        
        assert rows == 0
        assert cols == 0
    
    def test_update_camera_frame(self, view_with_cameras):
        """Test updating camera frame info."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        view_with_cameras.update_camera_frame(
            camera_id=0,
            frame=frame,
            fps=30.0,
            board_visible=True,
            apriltags_detected=2,
        )
        
        info = view_with_cameras.state.cameras[0]
        assert info.fps == 30.0
        assert info.board_visible is True
        assert info.apriltags_detected == 2


# ============================================================================
# Calibration Mode Tests
# ============================================================================

class TestCalibrationMode:
    """Tests for calibration mode functionality."""
    
    def test_calibration_mode_toggle(self, unified_view):
        """Test switching calibration modes."""
        assert unified_view.state.calibration.mode == CalibrationMode.CHARUCO
        
        unified_view.set_calibration_mode(CalibrationMode.SKELETON)
        assert unified_view.state.calibration.mode == CalibrationMode.SKELETON
        
        unified_view.set_calibration_mode(CalibrationMode.CHARUCO)
        assert unified_view.state.calibration.mode == CalibrationMode.CHARUCO
    
    def test_skeleton_mode_fields_exist(self, unified_view):
        """Test that skeleton mode fields still exist (hidden in UI but kept for future)."""
        # ChArUco mode - default
        unified_view.set_calibration_mode(CalibrationMode.CHARUCO)
        assert unified_view.state.calibration.mode == CalibrationMode.CHARUCO
        
        # Skeleton mode enum still exists
        unified_view.set_calibration_mode(CalibrationMode.SKELETON)
        assert unified_view.state.calibration.mode == CalibrationMode.SKELETON
        
        # Skeleton-related fields still exist
        cal = unified_view.state.calibration
        assert hasattr(cal, 'skeleton_poses')
        assert hasattr(cal, 'skeleton_required')
        assert hasattr(cal, 'is_skeleton_only')
    
    def test_calibration_progress_update(self, unified_view):
        """Test calibration progress updates with pairwise fields."""
        unified_view.update_calibration_progress(
            charuco_frames=5,
            is_calibrated=False,
            per_camera_progress={0: 3, 2: 5},
            pairwise_progress={(0, 2): 4},
            is_connected=False,
        )
        
        cal = unified_view.state.calibration
        assert cal.charuco_frames == 5
        assert cal.is_calibrated is False
        assert cal.per_camera_progress == {0: 3, 2: 5}
        assert cal.pairwise_progress == {(0, 2): 4}
        assert cal.is_connected is False
    
    def test_calibration_status_text_charuco(self, unified_view):
        """Test calibration status text for ChArUco mode."""
        unified_view.set_calibration_mode(CalibrationMode.CHARUCO)
        unified_view.update_calibration_progress(
            per_camera_progress={0: 5, 2: 3}
        )
        
        text = unified_view.get_calibration_status_text()
        # New format shows camera count
        assert "ChArUco" in text or "cameras" in text
    
    def test_calibration_status_text_with_connectivity(self, unified_view):
        """Test calibration status text with connectivity."""
        unified_view.update_calibration_progress(is_connected=True)
        
        text = unified_view.get_calibration_status_text()
        assert "Computing" in text
        
        conn_status = unified_view.get_connectivity_status()
        assert "Connected" in conn_status
    
    def test_charuco_capture_callbacks(self, unified_view):
        """Test ChArUco capture start/stop callbacks."""
        started = []
        stopped = []
        
        unified_view.set_charuco_callbacks(
            on_start=lambda: started.append(True),
            on_stop=lambda: stopped.append(True),
        )
        
        unified_view.start_charuco_capture()
        assert len(started) == 1
        assert unified_view.state.is_charuco_capture_active is True
        
        unified_view.stop_charuco_capture()
        assert len(stopped) == 1
        assert unified_view.state.is_charuco_capture_active is False


# ============================================================================
# Tracking Control Tests
# ============================================================================

class TestTrackingControls:
    """Tests for tracking controls."""
    
    def test_start_tracking_button(self, unified_view):
        """Test starting tracking."""
        started = []
        unified_view.set_tracking_callbacks(
            on_start=lambda: started.append(True),
            on_stop=lambda: None,
        )
        
        assert unified_view.state.tracking_mode == TrackingMode.STOPPED
        
        unified_view.start_tracking()
        assert len(started) == 1
        assert unified_view.state.tracking_mode == TrackingMode.STARTING
    
    def test_stop_tracking_button(self, unified_view):
        """Test stopping tracking."""
        stopped = []
        unified_view.set_tracking_callbacks(
            on_start=lambda: None,
            on_stop=lambda: stopped.append(True),
        )
        
        # Start first
        unified_view.start_tracking()
        unified_view.on_tracking_started()
        assert unified_view.state.tracking_mode == TrackingMode.RUNNING
        
        # Then stop
        unified_view.stop_tracking()
        assert len(stopped) == 1
        assert unified_view.state.tracking_mode == TrackingMode.STOPPING
    
    def test_tracking_state_transitions(self, unified_view):
        """Test tracking state transitions."""
        assert unified_view.state.tracking_mode == TrackingMode.STOPPED
        
        unified_view.start_tracking()
        assert unified_view.state.tracking_mode == TrackingMode.STARTING
        
        unified_view.on_tracking_started()
        assert unified_view.state.tracking_mode == TrackingMode.RUNNING
        
        unified_view.stop_tracking()
        assert unified_view.state.tracking_mode == TrackingMode.STOPPING
        
        unified_view.on_tracking_stopped()
        assert unified_view.state.tracking_mode == TrackingMode.STOPPED
    
    def test_tracking_button_label(self, unified_view):
        """Test tracking button label changes with state."""
        assert "Start" in unified_view.get_tracking_button_label()
        
        unified_view.start_tracking()
        assert "Starting" in unified_view.get_tracking_button_label()
        
        unified_view.on_tracking_started()
        assert "Stop" in unified_view.get_tracking_button_label()
        
        unified_view.stop_tracking()
        assert "Stopping" in unified_view.get_tracking_button_label()
    
    def test_tracking_button_enabled(self, unified_view):
        """Test tracking button enabled state."""
        # Enabled when stopped
        assert unified_view.is_tracking_button_enabled() is True
        
        # Disabled when starting
        unified_view.start_tracking()
        assert unified_view.is_tracking_button_enabled() is False
        
        # Enabled when running
        unified_view.on_tracking_started()
        assert unified_view.is_tracking_button_enabled() is True
        
        # Disabled when stopping
        unified_view.stop_tracking()
        assert unified_view.is_tracking_button_enabled() is False
    
    def test_tracking_status_update(self, unified_view):
        """Test tracking status metrics update."""
        unified_view.update_tracking_status(fps=62.5, active_joints=15)
        
        assert unified_view.state.tracking_fps == 62.5
        assert unified_view.state.active_joints == 15


# ============================================================================
# AprilTag Toggle Tests
# ============================================================================

class TestAprilTagToggle:
    """Tests for AprilTag precision toggle."""
    
    def test_apriltag_toggle(self, unified_view):
        """Test toggling AprilTag mode."""
        assert unified_view.state.apriltags_enabled is False
        
        unified_view.toggle_apriltags()
        assert unified_view.state.apriltags_enabled is True
        
        unified_view.toggle_apriltags()
        assert unified_view.state.apriltags_enabled is False
    
    def test_apriltag_callback(self, unified_view):
        """Test AprilTag toggle callback."""
        toggled = []
        
        unified_view.set_apriltag_callback(lambda enabled: toggled.append(enabled))
        
        unified_view.toggle_apriltags()
        assert toggled == [True]
        
        unified_view.toggle_apriltags()
        assert toggled == [True, False]
    
    def test_set_apriltags_enabled(self, unified_view):
        """Test setting AprilTag mode directly."""
        unified_view.set_apriltags_enabled(True)
        assert unified_view.state.apriltags_enabled is True
        
        unified_view.set_apriltags_enabled(False)
        assert unified_view.state.apriltags_enabled is False


# ============================================================================
# Export Button Tests
# ============================================================================

class TestExportButtons:
    """Tests for export buttons."""
    
    def test_export_charuco_button(self, unified_view):
        """Test ChArUco export callback."""
        exported = []
        
        unified_view.set_export_callbacks(
            on_export_charuco=lambda p: (exported.append(('charuco', p)), True)[1],
            on_export_apriltag=lambda p: True,
        )
        
        result = unified_view.export_charuco_pdf(Path("/tmp/board.pdf"))
        
        assert result is True
        assert len(exported) == 1
        assert exported[0][0] == 'charuco'
    
    def test_export_apriltag_button(self, unified_view):
        """Test AprilTag export callback."""
        exported = []
        
        unified_view.set_export_callbacks(
            on_export_charuco=lambda p: True,
            on_export_apriltag=lambda p: (exported.append(('apriltag', p)), True)[1],
        )
        
        result = unified_view.export_apriltag_pdf(Path("/tmp/tags.pdf"))
        
        assert result is True
        assert len(exported) == 1
        assert exported[0][0] == 'apriltag'


# ============================================================================
# OSC Settings Tests
# ============================================================================

class TestOSCSettings:
    """Tests for OSC configuration."""
    
    def test_osc_settings_update(self, unified_view):
        """Test updating OSC settings."""
        callback_args = []
        
        unified_view.set_osc_callback(lambda ip, port: callback_args.append((ip, port)))
        
        unified_view.set_osc_config("192.168.1.100", 9001)
        
        assert unified_view.state.osc_ip == "192.168.1.100"
        assert unified_view.state.osc_port == 9001
        assert callback_args == [("192.168.1.100", 9001)]
    
    def test_osc_connected_status(self, unified_view):
        """Test OSC connection status."""
        assert unified_view.state.osc_connected is False
        assert "Disconnected" in unified_view.get_osc_status_text()
        
        unified_view.set_osc_connected(True)
        assert unified_view.state.osc_connected is True
        assert "Connected" in unified_view.get_osc_status_text()


# ============================================================================
# Floating Window Tests
# ============================================================================

class TestFloatingWindows:
    """Tests for floating window controls."""
    
    def test_floating_window_toggle(self, unified_view):
        """Test toggling floating windows."""
        assert unified_view.state.debug_window_open is False
        assert unified_view.state.performance_window_open is False
        
        unified_view.toggle_debug_window()
        assert unified_view.state.debug_window_open is True
        
        unified_view.toggle_performance_window()
        assert unified_view.state.performance_window_open is True
        
        unified_view.toggle_debug_window()
        assert unified_view.state.debug_window_open is False
    
    def test_set_window_visibility(self, unified_view):
        """Test setting window visibility directly."""
        unified_view.set_debug_window_open(True)
        assert unified_view.state.debug_window_open is True
        
        unified_view.set_performance_window_open(True)
        assert unified_view.state.performance_window_open is True


# ============================================================================
# State Callback Tests
# ============================================================================

class TestStateCallbacks:
    """Tests for state change callbacks."""
    
    def test_state_callback_on_camera_change(self, unified_view):
        """Test state callback fires on camera changes."""
        callbacks = []
        unified_view.add_state_callback(lambda state: callbacks.append(1))
        
        unified_view.set_cameras([0, 1])
        
        assert len(callbacks) == 1
    
    def test_state_callback_on_tracking_change(self, unified_view):
        """Test state callback fires on tracking changes."""
        callbacks = []
        unified_view.add_state_callback(lambda state: callbacks.append(1))
        
        unified_view.start_tracking()
        
        assert len(callbacks) == 1
    
    def test_multiple_callbacks(self, unified_view):
        """Test multiple state callbacks."""
        callback1 = []
        callback2 = []
        
        unified_view.add_state_callback(lambda state: callback1.append(1))
        unified_view.add_state_callback(lambda state: callback2.append(1))
        
        unified_view.toggle_apriltags()
        
        assert len(callback1) == 1
        assert len(callback2) == 1
