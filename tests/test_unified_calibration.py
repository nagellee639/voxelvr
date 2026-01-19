"""
Tests for Unified GUI Calibration Integration

Tests the integration of CalibrationPanel with the unified GUI,
pairwise progress tracking, and auto-calibration workflow.
"""

import pytest
import numpy as np
from typing import Dict, Any
from pathlib import Path
import tempfile

from voxelvr.gui.unified_view import (
    UnifiedView,
    UnifiedViewState, 
    CalibrationMode,
    CalibrationStatus,
)
from voxelvr.gui.calibration_panel import (
    CalibrationPanel,
    CalibrationStep,
    CalibrationState,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def unified_view():
    """Create a UnifiedView instance."""
    return UnifiedView()


@pytest.fixture
def calibration_panel():
    """Create a CalibrationPanel instance."""
    return CalibrationPanel()


@pytest.fixture
def panel_with_cameras(calibration_panel):
    """Create a CalibrationPanel with cameras set up."""
    calibration_panel.set_cameras([0, 2, 4, 6])
    return calibration_panel


# ============================================================================
# CalibrationMode Tests
# ============================================================================

class TestCalibrationModeRetention:
    """Tests that skeleton mode is still available in code but hidden in UI."""
    
    def test_skeleton_mode_exists(self):
        """Skeleton mode enum value should still exist."""
        assert hasattr(CalibrationMode, 'SKELETON')
        assert CalibrationMode.SKELETON.value == "skeleton"
    
    def test_charuco_mode_is_default(self):
        """ChArUco should be the default mode."""
        status = CalibrationStatus()
        assert status.mode == CalibrationMode.CHARUCO
    
    def test_skeleton_fields_exist(self):
        """Skeleton-related fields should still exist."""
        status = CalibrationStatus()
        assert hasattr(status, 'skeleton_poses')
        assert hasattr(status, 'skeleton_required')
        assert hasattr(status, 'is_skeleton_only')


# ============================================================================
# Pairwise Progress Tests
# ============================================================================

class TestPairwiseProgress:
    """Tests for pairwise calibration progress tracking."""
    
    def test_pairwise_fields_exist(self, unified_view):
        """Pairwise progress fields should exist."""
        cal = unified_view.state.calibration
        assert hasattr(cal, 'per_camera_progress')
        assert hasattr(cal, 'pairwise_progress')
        assert hasattr(cal, 'is_connected')
    
    def test_update_pairwise_progress(self, unified_view):
        """Test updating pairwise progress."""
        per_camera = {0: 5, 2: 3, 4: 7}
        pairwise = {(0, 2): 4, (2, 4): 6}
        
        unified_view.update_calibration_progress(
            per_camera_progress=per_camera,
            pairwise_progress=pairwise,
            is_connected=False,
        )
        
        cal = unified_view.state.calibration
        assert cal.per_camera_progress == per_camera
        assert cal.pairwise_progress == pairwise
        assert cal.is_connected == False
    
    def test_connectivity_update(self, unified_view):
        """Test connectivity flag update."""
        unified_view.update_calibration_progress(is_connected=True)
        assert unified_view.state.calibration.is_connected == True
        
        unified_view.update_calibration_progress(is_connected=False)
        assert unified_view.state.calibration.is_connected == False
    
    def test_connectivity_status_text(self, unified_view):
        """Test connectivity status text."""
        unified_view.update_calibration_progress(is_connected=False)
        assert "Not Connected" in unified_view.get_connectivity_status()
        
        unified_view.update_calibration_progress(is_connected=True)
        assert "Connected" in unified_view.get_connectivity_status()


class TestCalibrationPanelIntegration:
    """Tests for CalibrationPanel integration with pairwise tracking."""
    
    def test_panel_has_pairwise_captures(self, panel_with_cameras):
        """CalibrationPanel should track pairwise captures."""
        assert hasattr(panel_with_cameras, '_pairwise_captures')
        assert isinstance(panel_with_cameras._pairwise_captures, dict)
    
    def test_progress_summary_structure(self, panel_with_cameras):
        """Progress summary should include pairwise data."""
        summary = panel_with_cameras.get_progress_summary()
        
        assert 'cameras' in summary
        assert 'pairwise' in summary
        assert 'pairwise_connected' in summary
        assert 'pairwise_disconnected_cameras' in summary
    
    def test_empty_cameras_not_connected(self, panel_with_cameras):
        """Empty calibration should not be connected."""
        summary = panel_with_cameras.get_progress_summary()
        assert summary['pairwise_connected'] == False
    
    def test_simulate_pairwise_progress(self, panel_with_cameras):
        """Test simulating pairwise capture progress."""
        # Manually add enough pairwise captures to connect cameras 0-2-4-6
        panel_with_cameras._pairwise_captures[(0, 2)] = [{}] * 10
        panel_with_cameras._pairwise_captures[(2, 4)] = [{}] * 10
        panel_with_cameras._pairwise_captures[(4, 6)] = [{}] * 10
        
        summary = panel_with_cameras.get_progress_summary()
        
        # Should now be connected (chain 0->2->4->6)
        assert summary['pairwise_connected'] == True
        assert len(summary['pairwise_disconnected_cameras']) == 0
    
    def test_partial_connectivity(self, panel_with_cameras):
        """Test partial connectivity detection."""
        # Only connect 0-2 and 4-6 (disconnected subgraphs)
        panel_with_cameras._pairwise_captures[(0, 2)] = [{}] * 10
        panel_with_cameras._pairwise_captures[(4, 6)] = [{}] * 10
        
        summary = panel_with_cameras.get_progress_summary()
        
        # Should NOT be fully connected
        assert summary['pairwise_connected'] == False


# ============================================================================
# Auto-Calibration Workflow Tests
# ============================================================================

class TestAutoCalibrationWorkflow:
    """Tests for auto-calibration workflow."""
    
    def test_auto_start_calibration(self, calibration_panel):
        """Test that calibration can be auto-started."""
        calibration_panel.set_cameras([0, 2])
        
        # Begin calibration phase directly (mimics auto-start)
        calibration_panel.begin_calibration()
        
        assert calibration_panel.current_step == CalibrationStep.CALIBRATION
    
    def test_process_frame_detections_format(self, panel_with_cameras):
        """Test detection processing with correct input format."""
        panel_with_cameras.begin_calibration()
        
        # Create mock detections
        detections = {
            0: {'success': False, 'corners': None, 'ids': None, 'frame': None},
            2: {'success': False, 'corners': None, 'ids': None, 'frame': None},
            4: {'success': False, 'corners': None, 'ids': None, 'frame': None},
            6: {'success': False, 'corners': None, 'ids': None, 'frame': None},
        }
        
        result = panel_with_cameras.process_frame_detections(detections)
        
        # Should return structured result
        assert 'intrinsics_captured' in result
        assert 'pairwise_captured' in result
        assert 'all_cameras_captured' in result


# ============================================================================
# Recording Tests
# ============================================================================

class TestRecordingFunctionality:
    """Tests for time-synced recording."""
    
    def test_unified_app_has_recording_fields(self):
        """UnifiedVoxelVRApp should have recording-related fields."""
        # We can't instantiate the full app without DearPyGui,
        # but we can verify the class has the expected methods
        from voxelvr.gui.unified_app import UnifiedVoxelVRApp
        
        assert hasattr(UnifiedVoxelVRApp, 'enable_recording')
        assert hasattr(UnifiedVoxelVRApp, '_record_frames')
        assert hasattr(UnifiedVoxelVRApp, '_cleanup_recording')


class TestCalibrationStatusText:
    """Tests for calibration status display."""
    
    def test_status_before_calibration(self, unified_view):
        """Test status text before calibration."""
        status = unified_view.get_calibration_status_text()
        # Should show camera count or calibration state
        assert "cameras" in status.lower() or "charuco" in status.lower()
    
    def test_status_when_connected(self, unified_view):
        """Test status when connected but not calibrated."""
        unified_view.update_calibration_progress(is_connected=True)
        status = unified_view.get_calibration_status_text()
        assert "Computing" in status or "calibrat" in status.lower()
    
    def test_status_when_calibrated(self, unified_view):
        """Test status when calibration complete."""
        unified_view.update_calibration_progress(
            is_calibrated=True,
            reprojection_error=0.45,
        )
        status = unified_view.get_calibration_status_text()
        assert "Calibrated" in status or "0.45" in status


# ============================================================================
# Integration Tests
# ============================================================================

class TestUnifiedViewCalibrationPanel:
    """Integration tests for UnifiedView with CalibrationPanel."""
    
    def test_view_and_panel_can_coexist(self, unified_view, panel_with_cameras):
        """Test that UnifiedView and CalibrationPanel work together."""
        # Start calibration in panel
        panel_with_cameras.begin_calibration()
        
        # Get progress from panel
        progress = panel_with_cameras.get_progress_summary()
        
        # Update view with panel progress
        unified_view.update_calibration_progress(
            per_camera_progress={0: 0, 2: 0, 4: 0, 6: 0},
            pairwise_progress={},
            is_connected=progress['pairwise_connected'],
        )
        
        # Both should reflect the same state
        assert unified_view.state.calibration.is_connected == progress['pairwise_connected']


# ============================================================================
# Intrinsic Retry Logic Tests  
# ============================================================================

class TestIntrinsicRetryLogic:
    """Tests for intrinsic calibration failure recovery."""
    
    def test_handle_intrinsic_failure_first_attempt(self, panel_with_cameras):
        """Test that first failure increments retry count and clears frames."""
        panel_with_cameras.begin_calibration()
        status = panel_with_cameras._state.cameras[0]
        
        # Initial state
        initial_required = status.intrinsic_frames_required
        
        # Simulate having some captured frames
        panel_with_cameras._intrinsic_frames[0] = [{'frame': None} for _ in range(15)]
        
        # Call failure handler
        panel_with_cameras._handle_intrinsic_failure(0, status)
        
        # Verify retry behavior - frames cleared, required stays same
        assert status.intrinsic_retry_count == 1
        assert status.intrinsic_frames_required == initial_required  # No increase
        assert status.intrinsic_frames_captured == 0  # Frames cleared
    
    def test_handle_intrinsic_failure_max_retries(self, panel_with_cameras):
        """Test that 10th failure marks as final failure."""
        panel_with_cameras.begin_calibration()
        status = panel_with_cameras._state.cameras[0]
        
        # Simulate 9 prior failures
        status.intrinsic_retry_count = 9
        initial_required = status.intrinsic_frames_required
        
        # Call failure handler for 10th failure
        panel_with_cameras._handle_intrinsic_failure(0, status)
        
        # Verify max retries reached and marked as failed
        assert status.intrinsic_retry_count == 10
        assert status.intrinsic_failed == True
        # Required count should NOT increase after max retries
        assert status.intrinsic_frames_required == initial_required
    
    def test_frames_cleared_on_failure(self, panel_with_cameras):
        """Test that captured frames are cleared on calibration failure."""
        panel_with_cameras.begin_calibration()
        
        # Simulate captured frames
        mock_frames = [{'frame': f'frame_{i}', 'corners': None, 'ids': None} for i in range(20)]
        panel_with_cameras._intrinsic_frames[0] = mock_frames
        
        status = panel_with_cameras._state.cameras[0]
        initial_required = status.intrinsic_frames_required
        
        panel_with_cameras._handle_intrinsic_failure(0, status)
        
        # Frames should be cleared (not preserved)
        assert len(panel_with_cameras._intrinsic_frames[0]) == 0
        assert status.intrinsic_frames_captured == 0
        # Required count should stay the same (not increase)
        assert status.intrinsic_frames_required == initial_required

