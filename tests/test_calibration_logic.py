
import pytest
import numpy as np
import time
from unittest.mock import MagicMock
from voxelvr.gui.calibration_panel import CalibrationPanel, CalibrationState, CalibrationStep

class TestCalibrationLogic:
    """Tests for critical calibration logic."""

    def test_should_auto_capture_cooldown(self):
        """Test that auto-capture respects the time cooldown."""
        panel = CalibrationPanel()
        panel._state.auto_capture = True
        panel._capture_cooldown = 1.0
        
        camera_id = 0
        corners = np.zeros((4, 1, 2)) # Dummy corners
        
        # First capture
        # We simulate that a capture just happened
        panel._last_capture_time[camera_id] = time.time()
        
        # Immediate check should fail (too soon)
        assert panel.should_auto_capture(camera_id, corners) == False
        
        # Simulate time passing (hack time.time or just manually set last_capture_time back)
        panel._last_capture_time[camera_id] = time.time() - 1.5
        
        # Should now pass (if displacement check passes - which we need to handle)
        # For this test, we assume displacement is large enough or not checked yet if first frame?
        # Actually logic is: if cooldown passed -> check displacement.
        # If we provide different corners, it should pass.
        
        # We need to set _last_capture_corners to something different
        panel._last_capture_corners[camera_id] = np.ones((4, 1, 2)) * 100
        
        assert panel.should_auto_capture(camera_id, corners) == True

    def test_should_auto_capture_displacement(self):
        """Test that auto-capture requires sufficient movement."""
        panel = CalibrationPanel()
        panel._state.auto_capture = True
        panel._capture_cooldown = 0.0 # Disable cooldown for this test
        panel._movement_threshold = 10.0 # Set a known threshold
        
        camera_id = 0
        
        # Initial state: Captured corners at (0,0)
        last_corners = np.zeros((4, 1, 2))
        panel._last_capture_time[camera_id] = 0
        panel._last_capture_corners[camera_id] = last_corners
        
        # 1. Test EXACT same corners -> Should FAIL
        current_corners = np.zeros((4, 1, 2))
        assert panel.should_auto_capture(camera_id, current_corners) == False, "Should not capture identical pose"
        
        # 2. Test SMALL movement -> Should FAIL
        # Move all corners by 1 pixel. Sum of movement = 4 * sqrt(1^2 + 1^2) approx 4 * 1.414 = 5.6
        # Threshold is 10.0
        small_move = last_corners + 1.0 
        assert panel.should_auto_capture(camera_id, small_move) == False, "Should not capture small movement"
        
        # 3. Test LARGE movement -> Should PASS
        # Move all corners by 10 pixels. 
        large_move = last_corners + 10.0
        assert panel.should_auto_capture(camera_id, large_move) == True, "Should capture large movement"

    def test_should_auto_capture_corner_count_change(self):
        """Test that changing number of corners triggers capture."""
        panel = CalibrationPanel()
        panel._state.auto_capture = True
        panel._capture_cooldown = 0.0
        
        camera_id = 0
        panel._last_capture_time[camera_id] = 0
        panel._last_capture_corners[camera_id] = np.zeros((4, 1, 2))
        
        # Change to 6 corners
        new_corners = np.zeros((6, 1, 2))
        
        assert panel.should_auto_capture(camera_id, new_corners) == True
