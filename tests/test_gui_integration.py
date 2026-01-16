
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import os

from voxelvr.gui.app import VoxelVRApp
from voxelvr.gui.calibration_panel import CalibrationStep
from voxelvr.pose import PoseFilter
from voxelvr.gui.param_optimizer import FilterProfile

class TestGUIIntegration:
    @pytest.fixture
    def app(self):
        # Create app in headless mode (mocking dpg)
        with patch('dearpygui.dearpygui.create_context'), \
             patch('dearpygui.dearpygui.create_viewport'), \
             patch('dearpygui.dearpygui.setup_dearpygui'), \
             patch('dearpygui.dearpygui.show_viewport'), \
             patch('dearpygui.dearpygui.destroy_context'), \
             patch('dearpygui.dearpygui.add_texture_registry'):
            
            app = VoxelVRApp(title="Test App")
            # We mock the internal dpg calls
            return app

    def test_debug_panel_updates_filter(self, app):
        """Test that debug panel updates flow to a connected PoseFilter."""
        
        # 1. Create a real PoseFilter
        pose_filter = PoseFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
        
        # 2. Connect app callback to the filter (mimicking run_gui.py logic)
        def on_params_change(min_cutoff, beta, d_cutoff):
            pose_filter.update_parameters(min_cutoff, beta, d_cutoff)
        
        app.debug_panel.add_param_callback(on_params_change)
        
        # 3. Change parameters in debug panel (via optimizer)
        # Verify initial state
        assert pose_filter.min_cutoff == 1.0
        
        # Change profile to Low Latency (should update params)
        app.debug_panel.set_profile('low_latency')
        
        # Check if filter updated
        # Low Latency profile: min_cutoff=2.0, beta=1.5, d_cutoff=1.5
        assert pose_filter.min_cutoff == 2.0
        assert pose_filter.beta == 1.5
        
        # 4. Manual override
        app.debug_panel.enable_manual_override()
        app.debug_panel.set_min_cutoff(5.0)
        
        assert pose_filter.min_cutoff == 5.0
        
    def test_calibration_export_pdf(self, app, tmp_path):
        """Test PDF export hook in calibration panel."""
        
        # Mock the generate_charuco_pdf_file function to avoid actual PDF creation
        with patch('voxelvr.calibration.charuco.generate_charuco_pdf_file') as mock_gen_pdf:
            output_path = tmp_path / "test_board.pdf"
            
            # Trigger export
            result = app.calibration_panel.export_board_pdf(output_path)
            
            assert result is True
            mock_gen_pdf.assert_called_once()
            
            # Verify call args
            args, _ = mock_gen_pdf.call_args
            assert args[0] == output_path.with_suffix('.pdf')
            assert args[1] == 5  # squares_x default
            assert args[2] == 5  # squares_y default
            
    def test_optimizer_integration(self, app):
        """Test optimizer logic within the app context."""
        
        optimizer = app.debug_panel.optimizer
        
        # Simulate some data inputs
        positions = np.zeros((17, 3))
        valid_mask = np.ones(17, dtype=bool)
        
        # Feed data
        app.debug_panel.update(positions, valid_mask, timestamp=1.0)
        app.debug_panel.update(positions, valid_mask, timestamp=1.1)
        
        # Check metrics updated
        assert app.debug_panel.metrics.jitter_position_mm == 0.0
