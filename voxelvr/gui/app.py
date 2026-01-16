"""
VoxelVR Main GUI Application

Main application window integrating all GUI panels.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import time
import threading
from typing import Optional, Dict, List, Callable
from pathlib import Path
from dataclasses import dataclass

from .camera_panel import CameraPanel
from .calibration_panel import CalibrationPanel, CalibrationStep
from .tracking_panel import TrackingPanel, TrackingState
from .performance_panel import PerformancePanel, PerformanceMetrics
from .debug_panel import DebugPanel
from .osc_status import OSCStatusIndicator, ConnectionState
from .param_optimizer import ParameterOptimizer, FilterProfile


@dataclass
class AppState:
    """Global application state."""
    is_running: bool = True
    active_tab: str = "cameras"
    cameras_detected: int = 0
    calibration_loaded: bool = False
    tracking_active: bool = False


class VoxelVRApp:
    """
    Main VoxelVR GUI application.
    
    Provides a tabbed interface with:
    - Camera Preview tab
    - Calibration tab
    - Tracking tab (with performance and OSC status)
    - Debug tab
    """
    
    def __init__(
        self,
        title: str = "VoxelVR - Full Body Tracking",
        width: int = 1280,
        height: int = 800,
    ):
        """
        Initialize the application.
        
        Args:
            title: Window title
            width: Window width
            height: Window height
        """
        self.title = title
        self.width = width
        self.height = height
        
        # Application state
        self.state = AppState()
        
        # Initialize panels
        self.camera_panel = CameraPanel()
        self.calibration_panel = CalibrationPanel()
        self.tracking_panel = TrackingPanel()
        self.performance_panel = PerformancePanel()
        self.debug_panel = DebugPanel()
        self.osc_status = OSCStatusIndicator()
        
        # DearPyGui IDs
        self._texture_registry: Optional[int] = None
        self._camera_textures: Dict[int, int] = {}
        
        # Update thread
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Camera Preview
        from ..capture.manager import CameraManager
        self.camera_manager: Optional[CameraManager] = None
        self.preview_thread: Optional[threading.Thread] = None
        self.preview_active = False
        
        # External callbacks
        self._on_start_tracking: Optional[Callable] = None
        self._on_stop_tracking: Optional[Callable] = None
        self._on_calibration_start: Optional[Callable] = None
        self._on_calibration_capture: Optional[Callable] = None
        
        # Connect panel callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        """Setup internal callbacks between panels."""
        # Tracking state changes
        self.tracking_panel.add_state_callback(self._on_tracking_state_change)
        
        # Calibration step changes
        self.calibration_panel.add_step_callback(self._on_calibration_step_change)
        
        # OSC status changes
        self.osc_status.add_state_callback(self._on_osc_state_change)
        
        # Debug panel parameter changes -> tracking pipeline
        self.debug_panel.add_param_callback(self._on_filter_params_change)
    
    def _on_tracking_state_change(self, state: TrackingState) -> None:
        """Handle tracking state changes."""
        self.state.tracking_active = state == TrackingState.RUNNING
        
        if state == TrackingState.STARTING and self._on_start_tracking:
            self._on_start_tracking()
        elif state == TrackingState.STOPPING and self._on_stop_tracking:
            self._on_stop_tracking()
    
    def _on_calibration_step_change(self, step: CalibrationStep) -> None:
        """Handle calibration step changes."""
        pass  # UI updates handled by calibration panel
    
    def _on_osc_state_change(self, state: ConnectionState) -> None:
        """Handle OSC connection state changes."""
        # Update UI indicator
        if dpg.does_item_exist("osc_status_text"):
            dpg.set_value("osc_status_text", self.osc_status.get_state_text())
    
    def _on_filter_params_change(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        """Handle filter parameter changes from debug panel."""
        # This would be connected to the actual pose filter
        pass
    
    def set_tracking_callbacks(
        self,
        on_start: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
    ) -> None:
        """Set callbacks for tracking start/stop."""
        self._on_start_tracking = on_start
        self._on_stop_tracking = on_stop
    
    def set_calibration_callbacks(
        self,
        on_start: Optional[Callable] = None,
        on_capture: Optional[Callable] = None,
    ) -> None:
        """Set callbacks for calibration events."""
        self._on_calibration_start = on_start
        self._on_calibration_capture = on_capture
    
    def setup(self) -> None:
        """Setup DearPyGui context and window."""
        dpg.create_context()
        dpg.create_viewport(title=self.title, width=self.width, height=self.height)
        
        # Create texture registry for camera feeds
        self._texture_registry = dpg.add_texture_registry()
        
        # Setup theme
        self._setup_theme()
        
        # Create main window
        self._create_main_window()
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def _setup_theme(self) -> None:
        """Setup global theme."""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 4)
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 35))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (45, 45, 50))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 70))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 80, 90))
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (50, 50, 60))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (70, 130, 180))
        
        dpg.bind_theme(global_theme)
    
    def _create_main_window(self) -> None:
        """Create the main application window."""
        with dpg.window(label="VoxelVR", tag="main_window", no_title_bar=True):
            # Header bar
            with dpg.group(horizontal=True):
                dpg.add_text("VoxelVR", color=(100, 180, 255))
                dpg.add_spacer(width=20)
                
                # OSC status indicator
                with dpg.group(horizontal=True):
                    dpg.add_text("OSC:", color=(150, 150, 150))
                    dpg.add_text("Disconnected", tag="osc_status_text", color=(128, 128, 128))
                
                dpg.add_spacer()
                
                # Performance summary
                dpg.add_text("FPS: --", tag="fps_summary")
            
            dpg.add_separator()
            
            # Tab bar
            with dpg.tab_bar(tag="main_tabs"):
                # Camera Preview Tab
                with dpg.tab(label="Cameras", tag="cameras_tab"):
                    self._create_cameras_tab()
                
                # Calibration Tab
                with dpg.tab(label="Calibration", tag="calibration_tab"):
                    self._create_calibration_tab()
                
                # Tracking Tab
                with dpg.tab(label="Tracking", tag="tracking_tab"):
                    self._create_tracking_tab()
                
                # Debug Tab
                with dpg.tab(label="Debug", tag="debug_tab"):
                    self._create_debug_tab()
        
        # Set main window as primary
        dpg.set_primary_window("main_window", True)
    
    def _create_cameras_tab(self) -> None:
        """Create the camera preview tab."""
        with dpg.group():
            dpg.add_text("Camera Preview", color=(150, 200, 255))
            dpg.add_text("Position your cameras to cover your play area.", color=(150, 150, 150))
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Detect Cameras", callback=self._on_detect_cameras)
                dpg.add_button(label="Refresh", callback=self._on_refresh_cameras)
            
            dpg.add_separator()
            
            # Camera grid container
            with dpg.child_window(tag="camera_grid", autosize_x=True, height=500):
                dpg.add_text("No cameras detected. Click 'Detect Cameras'.", tag="no_cameras_text")
    
    def _create_calibration_tab(self) -> None:
        """Create the calibration tab."""
        with dpg.group():
            dpg.add_text("Camera Calibration", color=(150, 200, 255))
            dpg.add_separator()
            
            # Progress indicator
            with dpg.group(horizontal=True):
                dpg.add_text("Step: ")
                dpg.add_text("Not Started", tag="calib_step_text", color=(200, 200, 100))
            
            dpg.add_separator()
            
            # Instructions panel
            with dpg.child_window(tag="calib_instructions", height=150, autosize_x=True):
                dpg.add_text(
                    self.calibration_panel.get_step_instructions(),
                    tag="calib_instructions_text",
                    wrap=0,
                )
            
            dpg.add_separator()
            
            # Action buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start Calibration",
                    tag="calib_start_btn",
                    callback=self._on_calibration_start_click,
                )
                dpg.add_button(
                    label="Export Board PDF",
                    tag="calib_export_btn",
                    callback=self._on_export_board_click,
                )
                dpg.add_button(
                    label="Next Step",
                    tag="calib_next_btn",
                    callback=self._on_calibration_next_click,
                    enabled=False,
                )
                dpg.add_button(
                    label="Cancel",
                    tag="calib_cancel_btn",
                    callback=self._on_calibration_cancel_click,
                    enabled=False,
                )
            
            dpg.add_separator()
            
            # Camera status grid
            dpg.add_text("Camera Status:", color=(150, 150, 150))
            with dpg.child_window(tag="calib_camera_status", height=100, autosize_x=True):
                dpg.add_text("Configure cameras first.", tag="calib_no_cameras")
    
    def _create_tracking_tab(self) -> None:
        """Create the tracking tab."""
        with dpg.group():
            dpg.add_text("Body Tracking", color=(150, 200, 255))
            dpg.add_separator()
            
            # Split view: controls on left, status on right
            with dpg.group(horizontal=True):
                # Controls column
                with dpg.child_window(width=350, height=400):
                    dpg.add_text("Controls", color=(150, 150, 150))
                    dpg.add_separator()
                    
                    # Start/Stop button
                    dpg.add_button(
                        label="Start Tracking",
                        tag="tracking_toggle_btn",
                        callback=self._on_tracking_toggle,
                        width=200,
                    )
                    
                    dpg.add_spacer(height=10)
                    
                    # OSC Configuration
                    dpg.add_text("OSC Configuration", color=(150, 150, 150))
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("IP:")
                        dpg.add_input_text(
                            tag="osc_ip_input",
                            default_value=self.tracking_panel.osc_ip,
                            width=120,
                            callback=self._on_osc_config_change,
                        )
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Port:")
                        dpg.add_input_int(
                            tag="osc_port_input",
                            default_value=self.tracking_panel.osc_port,
                            width=100,
                            callback=self._on_osc_config_change,
                        )
                    
                    dpg.add_spacer(height=10)
                    
                    # Tracker toggles
                    dpg.add_text("Enabled Trackers", color=(150, 150, 150))
                    
                    for name, config in self.tracking_panel.get_all_trackers().items():
                        dpg.add_checkbox(
                            label=config.display_name,
                            tag=f"tracker_{name}",
                            default_value=config.enabled,
                            callback=lambda s, a, u: self._on_tracker_toggle(u, dpg.get_value(s)),
                            user_data=name,
                        )
                
                # Status column
                with dpg.child_window(autosize_x=True, height=400):
                    dpg.add_text("Status", color=(150, 150, 150))
                    dpg.add_separator()
                    
                    dpg.add_text("Tracking stopped", tag="tracking_status_text")
                    
                    dpg.add_spacer(height=10)
                    
                    # Joint validity display
                    dpg.add_text("Joint Status:", color=(150, 150, 150))
                    
                    with dpg.child_window(height=200, autosize_x=True, tag="joint_status_container"):
                        for joint in self.tracking_panel.get_joint_info():
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"{joint['name']}:", tag=f"joint_label_{joint['index']}")
                                dpg.add_text("--", tag=f"joint_status_{joint['index']}", color=(128, 128, 128))
            
            dpg.add_separator()
            
            # Performance section
            dpg.add_text("Performance", color=(150, 150, 150))
            with dpg.child_window(height=100, autosize_x=True):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("FPS")
                        dpg.add_text("--", tag="perf_fps", color=(100, 200, 100))
                    
                    dpg.add_spacer(width=30)
                    
                    with dpg.group():
                        dpg.add_text("Latency")
                        dpg.add_text("-- ms", tag="perf_latency", color=(100, 200, 100))
                    
                    dpg.add_spacer(width=30)
                    
                    with dpg.group():
                        dpg.add_text("Joints")
                        dpg.add_text("--/17", tag="perf_joints", color=(100, 200, 100))
    
    def _create_debug_tab(self) -> None:
        """Create the debug tab."""
        with dpg.group():
            dpg.add_text("Debug & Tuning", color=(150, 200, 255))
            dpg.add_text("Adjust filter parameters to balance jitter vs. latency.", color=(150, 150, 150))
            dpg.add_separator()
            
            # Split view
            with dpg.group(horizontal=True):
                # Parameter controls
                with dpg.child_window(width=400, height=400):
                    dpg.add_text("Filter Parameters", color=(150, 150, 150))
                    dpg.add_separator()
                    
                    # Profile selector
                    dpg.add_text("Profile:")
                    dpg.add_combo(
                        items=["Low Jitter", "Balanced", "Low Latency", "Precision"],
                        tag="profile_combo",
                        default_value="Balanced",
                        callback=self._on_profile_change,
                        width=200,
                    )
                    
                    dpg.add_spacer(height=10)
                    
                    # Auto-adjust toggle
                    dpg.add_checkbox(
                        label="Auto-Adjust (Continuous)",
                        tag="auto_adjust_checkbox",
                        default_value=True,
                        callback=self._on_auto_adjust_toggle,
                    )
                    
                    dpg.add_spacer(height=10)
                    
                    # Parameter sliders
                    ranges = self.debug_panel.get_slider_ranges()
                    
                    dpg.add_text("Min Cutoff (lower = smoother)")
                    dpg.add_slider_float(
                        tag="min_cutoff_slider",
                        min_value=ranges['min_cutoff'][0],
                        max_value=ranges['min_cutoff'][1],
                        default_value=ranges['min_cutoff'][2],
                        callback=lambda s, a: self._on_slider_change('min_cutoff', a),
                        width=300,
                    )
                    
                    dpg.add_text("Beta (higher = less lag during movement)")
                    dpg.add_slider_float(
                        tag="beta_slider",
                        min_value=ranges['beta'][0],
                        max_value=ranges['beta'][1],
                        default_value=ranges['beta'][2],
                        callback=lambda s, a: self._on_slider_change('beta', a),
                        width=300,
                    )
                    
                    dpg.add_text("D Cutoff")
                    dpg.add_slider_float(
                        tag="d_cutoff_slider",
                        min_value=ranges['d_cutoff'][0],
                        max_value=ranges['d_cutoff'][1],
                        default_value=ranges['d_cutoff'][2],
                        callback=lambda s, a: self._on_slider_change('d_cutoff', a),
                        width=300,
                    )
                    
                    dpg.add_spacer(height=10)
                    
                    dpg.add_button(
                        label="Manual Override",
                        tag="manual_override_btn",
                        callback=self._on_manual_override,
                    )
                    dpg.add_button(
                        label="Reset to Defaults",
                        callback=self._on_reset_defaults,
                    )
                
                # Metrics display
                with dpg.child_window(autosize_x=True, height=400):
                    dpg.add_text("Metrics", color=(150, 150, 150))
                    dpg.add_separator()
                    
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("Jitter")
                            dpg.add_text("-- mm", tag="debug_jitter", color=(100, 200, 100))
                        
                        dpg.add_spacer(width=30)
                        
                        with dpg.group():
                            dpg.add_text("Est. Latency")
                            dpg.add_text("-- ms", tag="debug_latency", color=(100, 200, 100))
                    
                    dpg.add_spacer(height=10)
                    
                    # Jitter graph placeholder
                    dpg.add_text("Jitter Over Time:", color=(150, 150, 150))
                    
                    # Simple line series for jitter
                    with dpg.plot(height=200, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="jitter_x_axis")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Jitter (mm)", tag="jitter_y_axis")
                        dpg.add_line_series(
                            [],
                            [],
                            tag="jitter_series",
                            parent="jitter_y_axis",
                        )
    
    # =========================================================================
    # Callback handlers
    # =========================================================================
    
    
    def _start_preview(self, camera_ids: List[int]) -> None:
        """Start camera preview."""
        self._stop_preview()
        
        print(f"Starting preview for cameras: {camera_ids}")
        from ..capture.manager import CameraManager
        from ..config import CameraConfig
        
        from ..capture.camera import Camera
        
        configs = [
            CameraConfig(id=i, resolution=Camera.get_best_resolution(i), fps=30) 
            for i in camera_ids
        ]
        
        self.camera_manager = CameraManager(configs)
        if self.camera_manager.start_all():
            self.preview_active = True
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
        else:
            print("Failed to start preview cameras")

    def _stop_preview(self) -> None:
        """Stop camera preview."""
        self.preview_active = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
            self.preview_thread = None
            
        if self.camera_manager:
            self.camera_manager.stop_all()
            self.camera_manager = None

    def _preview_loop(self) -> None:
        """Background loop for camera preview."""
        while self.preview_active and self.state.is_running:
            if not self.camera_manager or not dpg.is_dearpygui_running():
                break
                
            # Get frames (non-blocking, don't need strict sync for preview)
            frames = self.camera_manager.get_all_latest_frames()
            
            for cam_id, frame in frames.items():
                self.update_camera_frame(cam_id, frame.image)
                
            time.sleep(0.01)

    def _on_detect_cameras(self, sender, app_data) -> None:
        """Handle detect cameras button."""
        import cv2
        
        from ..capture.manager import CameraManager
        detected = CameraManager.detect_cameras()
        
        self.state.cameras_detected = len(detected)
        
        # Update camera panel
        for cam_id in detected:
            self.camera_panel.add_camera(cam_id, f"Camera {cam_id}")
        
        # Update calibration panel
        self.calibration_panel.set_cameras(detected)
        
        # Update UI
        self._update_camera_grid(detected)
        
        # Start preview
        if detected:
            self._start_preview(detected)
    
    def _on_refresh_cameras(self, sender, app_data) -> None:
        """Handle refresh cameras button."""
        self._on_detect_cameras(sender, app_data)
    
    def _on_calibration_start_click(self, sender, app_data) -> None:
        """Handle calibration start button."""
        self.calibration_panel.start_calibration()
        self._update_calibration_ui()
        
        if self._on_calibration_start:
            self._on_calibration_start()
    
    def _on_export_board_click(self, sender, app_data) -> None:
        """Handle export board button."""
        from ..config import CONFIG_DIR
        
        output_path = CONFIG_DIR / "calibration" / "charuco_board.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.calibration_panel.export_board_pdf(output_path):
            print(f"Board exported to: {output_path}")
    
    def _on_calibration_next_click(self, sender, app_data) -> None:
        """Handle calibration next button."""
        self.calibration_panel.next_step()
        self._update_calibration_ui()
    
    def _on_calibration_cancel_click(self, sender, app_data) -> None:
        """Handle calibration cancel button."""
        self.calibration_panel.cancel_calibration()
        self._update_calibration_ui()
    
    def _on_tracking_toggle(self, sender, app_data) -> None:
        """Handle tracking start/stop toggle."""
        if self.tracking_panel.is_running:
            self.tracking_panel.request_stop()
            dpg.set_item_label("tracking_toggle_btn", "Start Tracking")
            
            # Restart preview after tracking stops (simple timeout check or callback?)
            # Ideally we rely on _on_tracking_state_change, but validation there is cleaner.
        else:
            # Stop preview to free cameras for tracking thread
            self._stop_preview()
            
            self.tracking_panel.request_start()
            dpg.set_item_label("tracking_toggle_btn", "Stop Tracking")

    def _on_tracking_state_change(self, state: TrackingState) -> None:
        """Handle tracking state changes."""
        self.state.tracking_active = state == TrackingState.RUNNING
        
        if state == TrackingState.STARTING and self._on_start_tracking:
            self._on_start_tracking()
        elif state == TrackingState.STOPPING and self._on_stop_tracking:
            if self._on_stop_tracking:
                self._on_stop_tracking()
            
            # Restart preview when fully stopped (state becomes STOPPED/IDLE)
            # Actually STOPPING means it's requesting stop.
            pass
            
        if state == TrackingState.IDLE:
             # If we were tracking, and now we are idle, restart preview
             if not self.preview_active and self.state.cameras_detected > 0:
                 # Recover known cameras from current panel config
                 # This is a bit hacky, ideally we store detected IDs in state
                 ids = list(self.camera_panel.frames.keys())
                 if ids:
                     self._start_preview(ids)
    
    def _on_osc_config_change(self, sender, app_data) -> None:
        """Handle OSC configuration change."""
        ip = dpg.get_value("osc_ip_input")
        port = dpg.get_value("osc_port_input")
        self.tracking_panel.set_osc_config(ip, port)
        self.osc_status.set_target(ip, port)
    
    def _on_tracker_toggle(self, tracker_name: str, enabled: bool) -> None:
        """Handle tracker enable/disable toggle."""
        self.tracking_panel.set_tracker_enabled(tracker_name, enabled)
    
    def _on_profile_change(self, sender, app_data) -> None:
        """Handle profile selection change."""
        profile_map = {
            "Low Jitter": "low_jitter",
            "Balanced": "balanced",
            "Low Latency": "low_latency",
            "Precision": "precision",
        }
        profile_name = profile_map.get(app_data, "balanced")
        self.debug_panel.set_profile(profile_name)
        self._update_debug_sliders()
    
    def _on_auto_adjust_toggle(self, sender, app_data) -> None:
        """Handle auto-adjust toggle."""
        self.debug_panel.set_auto_adjust(app_data)
    
    def _on_slider_change(self, param_name: str, value: float) -> None:
        """Handle parameter slider change."""
        if param_name == 'min_cutoff':
            self.debug_panel.set_min_cutoff(value)
        elif param_name == 'beta':
            self.debug_panel.set_beta(value)
        elif param_name == 'd_cutoff':
            self.debug_panel.set_d_cutoff(value)
    
    def _on_manual_override(self, sender, app_data) -> None:
        """Handle manual override button."""
        self.debug_panel.enable_manual_override()
        dpg.set_value("auto_adjust_checkbox", False)
    
    def _on_reset_defaults(self, sender, app_data) -> None:
        """Handle reset to defaults button."""
        self.debug_panel.reset_to_defaults()
        dpg.set_value("profile_combo", "Balanced")
        dpg.set_value("auto_adjust_checkbox", True)
        self._update_debug_sliders()
    
    # =========================================================================
    # UI Update methods
    # =========================================================================
    
    def _update_camera_grid(self, camera_ids: List[int]) -> None:
        """Update camera grid display."""
        # Clear existing
        if dpg.does_item_exist("no_cameras_text"):
            dpg.delete_item("no_cameras_text")
        
        if not camera_ids:
            dpg.add_text(
                "No cameras detected. Click 'Detect Cameras'.",
                tag="no_cameras_text",
                parent="camera_grid",
            )
            return
        
        # Create camera texture placeholders
        for cam_id in camera_ids:
            # Create texture if not exists
            if cam_id not in self._camera_textures:
                w, h = self.camera_panel.preview_size
                data = [0.1] * (w * h * 4)  # Dark RGBA
                
                texture_id = dpg.add_dynamic_texture(
                    width=w,
                    height=h,
                    default_value=data,
                    parent=self._texture_registry,
                )
                self._camera_textures[cam_id] = texture_id
        
        # Clear existing images
        dpg.delete_item("camera_grid", children_only=True)
        
        # Add images to grid
        cols = 3
        with dpg.table(parent="camera_grid", header_row=False, policy=dpg.mvTable_SizingFixedFit, resizable=True):
            for _ in range(cols):
                dpg.add_table_column()
            
            # Add rows
            rows = (len(camera_ids) + cols - 1) // cols
            cam_idx = 0
            
            for _ in range(rows):
                with dpg.table_row():
                    for _ in range(cols):
                        if cam_idx < len(camera_ids):
                            cam_id = camera_ids[cam_idx]
                            with dpg.group():
                                dpg.add_text(f"Camera {cam_id}")
                                if cam_id in self._camera_textures:
                                    dpg.add_image(
                                        self._camera_textures[cam_id],
                                        width=self.camera_panel.preview_size[0],
                                        height=self.camera_panel.preview_size[1],
                                    )
                            cam_idx += 1
    
    def _update_calibration_ui(self) -> None:
        """Update calibration tab UI."""
        state = self.calibration_panel.state
        step = state.current_step
        
        # Update step text
        step_names = {
            CalibrationStep.IDLE: "Not Started",
            CalibrationStep.EXPORT_BOARD: "1. Export Board",
            CalibrationStep.INTRINSIC_CAPTURE: "2. Intrinsic Calibration",
            CalibrationStep.INTRINSIC_COMPUTE: "2. Computing...",
            CalibrationStep.EXTRINSIC_CAPTURE: "3. Extrinsic Calibration",
            CalibrationStep.EXTRINSIC_COMPUTE: "3. Computing...",
            CalibrationStep.VERIFICATION: "4. Verification",
            CalibrationStep.COMPLETE: "Complete!",
        }
        dpg.set_value("calib_step_text", step_names.get(step, "Unknown"))
        
        # Update instructions
        dpg.set_value("calib_instructions_text", self.calibration_panel.get_step_instructions())
        
        # Update buttons
        is_running = state.is_running
        dpg.configure_item("calib_start_btn", enabled=not is_running)
        dpg.configure_item("calib_next_btn", enabled=is_running and step not in [
            CalibrationStep.INTRINSIC_COMPUTE,
            CalibrationStep.EXTRINSIC_COMPUTE,
            CalibrationStep.COMPLETE,
        ])
        dpg.configure_item("calib_cancel_btn", enabled=is_running)
    
    def _update_debug_sliders(self) -> None:
        """Update debug sliders from current values."""
        values = self.debug_panel.get_current_values()
        dpg.set_value("min_cutoff_slider", values['min_cutoff'])
        dpg.set_value("beta_slider", values['beta'])
        dpg.set_value("d_cutoff_slider", values['d_cutoff'])
    
    def update_performance(self, metrics: PerformanceMetrics) -> None:
        """Update performance display with new metrics."""
        self.performance_panel.update(metrics)
        
        # Update UI
        if dpg.does_item_exist("fps_summary"):
            dpg.set_value("fps_summary", f"FPS: {metrics.total_fps:.1f}")
        
        if dpg.does_item_exist("perf_fps"):
            dpg.set_value("perf_fps", f"{metrics.total_fps:.1f}")
        if dpg.does_item_exist("perf_latency"):
            dpg.set_value("perf_latency", f"{metrics.total_latency_ms:.1f} ms")
        if dpg.does_item_exist("perf_joints"):
            dpg.set_value("perf_joints", f"{metrics.num_valid_joints}/17")
    
    def update_tracking_status(self) -> None:
        """Update tracking status display."""
        status = self.tracking_panel.status
        
        if dpg.does_item_exist("tracking_status_text"):
            dpg.set_value("tracking_status_text", self.tracking_panel.get_status_text())
        
        # Update joint status
        for joint in self.tracking_panel.get_joint_info():
            tag = f"joint_status_{joint['index']}"
            if dpg.does_item_exist(tag):
                if joint['valid']:
                    dpg.set_value(tag, f"{joint['confidence']:.2f}")
                    dpg.configure_item(tag, color=(100, 200, 100))
                else:
                    dpg.set_value(tag, "--")
                    dpg.configure_item(tag, color=(128, 128, 128))
    
    def update_debug_metrics(self) -> None:
        """Update debug panel metrics display."""
        metrics = self.debug_panel.metrics
        
        if dpg.does_item_exist("debug_jitter"):
            dpg.set_value("debug_jitter", f"{metrics.jitter_position_mm:.2f} mm")
        if dpg.does_item_exist("debug_latency"):
            dpg.set_value("debug_latency", f"{metrics.estimated_latency_ms:.1f} ms")
        
        # Update jitter graph
        times, values = self.debug_panel.get_graph_data('jitter')
        if times and dpg.does_item_exist("jitter_series"):
            dpg.set_value("jitter_series", [times, values])
            dpg.fit_axis_data("jitter_x_axis")
            dpg.fit_axis_data("jitter_y_axis")
    
    def update_camera_frame(self, camera_id: int, frame: np.ndarray) -> None:
        """Update a camera frame texture."""
        self.camera_panel.update_frame(camera_id, frame)
        
        # Update texture
        if camera_id in self._camera_textures:
            frames = self.camera_panel.get_dpg_frames_flat()
            if camera_id in frames:
                dpg.set_value(self._camera_textures[camera_id], frames[camera_id].tolist())
    
    def run(self) -> None:
        """Run the application main loop."""
        self.setup()
        
        while dpg.is_dearpygui_running() and self.state.is_running:
            dpg.render_dearpygui_frame()
        
        self.shutdown()
    
    def shutdown(self) -> None:
        """Clean up and shutdown."""
        self.state.is_running = False
        self._stop_event.set()
        
        dpg.destroy_context()
    
    def request_stop(self) -> None:
        """Request application stop."""
        self.state.is_running = False
