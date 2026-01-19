"""
VoxelVR Unified GUI Application

Simplified single-view interface focused on tracking with
integrated calibration controls.
"""

import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import time
import threading
from typing import Optional, Dict, List, Callable
from pathlib import Path
from dataclasses import dataclass

from .unified_view import (
    UnifiedView, UnifiedViewState, CalibrationMode, TrackingMode,
    CameraFeedInfo, CalibrationStatus
)
from .performance_panel import PerformancePanel, PerformanceMetrics
from .debug_panel import DebugPanel
from .skeleton_viewer import SkeletonViewer, get_tpose
from ..transport.osc_sender import OSCSender
from ..config import VoxelVRConfig
from ..utils.logging import log_info, log_warn, log_error, log_debug, osc_log, calibration_log, camera_log


class UnifiedVoxelVRApp:
    """
    Unified VoxelVR GUI application.
    
    Provides a single-view interface with:
    - Camera grid display (auto-detect)
    - Calibration mode toggle with status
    - Tracking controls with AprilTag toggle
    - Floating debug/performance windows
    """
    
    def __init__(
        self,
        title: str = "VoxelVR - Unified Tracking",
        width: int = 1400,
        height: int = 900,
        force_cpu: bool = False,
        config: Optional[VoxelVRConfig] = None,
    ):
        self.title = title
        self.width = width
        self.height = height
        self.force_cpu = force_cpu
        
        # Load config if not provided
        self.config = config if config is not None else VoxelVRConfig.load()
        
        # Core view state
        self.view = UnifiedView()
        
        # Panels for floating windows
        self.performance_panel = PerformancePanel()
        self.debug_panel = DebugPanel()
        self.skeleton_viewer = SkeletonViewer(size=(400, 400))
        
        # Calibration panel for pairwise capture logic
        # Calibration panel for pairwise capture logic
        from .calibration_panel import CalibrationPanel
        self.calibration_panel = CalibrationPanel(
            charuco_squares_x=self.config.calibration.charuco_squares_x,
            charuco_squares_y=self.config.calibration.charuco_squares_y,
            charuco_square_length=self.config.calibration.charuco_square_length,
            charuco_marker_length=self.config.calibration.charuco_marker_length,
            charuco_dict=self.config.calibration.charuco_dict,
            intrinsic_frames_required=self.config.calibration.intrinsic_frames_required,
            extrinsic_frames_required=self.config.calibration.extrinsic_frames_required,
        )
        self.calibration_panel.add_progress_callback(self._on_calibration_progress)
        
        # ChArUco board for detection
        # ChArUco board for detection
        from ..calibration.charuco import create_charuco_board
        self._charuco_board, self._aruco_dict = create_charuco_board(
            squares_x=self.config.calibration.charuco_squares_x,
            squares_y=self.config.calibration.charuco_squares_y,
            square_length=self.config.calibration.charuco_square_length,
            marker_length=self.config.calibration.charuco_marker_length,
            dictionary=self.config.calibration.charuco_dict,
        )
        
        # DearPyGui resources
        self._texture_registry: Optional[int] = None
        self._camera_textures: Dict[int, int] = {}
        self._skeleton_texture: Optional[int] = None
        
        # Threading
        self._stop_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None
        self._grid_lock = threading.Lock()
        
        # Camera preview
        from ..capture.manager import CameraManager
        self.camera_manager: Optional[CameraManager] = None
        self.preview_thread: Optional[threading.Thread] = None
        self.preview_active = False

        # OSC Sender (Owner)
        self.osc_sender = OSCSender(ip="127.0.0.1", port=9000)
        
        # Post-calibration (origin/axis alignment for VRChat)
        from ..transport.post_calibration import PostCalibrator
        self.post_calibrator = PostCalibrator()
        
        # Idle OSC Loop
        self._idle_osc_thread: Optional[threading.Thread] = None

        
        # Recording support
        self._recording_enabled = False
        self._recording_dir: Optional[Path] = None
        self._video_writers: Dict[int, cv2.VideoWriter] = {}
        
        # External callbacks
        self._on_start_tracking: Optional[Callable] = None
        self._on_stop_tracking: Optional[Callable] = None
        
        # Camera scaling
        self._camera_scale = 1.0  # 0.5 = small, 1.0 = default, 2.0 = large
        self._base_camera_size = (360, 270)  # Base size at scale 1.0
        
        # Tracker preset settings
        self._tracker_preset = "full_body"  # full_body, upper_body, lower_body, legs_waist, custom
        self._enabled_joints = set(range(17))  # All joints enabled by default
        
        # COCO joint names for UI
        self._joint_names = [
            "Nose", "L.Eye", "R.Eye", "L.Ear", "R.Ear",
            "L.Shoulder", "R.Shoulder", "L.Elbow", "R.Elbow",
            "L.Wrist", "R.Wrist", "L.Hip", "R.Hip",
            "L.Knee", "R.Knee", "L.Ankle", "R.Ankle"
        ]
        
        # Tracker presets
        self._tracker_presets = {
            "full_body": set(range(17)),
            "upper_body": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},  # Head + arms
            "lower_body": {5, 6, 11, 12, 13, 14, 15, 16},  # Shoulders + legs
            "legs_waist": {11, 12, 13, 14, 15, 16},  # Hips + legs only
        }
        
        # Connect view callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        """Wire up unified view callbacks."""
        self.view.set_detect_cameras_callback(self._detect_cameras)
        self.view.set_tracking_callbacks(
            on_start=self._on_tracking_start,
            on_stop=self._on_tracking_stop,
        )
        self.view.set_export_callbacks(
            on_export_charuco=self._export_charuco,
            on_export_apriltag=self._export_apriltag,
        )
    
    def set_tracking_callbacks(
        self,
        on_start: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
    ) -> None:
        """Set external callbacks for tracking lifecycle."""
        self._on_start_tracking = on_start
        self._on_stop_tracking = on_stop

    def load_calibration(self, calibration: object) -> None:
        """
        Load an existing calibration.
        
        Args:
            calibration: MultiCameraCalibration object
        """
        # Load into calibration panel (if supported)
        # self.calibration_panel.load_calibration(calibration)
        
        # Update skeleton viewer with camera positions
        camera_transforms = {}
        for cam_id, data in calibration.cameras.items():
            if 'extrinsics' in data:
                # Get transform matrix
                T = np.array(data['extrinsics']['transform_matrix'])
                camera_transforms[cam_id] = T
        
        if camera_transforms:
            self.skeleton_viewer.set_camera_positions(camera_transforms)
            calibration_log.info(f"Loaded {len(camera_transforms)} cameras into 3D viewer")
    
    def setup(self) -> None:
        """Initialize DearPyGui context and window."""
        dpg.create_context()
        dpg.create_viewport(title=self.title, width=self.width, height=self.height)
        
        # Connect OSC
        self.osc_sender.connect()
        self.osc_status.on_connect()
        
        # Start idle OSC loop
        self._idle_osc_thread = threading.Thread(target=self._idle_osc_loop, daemon=True)
        self._idle_osc_thread.start()
        
        # Texture registry
        self._texture_registry = dpg.add_texture_registry()
        
        # Skeleton viewer texture
        skeleton_w, skeleton_h = self.skeleton_viewer.size
        placeholder = self.skeleton_viewer.render_placeholder()
        self._skeleton_texture = dpg.add_dynamic_texture(
            width=skeleton_w,
            height=skeleton_h,
            default_value=placeholder.flatten().tolist(),
            parent=self._texture_registry,
            tag="skeleton_texture"
        )
        
        # Initialize viewer with T-Pose
        tpose = get_tpose()
        self._current_pose = tpose # Store for OSC loop
        self.update_skeleton_view(tpose['positions'], tpose['valid'])
        
        self._setup_theme()
        self._create_main_window()
        self._create_floating_windows()
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def _setup_theme(self) -> None:
        """Setup global theme."""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 6)
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 28, 35))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 45, 55))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (55, 65, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (75, 90, 110))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (60, 140, 200))
        
        dpg.bind_theme(global_theme)
        
        # Warning button theme (yellow)
        with dpg.theme(tag="warning_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (180, 140, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 160, 60))
        
        # Success button theme (green)
        with dpg.theme(tag="success_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 150, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 180, 100))
    
    def _create_main_window(self) -> None:
        """Create the unified main window."""
        with dpg.window(label="VoxelVR", tag="main_window", no_title_bar=True):
            # Header bar
            with dpg.group(horizontal=True):
                dpg.add_text("VoxelVR", color=(80, 180, 255), tag="logo_text")
                dpg.add_spacer(width=30)
                
                # Calibration status
                dpg.add_text("Calibration:", color=(150, 150, 150))
                dpg.add_text("Not Started", tag="calib_status", color=(200, 200, 100))
                
                dpg.add_spacer(width=20)
                
                # OSC status
                dpg.add_text("OSC:", color=(150, 150, 150))
                dpg.add_text("Disconnected", tag="osc_status", color=(128, 128, 128))
                
                dpg.add_spacer()
                
                # FPS counter
                dpg.add_text("FPS:", color=(150, 150, 150))
                dpg.add_text("--", tag="fps_counter", color=(100, 200, 100))
                
                dpg.add_spacer(width=10)
                
                # Floating window toggles
                dpg.add_button(label="📊 Debug", callback=self._toggle_debug_window)
                dpg.add_button(label="⚡ Perf", callback=self._toggle_perf_window)
            
            dpg.add_separator()
            
            # Main content split: camera grid left, controls right
            with dpg.group(horizontal=True):
                # Camera grid (left side)
                with dpg.child_window(tag="camera_panel", width=900, height=-1):
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="🔍 Detect Cameras",
                            callback=self._on_detect_cameras_click,
                        )
                        dpg.add_text("", tag="camera_count")
                        
                        dpg.add_spacer(width=20)
                        
                        # Zoom controls
                        dpg.add_button(label="−", callback=self._on_zoom_out, width=30)
                        dpg.add_button(label="+", callback=self._on_zoom_in, width=30)
                        dpg.add_button(label="⊡ Fit", callback=self._on_autofit)
                        dpg.add_text("100%", tag="zoom_label")
                        
                        dpg.add_spacer(width=20)
                        
                        # Tracker settings button
                        dpg.add_button(
                            label="🎮 Trackers",
                            callback=self._toggle_tracker_window,
                        )
                    
                    dpg.add_separator()
                    
                    with dpg.child_window(tag="camera_grid", autosize_x=True, height=500):
                        dpg.add_text("Click 'Detect Cameras' to begin.", tag="no_cameras_text")
                    
                    dpg.add_separator()
                    
                    # 3D Skeleton viewer
                    dpg.add_text("3D Skeleton Preview", color=(150, 150, 150))
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(
                            label="Auto-Rotate",
                            default_value=True,
                            callback=lambda s, a: self.skeleton_viewer.set_auto_rotate(a),
                        )
                        dpg.add_slider_float(
                            label="Speed",
                            default_value=1.0,
                            min_value=0.1,
                            max_value=5.0,
                            width=100,
                            callback=lambda s, a: self.skeleton_viewer.set_rotation_speed(a),
                        )
                    dpg.add_image("skeleton_texture", tag="skeleton_display")
                
                # Control panel (right side)
                with dpg.child_window(tag="control_panel", autosize_x=True, height=-1):
                    # Calibration section
                    dpg.add_text("ChArUco Calibration", color=(150, 200, 255))
                    dpg.add_separator()
                    
                    # Pairwise progress grid placeholder
                    with dpg.child_window(tag="pairwise_grid", height=180, border=True):
                        dpg.add_text("Detect cameras to show progress grid", tag="pairwise_placeholder")
                    
                    # Connectivity status
                    dpg.add_text("○ Not Connected", tag="connectivity_status", color=(200, 100, 100))
                    
                    dpg.add_spacer(height=5)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Export Board PDF",
                            callback=self._on_export_charuco_click,
                        )
                    
                    dpg.add_separator()
                    
                    # Tracking section
                    dpg.add_text("Tracking Controls", color=(150, 200, 255))
                    dpg.add_separator()
                    
                    dpg.add_button(
                        label="▶ Start Tracking",
                        tag="tracking_btn",
                        callback=self._on_tracking_toggle,
                        width=-1,
                        height=50,
                    )
                    
                    dpg.add_spacer(height=10)
                    
                    # Post-calibration section
                    dpg.add_text("Post-Calibration", color=(150, 200, 255))
                    dpg.add_separator()
                    
                    dpg.add_button(
                        label="🎯 Calibrate Origin",
                        tag="postcalib_btn",
                        callback=self._on_postcalib_click,
                        width=-1,
                        height=40,
                    )
                    dpg.add_text(
                        "Click to calibrate",
                        tag="postcalib_status",
                        color=(120, 120, 120),
                    )
                    
                    # Y-rotation adjustment
                    dpg.add_text("Yaw Offset", color=(120, 120, 120))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="-90°", callback=lambda: self._adjust_yaw(-90), width=45)
                        dpg.add_button(label="-45°", callback=lambda: self._adjust_yaw(-45), width=45)
                        dpg.add_button(label="-15°", callback=lambda: self._adjust_yaw(-15), width=45)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Reset", callback=lambda: self._reset_yaw(), width=45)
                        dpg.add_button(label="+15°", callback=lambda: self._adjust_yaw(15), width=45)
                        dpg.add_button(label="+45°", callback=lambda: self._adjust_yaw(45), width=45)
                        dpg.add_button(label="+90°", callback=lambda: self._adjust_yaw(90), width=45)
                    dpg.add_text("0°", tag="yaw_offset_label", color=(100, 200, 100))
                    
                    dpg.add_spacer(height=10)
                    
                    # AprilTag toggle
                    dpg.add_checkbox(
                        label="🎯 AprilTag Precision Mode",
                        tag="apriltag_toggle",
                        callback=self._on_apriltag_toggle,
                    )
                    dpg.add_text(
                        "Enables wearable marker tracking for improved accuracy",
                        color=(120, 120, 120),
                        wrap=200,
                    )
                    
                    with dpg.group(horizontal=True, show=False, tag="apriltag_export_group"):
                        dpg.add_button(
                            label="Export AprilTag Sheet",
                            callback=self._on_export_apriltag_click,
                        )
                    

                    dpg.add_separator()
                    
                    # OSC Settings
                    dpg.add_text("OSC Settings", color=(150, 200, 255))
                    dpg.add_separator()
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("IP:")
                        dpg.add_input_text(
                            tag="osc_ip_input",
                            default_value="127.0.0.1",
                            width=120,
                            callback=self._on_osc_config_change,
                        )
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Port:")
                        dpg.add_input_int(
                            tag="osc_port_input",
                            default_value=9000,
                            width=100,
                            callback=self._on_osc_config_change,
                        )
        
        dpg.set_primary_window("main_window", True)
    
    def _create_floating_windows(self) -> None:
        """Create floating debug and performance windows."""
        # Debug window
        with dpg.window(
            label="Debug Panel",
            tag="debug_window",
            width=400,
            height=300,
            pos=(50, 100),
            show=False,
            on_close=lambda: self.view.set_debug_window_open(False),
        ):
            dpg.add_text("Filter Parameters", color=(150, 150, 150))
            dpg.add_separator()
            
            ranges = self.debug_panel.get_slider_ranges()
            
            dpg.add_text("Min Cutoff (smoothing)")
            dpg.add_slider_float(
                tag="debug_min_cutoff",
                min_value=ranges['min_cutoff'][0],
                max_value=ranges['min_cutoff'][1],
                default_value=ranges['min_cutoff'][2],
                width=-1,
            )
            
            dpg.add_text("Beta (responsiveness)")
            dpg.add_slider_float(
                tag="debug_beta",
                min_value=ranges['beta'][0],
                max_value=ranges['beta'][1],
                default_value=ranges['beta'][2],
                width=-1,
            )
            
            dpg.add_text("D Cutoff")
            dpg.add_slider_float(
                tag="debug_d_cutoff",
                min_value=ranges['d_cutoff'][0],
                max_value=ranges['d_cutoff'][1],
                default_value=ranges['d_cutoff'][2],
                width=-1,
            )
        
        # Performance window
        with dpg.window(
            label="Performance",
            tag="perf_window",
            width=300,
            height=200,
            pos=(500, 100),
            show=False,
            on_close=lambda: self.view.set_performance_window_open(False),
        ):
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Total FPS")
                    dpg.add_text("--", tag="perf_total_fps", color=(100, 200, 100))
                
                dpg.add_spacer(width=20)
                
                with dpg.group():
                    dpg.add_text("Latency")
                    dpg.add_text("-- ms", tag="perf_latency", color=(100, 200, 100))
            
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Valid Joints")
                    dpg.add_text("--/17", tag="perf_joints", color=(100, 200, 100))
                
                dpg.add_spacer(width=20)
                
                with dpg.group():
                    dpg.add_text("Jitter")
                    dpg.add_text("-- mm", tag="perf_jitter", color=(100, 200, 100))
        
        # Tracker settings window
        with dpg.window(
            label="Tracker Settings",
            tag="tracker_window",
            width=350,
            height=400,
            pos=(200, 150),
            show=False,
        ):
            dpg.add_text("Tracker Preset", color=(150, 200, 255))
            dpg.add_separator()
            
            # Preset buttons
            with dpg.group(horizontal=True):
                dpg.add_button(label="Full Body", callback=lambda: self._apply_tracker_preset("full_body"))
                dpg.add_button(label="Upper", callback=lambda: self._apply_tracker_preset("upper_body"))
                dpg.add_button(label="Lower", callback=lambda: self._apply_tracker_preset("lower_body"))
                dpg.add_button(label="Legs+Waist", callback=lambda: self._apply_tracker_preset("legs_waist"))
            
            dpg.add_spacer(height=10)
            dpg.add_text("Current:", color=(120, 120, 120))
            dpg.add_text("Full Body", tag="current_preset", color=(100, 200, 100))
            
            dpg.add_separator()
            dpg.add_text("Custom Joint Selection", color=(150, 200, 255))
            dpg.add_separator()
            
            # Joint checkboxes in a grid layout
            # Head row
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Nose", tag="joint_0", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="L.Eye", tag="joint_1", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="R.Eye", tag="joint_2", default_value=True, callback=self._on_joint_toggle)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="L.Ear", tag="joint_3", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="R.Ear", tag="joint_4", default_value=True, callback=self._on_joint_toggle)
            
            dpg.add_spacer(height=5)
            
            # Arms row
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="L.Shoulder", tag="joint_5", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="R.Shoulder", tag="joint_6", default_value=True, callback=self._on_joint_toggle)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="L.Elbow", tag="joint_7", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="R.Elbow", tag="joint_8", default_value=True, callback=self._on_joint_toggle)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="L.Wrist", tag="joint_9", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="R.Wrist", tag="joint_10", default_value=True, callback=self._on_joint_toggle)
            
            dpg.add_spacer(height=5)
            
            # Legs row
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="L.Hip", tag="joint_11", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="R.Hip", tag="joint_12", default_value=True, callback=self._on_joint_toggle)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="L.Knee", tag="joint_13", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="R.Knee", tag="joint_14", default_value=True, callback=self._on_joint_toggle)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="L.Ankle", tag="joint_15", default_value=True, callback=self._on_joint_toggle)
                dpg.add_checkbox(label="R.Ankle", tag="joint_16", default_value=True, callback=self._on_joint_toggle)
            
            dpg.add_separator()
            dpg.add_text("", tag="joints_enabled_count", color=(120, 120, 120))
    
    # =========================================================================
    # Camera operations
    # =========================================================================
    
    def _detect_cameras(self) -> List[int]:
        """Detect available cameras."""
        from ..capture.manager import CameraManager
        return CameraManager.detect_cameras()
    
    def _on_detect_cameras_click(self, sender, app_data) -> None:
        """Handle detect cameras button."""
        cameras = self.view.detect_cameras()
        
        if cameras:
            dpg.set_value("camera_count", f"{len(cameras)} camera(s) detected")
            self._update_camera_grid(cameras)
            self._start_preview(cameras)
            
            # Initialize calibration panel with cameras
            self.calibration_panel.set_cameras(cameras)
            
            # Create the pairwise progress grid
            self._create_pairwise_grid(cameras)
            
            # Auto-start calibration capture
            self.calibration_panel.begin_calibration()
        else:
            dpg.set_value("camera_count", "No cameras found")
    
    def _update_camera_grid(self, camera_ids: List[int]) -> None:
        """Update camera grid display with proper row/column layout."""
        with self._grid_lock:
            dpg.delete_item("camera_grid", children_only=True)
            
            if not camera_ids:
                dpg.add_text("No cameras detected.", parent="camera_grid")
                return
            
            rows, cols = self.view.get_camera_grid_layout()
            scaled_w, scaled_h = self._get_scaled_camera_size()
            
            # Create textures first (recreate if size changed)
            for cam_id in camera_ids:
                texture_tag = f"cam_texture_{cam_id}"
                
                # Delete old texture if exists (size may have changed)
                if dpg.does_item_exist(texture_tag):
                    dpg.delete_item(texture_tag)
                    if cam_id in self._camera_textures:
                        del self._camera_textures[cam_id]
                
                # Create placeholder texture with scaled size
                placeholder = np.zeros((scaled_h, scaled_w, 4), dtype=np.float32)
                placeholder[:, :, 3] = 1.0
                
                self._camera_textures[cam_id] = dpg.add_dynamic_texture(
                    width=scaled_w,
                    height=scaled_h,
                    default_value=placeholder.flatten().tolist(),
                    parent=self._texture_registry,
                    tag=texture_tag,
                )
            
            # Arrange cameras in grid
            cam_idx = 0
            for row in range(rows):
                # Create horizontal group for each row
                with dpg.group(horizontal=True, parent="camera_grid"):
                    for col in range(cols):
                        if cam_idx >= len(camera_ids):
                            break
                        
                        cam_id = camera_ids[cam_idx]
                        texture_tag = f"cam_texture_{cam_id}"
                        
                        # Camera cell with label and image
                        with dpg.group():
                            dpg.add_text(f"Camera {cam_id}")
                            dpg.add_image(texture_tag)
                        
                        # Spacer between cameras in row (except last)
                        if col < cols - 1 and cam_idx < len(camera_ids) - 1:
                            dpg.add_spacer(width=10)
                        
                        cam_idx += 1
    
    def update_camera_frame(self, camera_id: int, frame: np.ndarray) -> None:
        """Update a camera's texture with new frame (optimized)."""
        texture_tag = f"cam_texture_{camera_id}"
        
        # Helper to avoid lock overhead if obviously missing
        # (Though race condition can still happen after this check, hence the lock below)
        if not dpg.does_item_exist(texture_tag):
            return
        
        # Get target size
        scaled_w, scaled_h = self._get_scaled_camera_size()
        
        # Use pre-allocated buffer if available (avoids repeated allocation)
        buffer_key = (scaled_w, scaled_h)
        if not hasattr(self, '_texture_buffers'):
            self._texture_buffers = {}
        
        if buffer_key not in self._texture_buffers:
            # Pre-allocate float32 RGBA buffer
            self._texture_buffers[buffer_key] = np.empty((scaled_h, scaled_w, 4), dtype=np.float32)
        
        buffer = self._texture_buffers[buffer_key]
        
        # Resize directly (uses OpenCV's optimized routines)
        frame_resized = cv2.resize(frame, (scaled_w, scaled_h))
        
        # Convert BGR to RGBA and normalize in one pass
        # This is faster than separate cvtColor + division
        buffer[:, :, 0] = frame_resized[:, :, 2] / 255.0  # R from B
        buffer[:, :, 1] = frame_resized[:, :, 1] / 255.0  # G from G  
        buffer[:, :, 2] = frame_resized[:, :, 0] / 255.0  # B from R
        buffer[:, :, 3] = 1.0  # Alpha
        
        # Update texture with contiguous flattened data
        # PROTECTED UPDATE: Lock against grid rebuilding and catch internal errors
        with self._grid_lock:
            try:
                if dpg.does_item_exist(texture_tag):
                    dpg.set_value(texture_tag, buffer.ravel())
            except SystemError:
                # Can happen if item is deleted during set_value call even with our checks
                pass
            except Exception as e:
                # Log other unexpected errors but don't crash
                 print(f"Texture update error: {e}")
    
    def _start_preview(self, camera_ids: List[int]) -> None:
        """Start camera preview."""
        self._stop_preview()
        
        from ..capture.manager import CameraManager
        from ..config import CameraConfig
        from ..capture.camera import Camera
        
        configs = []
        for cam_id in camera_ids:
            w, h, fps = Camera.get_best_configuration(cam_id, target_fps=30)
            configs.append(CameraConfig(id=cam_id, resolution=(w, h), fps=fps))
        
        self.camera_manager = CameraManager(configs)
        if self.camera_manager.start_all():
            self.preview_active = True
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
    
    def _stop_preview(self) -> None:
        """Stop camera preview."""
        self.preview_active = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1.0)
            self.preview_thread = None
        
        if self.camera_manager:
            self.camera_manager.stop_all()
            self.camera_manager = None
    
    def update_2d_detections(self, detections: Dict[int, object]) -> None:
        """Update the latest 2D detections for visualization without re-inference."""
        self._latest_detections = detections
        self._detection_timestamp = time.time()  # Track freshness

    def _preview_loop(self) -> None:
        """Camera preview update loop with ChArUco detection and 2D pose overlay."""
        from ..calibration.charuco import detect_charuco
        from ..pose.detector_2d import PoseDetector2D
        
        # Lazy-load pose detector for 2D overlay
        # Used when tracking is active but tracking thread isn't providing detections
        pose_detector = None
        self._latest_detections = {}
        self._detection_timestamp = 0  # Track when we last received external detections
        
        # Performance tracking
        frame_count = 0
        fps_start_time = time.time()
        timing_stats = {
            'capture': [],
            'charuco': [],
            'pose_detect': [],
            'skeleton_draw': [],
            'texture_update': [],
            'calib_process': [],
            'loop_total': [],
        }
        
        while self.preview_active and dpg.is_dearpygui_running():
            loop_start = time.time()
            
            if not self.camera_manager:
                break
            
            # Time: Frame capture
            t0 = time.time()
            frames = self.camera_manager.get_all_latest_frames()
            timing_stats['capture'].append(time.time() - t0)
            
            if frames:
                detections = {}
                is_tracking = self.view.state.tracking_mode in (TrackingMode.RUNNING, TrackingMode.STARTING)
                
                # Check if external detections are recent (within 0.5 seconds)
                external_detections_fresh = (time.time() - self._detection_timestamp) < 0.5
                
                charuco_time = 0
                pose_time = 0
                draw_time = 0
                texture_time = 0
                
                for cam_id, frame in frames.items():
                    display_frame = frame.image.copy()
                    
                    # Only run ChArUco detection when NOT tracking (calibration phase)
                    # Skip during tracking to save CPU (~10-50ms per frame)
                    if not is_tracking:
                        t1 = time.time()
                        result = detect_charuco(display_frame, self._charuco_board, self._aruco_dict)
                        charuco_time += time.time() - t1
                        
                        detections[cam_id] = {
                            'success': result['success'],
                            'corners': result['corners'],
                            'ids': result['ids'],
                            'frame': frame.image
                        }
                        
                        # Use ChArUco annotated frame
                        display_frame = result['image_with_markers']
                    
                    # If tracking, draw 2D pose using available data
                    if is_tracking:
                        pose_result = None
                        
                        # Case 1: Use externally provided detections (from tracking thread)
                        if external_detections_fresh and cam_id in self._latest_detections:
                            pose_result = self._latest_detections[cam_id]
                        
                        # Case 2: Run local inference (fallback when no tracking thread)
                        else:
                            # Initialize detector if needed (lazy load)
                            if pose_detector is None:
                                backend = "cpu" if self.force_cpu else "auto"
                                log_info(f"Preview loop: Loading pose detector (backend={backend})...")
                                pose_detector = PoseDetector2D(confidence_threshold=0.3, backend=backend)
                                if not pose_detector.load_model():
                                    log_error("Failed to load pose detector for preview overlay")
                                    pose_detector = False  # Mark as failed to avoid retrying
                            
                            # Run inference if detector is available
                            t2 = time.time()
                            if pose_detector and pose_detector is not False:
                                pose_result = pose_detector.detect(frame.image, camera_id=cam_id)
                            pose_time += time.time() - t2
                        
                        t3 = time.time()
                        if pose_result and hasattr(pose_result, 'positions'):
                            # Draw 2D skeleton on frame
                            display_frame = self._draw_2d_skeleton(display_frame, pose_result)
                            
                            # Add detection indicator
                            valid_joints = int(np.sum(pose_result.confidences > 0.3))
                            self._draw_detection_indicator(display_frame, valid_joints)
                        else:
                            # Show indicator even when no detection
                            self._draw_detection_indicator(display_frame, 0)
                        draw_time += time.time() - t3
                    
                    # Time: Texture update
                    t4 = time.time()
                    self.update_camera_frame(cam_id, display_frame)
                    texture_time += time.time() - t4
                    
                    # Update camera visibility in view state
                    self.view.update_camera_frame(
                        cam_id, frame.image,
                        board_visible=result['success']
                    )
                
                timing_stats['charuco'].append(charuco_time)
                timing_stats['pose_detect'].append(pose_time)
                timing_stats['skeleton_draw'].append(draw_time)
                timing_stats['texture_update'].append(texture_time)
                
                # Time: Calibration processing
                t5 = time.time()
                capture_result = self.calibration_panel.process_frame_detections(detections)
                timing_stats['calib_process'].append(time.time() - t5)
                
                # Handle recording if enabled
                if self._recording_enabled:
                    self._record_frames(frames)
            
            # Track loop time
            loop_time = time.time() - loop_start
            timing_stats['loop_total'].append(loop_time)
            frame_count += 1
            
            # Log timing stats every 60 frames (~2 seconds)
            if frame_count % 60 == 0:
                elapsed = time.time() - fps_start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Calculate averages from last 60 samples
                def avg_ms(data):
                    recent = data[-60:] if len(data) >= 60 else data
                    return (sum(recent) / len(recent) * 1000) if recent else 0
                
                log_info(f"[PERF] Preview FPS: {fps:.1f} | Loop: {avg_ms(timing_stats['loop_total']):.1f}ms")
                log_debug(f"[PERF] Capture: {avg_ms(timing_stats['capture']):.1f}ms | "
                         f"ChArUco: {avg_ms(timing_stats['charuco']):.1f}ms | "
                         f"Pose: {avg_ms(timing_stats['pose_detect']):.1f}ms | "
                         f"Draw: {avg_ms(timing_stats['skeleton_draw']):.1f}ms | "
                         f"Texture: {avg_ms(timing_stats['texture_update']):.1f}ms | "
                         f"Calib: {avg_ms(timing_stats['calib_process']):.1f}ms")
                
                # Reset counters
                fps_start_time = time.time()
                frame_count = 0
            
            time.sleep(0.033)  # ~30 FPS target
    
    def _draw_2d_skeleton(self, frame: np.ndarray, pose_result) -> np.ndarray:
        """Draw 2D skeleton overlay on frame."""
        # COCO skeleton connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6),  # Shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Torso
            (11, 12),  # Hips
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]
        
        output = frame.copy()
        positions = pose_result.positions  # (17, 2)
        confidences = pose_result.confidences  # (17,)
        
        # Draw connections
        for i, j in connections:
            if confidences[i] > 0.3 and confidences[j] > 0.3:
                pt1 = tuple(map(int, positions[i]))
                pt2 = tuple(map(int, positions[j]))
                cv2.line(output, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for i, (pt, conf) in enumerate(zip(positions, confidences)):
            if conf > 0.3:
                center = tuple(map(int, pt))
                color = (0, 255, 255) if conf > 0.5 else (0, 165, 255)
                cv2.circle(output, center, 4, color, -1)
        
        return output
    
    def _draw_detection_indicator(self, frame: np.ndarray, valid_joints: int) -> None:
        """Draw detection quality indicator on frame (in-place)."""
        # Determine color based on joint count
        if valid_joints >= 12:
            color = (0, 200, 0)  # Green - good detection
            status = "GOOD"
        elif valid_joints >= 6:
            color = (0, 200, 200)  # Yellow - partial detection  
            status = "PARTIAL"
        elif valid_joints > 0:
            color = (0, 100, 200)  # Orange - weak detection
            status = "WEAK"
        else:
            color = (0, 0, 180)  # Red - no detection
            status = "NONE"
        
        # Draw indicator box in top-left corner
        text = f"Joints: {valid_joints}/17 ({status})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background box
        cv2.rectangle(frame, (5, 5), (text_w + 15, text_h + baseline + 15), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (text_w + 15, text_h + baseline + 15), color, 2)
        
        # Text
        cv2.putText(frame, text, (10, text_h + 10), font, font_scale, color, thickness)
    
    # =========================================================================
    # Calibration callbacks
    # =========================================================================
    
    def _on_calibration_progress(self, state) -> None:
        """Handle calibration panel progress updates."""
        # Update view state with progress
        progress = self.calibration_panel.get_progress_summary()
        
        # Build per-camera progress dict
        per_camera = {
            cam_id: data['intrinsic_frames']
            for cam_id, data in progress['cameras'].items()
        }
        
        # Build pairwise progress dict
        pairwise = {
            pair: data['frames']
            for pair, data in progress['pairwise'].items()
        }
        
        # Update unified view state
        self.view.update_calibration_progress(
            per_camera_progress=per_camera,
            pairwise_progress=pairwise,
            is_connected=progress['pairwise_connected'],
            is_calibrated=progress['all_ready'],
        )
        
        # Update UI
        self._update_pairwise_grid_ui()
        
        # Check for completion
        if progress['all_ready'] and not self.view.state.calibration.is_calibrated:
            calibration_log.info("Calibration complete! Ready for tracking.")
            
            # Load calibration into viewer immediately
            if self.calibration_panel.state.extrinsics.result:
                self.load_calibration(self.calibration_panel.state.extrinsics.result)
                print(f"Loaded calibration into 3D viewer")
            
            self._on_calibration_complete()
    
    def _update_pairwise_grid_ui(self) -> None:
        """Update the pairwise progress grid display."""
        progress = self.calibration_panel.get_progress_summary()
        
        # Update per-camera intrinsic bars with state colors
        for cam_id, data in progress['cameras'].items():
            tag = f"intrinsic_{cam_id}"
            label_tag = f"intrinsic_label_{cam_id}"
            
            if dpg.does_item_exist(tag):
                percent = data['intrinsic_percent'] / 100.0
                dpg.set_value(tag, percent)
                
                # Get status from calibration panel state
                status = self.calibration_panel._state.cameras.get(cam_id)
                if status:
                    if status.intrinsic_failed:
                        # Red for failed
                        dpg.configure_item(tag, overlay="FAILED")
                    elif status.intrinsic_computing:
                        # Show computing with retry count
                        if status.intrinsic_retry_count > 0:
                            dpg.configure_item(tag, overlay=f"Retry {status.intrinsic_retry_count}")
                        else:
                            dpg.configure_item(tag, overlay="...")
                    elif status.intrinsic_complete:
                        dpg.configure_item(tag, overlay="OK")
                    elif status.intrinsic_retry_count > 0:
                        # Between retries - show retry count
                        dpg.configure_item(tag, overlay=f"#{status.intrinsic_retry_count}")
                    else:
                        dpg.configure_item(tag, overlay="")
        
        # Update pairwise bars
        for pair, data in progress['pairwise'].items():
            cam_a, cam_b = pair
            tag = f"pair_{cam_a}_{cam_b}"
            if dpg.does_item_exist(tag):
                percent = data['percent'] / 100.0
                dpg.set_value(tag, percent)
        
        # Update connectivity status
        if dpg.does_item_exist("connectivity_status"):
            if progress['pairwise_connected']:
                dpg.set_value("connectivity_status", "● Connected")
                dpg.configure_item("connectivity_status", color=(100, 200, 100))
            else:
                disconnected = progress['pairwise_disconnected_cameras']
                if disconnected:
                    dpg.set_value("connectivity_status", f"○ Cameras {disconnected} disconnected")
                else:
                    dpg.set_value("connectivity_status", "○ Not Connected")
                dpg.configure_item("connectivity_status", color=(200, 100, 100))
            
    def _on_calibration_complete(self) -> None:
        """Handle calibration completion - transition to tracking."""
        # Get final calibration from state
        calibration = self.calibration_panel.state.extrinsics.result
        
        if calibration:
            # Update skeleton viewer with camera positions
            self.load_calibration(calibration)
            print(f"Loaded calibration into 3D viewer")
            
            # Auto-save?
            # calibration.save("calibration.json")
            
        # Update status text
        dpg.set_value("calib_status", "Calibrated")
        dpg.configure_item("calib_status", color=(100, 200, 100))
        
        # Enable tracking button
        dpg.configure_item("tracking_btn", enabled=True, label="▶ Start Tracking")
    
    def _create_pairwise_grid(self, camera_ids: List[int]) -> None:
        """Create or update the pairwise progress grid."""
        # Clear existing grid content
        if dpg.does_item_exist("pairwise_placeholder"):
            dpg.delete_item("pairwise_placeholder")
        if dpg.does_item_exist("progress_table"):
            dpg.delete_item("progress_table")
        
        n = len(camera_ids)
        if n == 0:
            dpg.add_text("No cameras detected", parent="pairwise_grid", tag="pairwise_placeholder")
            return
        
        # Create table
        with dpg.table(tag="progress_table", parent="pairwise_grid", 
                       header_row=True, borders_innerH=True, borders_innerV=True):
            # Columns
            dpg.add_table_column(label="")  # Corner label column
            for cam_id in camera_ids:
                dpg.add_table_column(label=f"Cam {cam_id}", width_fixed=True, init_width_or_weight=65)
            
            # Rows
            for i, cam_a in enumerate(camera_ids):
                with dpg.table_row():
                    dpg.add_text(f"Cam {cam_a}")  # Row label
                    
                    for j, cam_b in enumerate(camera_ids):
                        if i == j:
                            # Diagonal: intrinsic progress
                            dpg.add_progress_bar(
                                tag=f"intrinsic_{cam_a}",
                                default_value=0.0,
                                width=55,
                            )
                        elif i < j:
                            # Upper triangle: pairwise extrinsic progress
                            dpg.add_progress_bar(
                                tag=f"pair_{cam_a}_{cam_b}",
                                default_value=0.0,
                                width=55,
                            )
                        else:
                            # Lower triangle: skip (symmetric)
                            dpg.add_text("-")
    
    def _update_calibration_ui(self) -> None:
        """Update calibration status display."""
        status_text = self.view.get_calibration_status_text()
        if dpg.does_item_exist("calib_status"):
            dpg.set_value("calib_status", status_text)
    
    def _on_export_charuco_click(self, sender, app_data) -> None:
        """Handle export ChArUco board."""
        from ..config import CONFIG_DIR
        output_path = CONFIG_DIR / "calibration" / "charuco_board.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.view.export_charuco_pdf(output_path)
        print(f"Exported ChArUco board to: {output_path}")
    
    def _export_charuco(self, path: Path) -> bool:
        """Export ChArUco board PDF."""
        from ..calibration.charuco import export_charuco_board
        try:
            export_charuco_board(path)
            return True
        except Exception as e:
            print(f"Failed to export ChArUco: {e}")
            return False
    
    def _export_apriltag(self, path: Path) -> bool:
        """Export AprilTag sheet PDF."""
        from ..calibration.apriltag import export_apriltag_sheet_pdf
        return export_apriltag_sheet_pdf(path)
    
    # =========================================================================
    # Tracking callbacks
    # =========================================================================
    
    def _on_tracking_toggle(self, sender, app_data) -> None:
        """Handle tracking start/stop toggle."""
        current_mode = self.view.state.tracking_mode
        
        if current_mode in (TrackingMode.RUNNING, TrackingMode.STARTING):
            # Stop tracking
            self.view.state.tracking_mode = TrackingMode.STOPPED
            self.view._notify_state_change()
            dpg.set_item_label("tracking_btn", "▶ Start Tracking")
            log_info("Tracking stopped")
            if self._on_stop_tracking:
                self._on_stop_tracking()
        else:
            # Start tracking - go directly to RUNNING for immediate feedback
            self.view.state.tracking_mode = TrackingMode.RUNNING
            self.view._notify_state_change()
            dpg.set_item_label("tracking_btn", "⏹ Stop Tracking")
            
            # Show warning if not calibrated
            if not self.view.state.calibration.is_calibrated:
                log_warn("Tracking started without calibration - skeleton overlay visible for debugging")
            else:
                log_info("Tracking started")
            
            if self._on_start_tracking:
                self._on_start_tracking()
    
    def _on_tracking_start(self) -> None:
        """Called when tracking starts."""
        if self._on_start_tracking:
            self._on_start_tracking()
    
    def _on_tracking_stop(self) -> None:
        """Called when tracking stops."""
        if self._on_stop_tracking:
            self._on_stop_tracking()
    
    def _on_apriltag_toggle(self, sender, app_data) -> None:
        """Handle AprilTag mode toggle."""
        self.view.set_apriltags_enabled(app_data)
        dpg.configure_item("apriltag_export_group", show=app_data)
    
    def _on_postcalib_click(self, sender=None, app_data=None) -> None:
        """Handle post-calibration button click."""
        from ..transport.post_calibration import PostCalibrationState
        
        if self.post_calibrator.state == PostCalibrationState.IDLE:
            # Start calibration
            self.post_calibrator.start()
            dpg.set_item_label("postcalib_btn", "⏳ Calibrating...")
        elif self.post_calibrator.state == PostCalibrationState.COMPLETE:
            # Reset and start again
            self.post_calibrator.reset()
            self.post_calibrator.start()
            dpg.set_item_label("postcalib_btn", "⏳ Calibrating...")
    
    def _adjust_yaw(self, delta: float) -> None:
        """Adjust yaw offset by delta degrees."""
        self.post_calibrator.adjust_yaw(delta)
        self._update_yaw_label()
    
    def _reset_yaw(self) -> None:
        """Reset yaw offset to zero."""
        self.post_calibrator.reset_yaw()
        self._update_yaw_label()
    
    def _update_yaw_label(self) -> None:
        """Update yaw offset label in UI."""
        yaw = self.post_calibrator.yaw_offset
        dpg.set_value("yaw_offset_label", f"{yaw:.0f}°")
    
    def _update_postcalib_status(self) -> None:
        """Update post-calibration status display. Call from main loop."""
        from ..transport.post_calibration import PostCalibrationState
        
        status_text = self.post_calibrator.get_status_text()
        dpg.set_value("postcalib_status", status_text)
        
        # Update button label when complete
        if self.post_calibrator.state == PostCalibrationState.COMPLETE:
            dpg.set_item_label("postcalib_btn", "✓ Re-Calibrate")
        elif self.post_calibrator.state == PostCalibrationState.IDLE:
            dpg.set_item_label("postcalib_btn", "🎯 Calibrate Origin")

    def _on_export_apriltag_click(self, sender, app_data) -> None:
        """Handle export AprilTag sheet."""
        from ..config import CONFIG_DIR
        output_path = CONFIG_DIR / "calibration" / "apriltag_sheet.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.view.export_apriltag_pdf(output_path)
        print(f"Exported AprilTag sheet to: {output_path}")
    
    def _on_osc_config_change(self, sender, app_data) -> None:
        """Handle OSC config changes."""
        ip = dpg.get_value("osc_ip_input")
        port = dpg.get_value("osc_port_input")
        self.view.set_osc_config(ip, port)
        
        # Update actual sender
        if self.osc_sender:
            self.osc_sender.disconnect()
            self.osc_sender.ip = ip
            self.osc_sender.port = port
            self.osc_sender.connect()
            print(f"OSC configuration updated: {ip}:{port}")
    
    # =========================================================================
    # Floating windows
    # =========================================================================
    
    def _toggle_debug_window(self, sender=None, app_data=None) -> None:
        """Toggle debug window visibility."""
        self.view.toggle_debug_window()
        dpg.configure_item("debug_window", show=self.view.state.debug_window_open)
    
    def _toggle_perf_window(self, sender=None, app_data=None) -> None:
        """Toggle performance window visibility."""
        self.view.toggle_performance_window()
        dpg.configure_item("perf_window", show=self.view.state.performance_window_open)
    
    def _toggle_tracker_window(self, sender=None, app_data=None) -> None:
        """Toggle tracker settings window visibility."""
        is_visible = dpg.is_item_shown("tracker_window")
        dpg.configure_item("tracker_window", show=not is_visible)
        self._update_joints_count_label()
    
    # =========================================================================
    # Zoom controls
    # =========================================================================
    
    def _on_zoom_in(self, sender=None, app_data=None) -> None:
        """Zoom in on camera grid."""
        self._camera_scale = min(2.0, self._camera_scale + 0.25)
        self._apply_camera_scale()
    
    def _on_zoom_out(self, sender=None, app_data=None) -> None:
        """Zoom out on camera grid."""
        self._camera_scale = max(0.25, self._camera_scale - 0.25)
        self._apply_camera_scale()
    
    def _on_autofit(self, sender=None, app_data=None) -> None:
        """Auto-fit cameras to available space."""
        n_cameras = len(self.view.state.cameras)
        if n_cameras == 0:
            return
        
        # Get available space (camera_grid panel)
        available_width = 880  # Approximate usable width
        available_height = 480  # Approximate usable height
        
        rows, cols = self.view.get_camera_grid_layout()
        
        # Calculate max size per camera with margins
        margin = 15
        max_width = (available_width - margin * (cols + 1)) // cols
        max_height = (available_height - margin * (rows + 1)) // rows
        
        # Maintain 4:3 aspect ratio
        base_w, base_h = self._base_camera_size
        scale_w = max_width / base_w
        scale_h = max_height / base_h
        
        self._camera_scale = min(scale_w, scale_h, 2.0)
        self._camera_scale = max(0.25, self._camera_scale)
        self._apply_camera_scale()
    
    def _apply_camera_scale(self) -> None:
        """Apply current camera scale to grid."""
        percentage = int(self._camera_scale * 100)
        if dpg.does_item_exist("zoom_label"):
            dpg.set_value("zoom_label", f"{percentage}%")
        
        # Rebuild camera grid with new size
        camera_ids = list(self.view.state.cameras.keys())
        if camera_ids:
            self._update_camera_grid(camera_ids)
    
    def _get_scaled_camera_size(self) -> tuple:
        """Get camera display size based on current scale."""
        base_w, base_h = self._base_camera_size
        return (int(base_w * self._camera_scale), int(base_h * self._camera_scale))
    
    # =========================================================================
    # Tracker preset controls
    # =========================================================================
    
    def _apply_tracker_preset(self, preset_name: str) -> None:
        """Apply a tracker preset."""
        if preset_name in self._tracker_presets:
            self._enabled_joints = self._tracker_presets[preset_name].copy()
            self._tracker_preset = preset_name
            
            # Update checkboxes
            for i in range(17):
                tag = f"joint_{i}"
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, i in self._enabled_joints)
            
            # Update preset label
            preset_labels = {
                "full_body": "Full Body",
                "upper_body": "Upper Body",
                "lower_body": "Lower Body",
                "legs_waist": "Legs + Waist",
            }
            if dpg.does_item_exist("current_preset"):
                dpg.set_value("current_preset", preset_labels.get(preset_name, "Custom"))
            
            self._update_joints_count_label()
            print(f"Applied tracker preset: {preset_name} ({len(self._enabled_joints)} joints)")
    
    def _on_joint_toggle(self, sender, app_data) -> None:
        """Handle individual joint checkbox toggle."""
        # Extract joint index from sender tag (e.g., "joint_5" -> 5)
        tag = dpg.get_item_alias(sender) if dpg.get_item_alias(sender) else str(sender)
        if tag.startswith("joint_"):
            try:
                joint_idx = int(tag.split("_")[1])
                if app_data:
                    self._enabled_joints.add(joint_idx)
                else:
                    self._enabled_joints.discard(joint_idx)
                
                # Update to custom mode
                self._tracker_preset = "custom"
                if dpg.does_item_exist("current_preset"):
                    dpg.set_value("current_preset", "Custom")
                
                self._update_joints_count_label()
            except (ValueError, IndexError):
                pass
    
    def _update_joints_count_label(self) -> None:
        """Update the joints enabled count label."""
        if dpg.does_item_exist("joints_enabled_count"):
            count = len(self._enabled_joints)
            dpg.set_value("joints_enabled_count", f"{count}/17 joints enabled")
    
    # =========================================================================
    # Update methods
    # =========================================================================
    
    def update_performance(self, metrics: PerformanceMetrics) -> None:
        """Update performance display."""
        if dpg.does_item_exist("fps_counter"):
            dpg.set_value("fps_counter", f"{metrics.total_fps:.1f}")
        
        if dpg.does_item_exist("perf_total_fps"):
            dpg.set_value("perf_total_fps", f"{metrics.total_fps:.1f}")
        
        if dpg.does_item_exist("perf_latency"):
            dpg.set_value("perf_latency", f"{metrics.total_latency_ms:.1f} ms")
        
        if dpg.does_item_exist("perf_joints"):
            dpg.set_value("perf_joints", f"{metrics.num_valid_joints}/17")
        
        if dpg.does_item_exist("perf_jitter"):
            dpg.set_value("perf_jitter", f"{metrics.jitter_mm:.1f} mm")
    
    def update_skeleton_view(self, positions: np.ndarray, valid: np.ndarray) -> None:
        """Update 3D skeleton viewer."""
        if self._skeleton_texture is None:
            return
        
        frame = self.skeleton_viewer.render_skeleton(positions, valid)
        # frame is already float32 range [0, 1]
        dpg.set_value("skeleton_texture", frame.flatten().tolist())
    
    # =========================================================================
    # Compatibility shims for run_gui.py
    # =========================================================================
    
    @property
    def tracking_panel(self):
        """Compatibility shim for accessing tracking-related methods."""
        return self._TrackingPanelShim(self)
    
    class _TrackingPanelShim:
        """Minimal shim to provide tracking_panel interface for run_gui.py compatibility."""
        def __init__(self, app: 'UnifiedVoxelVRApp'):
            self.app = app
        
        def on_tracking_error(self, message: str) -> None:
            """Handle tracking error."""
            print(f"Tracking error: {message}")
        
        def on_tracking_started(self) -> None:
            """Handle tracking started."""
            log_info("Tracking started successfully")
        
        def on_tracking_stopped(self) -> None:
            """Handle tracking stopped."""
            log_info("Tracking stopped")
        
        def get_osc_config(self) -> tuple:
            """Get OSC IP and port from the unified app's inputs."""
            import dearpygui.dearpygui as dpg
            ip = "127.0.0.1"
            port = 9000
            if dpg.does_item_exist("osc_ip_input"):
                ip = dpg.get_value("osc_ip_input") or "127.0.0.1"
            if dpg.does_item_exist("osc_port_input"):
                port = dpg.get_value("osc_port_input") or 9000
            return (ip, port)
        
        def get_enabled_trackers(self) -> list:
            """Get list of enabled tracker names based on app's selected joints."""
            # Map joint indices to VRChat tracker names
            joint_to_tracker = {
                0: 'head',
                5: 'left_shoulder', 6: 'right_shoulder',
                7: 'left_elbow', 8: 'right_elbow',
                9: 'left_wrist', 10: 'right_wrist',
                11: 'left_hip', 12: 'right_hip',
                13: 'left_knee', 14: 'right_knee',
                15: 'left_foot', 16: 'right_foot',
            }
            enabled = []
            for joint_idx in self.app._enabled_joints:
                if joint_idx in joint_to_tracker:
                    enabled.append(joint_to_tracker[joint_idx])
            # Always include core trackers for VRChat
            core_trackers = ['hip', 'chest', 'left_foot', 'right_foot', 'left_knee', 'right_knee']
            for t in core_trackers:
                if t not in enabled:
                    enabled.append(t)
            return enabled
        
        def update_pose(self, positions, valid, confidences) -> None:
            """Update pose display."""
            self.app.update_skeleton_view(positions, valid)
        
        def update_status(self, fps: float, valid_joints: int, trackers_sending: int) -> None:
            """Update tracking status display."""
            pass
        
        @property
        def is_running(self) -> bool:
            return self.app.view.state.tracking_mode == TrackingMode.RUNNING

        def get_current_pose(self) -> Optional[Dict]:
            """Get current pose from app state."""
            return getattr(self.app, '_current_pose', None)
    
    @property 
    def state(self):
        """Compatibility shim for state access."""
        return self._StateShim(self)
    
    class _StateShim:
        """Minimal shim for state."""
        def __init__(self, app: 'UnifiedVoxelVRApp'):
            self.app = app
            self.is_running = True
    
    @property
    def osc_status(self):
        """Compatibility shim for OSC status."""
        return self._OscStatusShim()
    
    class _OscStatusShim:
        """Minimal shim for OSC status display."""
        def on_connect(self) -> None:
            osc_log.info("OSC connected")
        
        def on_disconnect(self) -> None:
            osc_log.info("OSC disconnected")
        
        def on_message_sent(self) -> None:
            pass  # Called frequently, don't log
    
    # Note: debug_panel is set in __init__ as a real DebugPanel instance
    # No shim needed since the actual debug_panel works correctly
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def enable_recording(self, output_dir: Path = None) -> None:
        """Enable time-synced recording of all camera streams."""
        from datetime import datetime
        
        if output_dir is None:
            output_dir = Path("dataset/recordings") / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        self._recording_dir = output_dir
        self._recording_enabled = True
        self._video_writers = {}
        self._recording_timestamps = []
        print(f"Recording enabled: {output_dir}")
    
    def _record_frames(self, frames: Dict) -> None:
        """Record frames from all cameras with synchronized timestamps."""
        if not self._recording_enabled or not self._recording_dir:
            return
        
        timestamp = time.time()
        
        for cam_id, frame in frames.items():
            if cam_id not in self._video_writers:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                path = self._recording_dir / f"camera_{cam_id}.mp4"
                h, w = frame.image.shape[:2]
                self._video_writers[cam_id] = cv2.VideoWriter(
                    str(path), fourcc, 30.0, (w, h)
                )
            self._video_writers[cam_id].write(frame.image)
        
        self._recording_timestamps.append(timestamp)
    
    def _cleanup_recording(self) -> None:
        """Cleanup recording resources and save timestamps."""
        if hasattr(self, '_video_writers'):
            for writer in self._video_writers.values():
                writer.release()
            self._video_writers.clear()
        
        if hasattr(self, '_recording_timestamps') and self._recording_dir:
            timestamps_file = self._recording_dir / "timestamps.txt"
            with open(timestamps_file, "w") as f:
                for ts in self._recording_timestamps:
                    f.write(f"{ts}\n")
            print(f"Saved {len(self._recording_timestamps)} frame timestamps")
            self._recording_timestamps.clear()
    
    def _idle_osc_loop(self) -> None:
        """
        Background loop to send T-Pose (or last valid pose) when tracking is not running.
        Also updates the local skeleton preview.
        """
        from ..transport.osc_sender import pose_to_trackers_with_rotations
        from ..transport.coordinate import create_default_transform, transform_pose_to_vrchat, CoordinateTransform
        
        osc_log.info("Starting idle OSC loop...")
        # T-Pose is already in Y-Up (VRChat-like) coordinates, so we don't need to flip Y.
        # Use identity rotation.
        coord_transform = CoordinateTransform(rotation=np.eye(3))
        
        print(f"OSC Idle Loop: Target {self.osc_sender.ip}:{self.osc_sender.port}")
        last_print = 0
        
        try:
            while not self._stop_event.is_set():
                # Check DPG status but don't exit loop immediately if false, just break if context destroyed
                # if not dpg.is_dearpygui_running():
                #      print("Idle OSC: DPG not running, exiting loop")
                #      break
                # ... (inner loop content)
                # Only run if tracking is NOT running (main tracking loop handles it otherwise)
                if self.view.state.tracking_mode not in (TrackingMode.RUNNING, TrackingMode.STARTING):
                    try:
                        # Get current pose from tracking panel (default is T-pose)
                        current_pose = self.tracking_panel.get_current_pose()
                        
                        if current_pose is None:
                            if time.time() - last_print > 5.0:
                                 print("OSC Idle: No current pose available")
                        else:
                            positions = current_pose['positions']
                            valid = current_pose['valid']
                            confidences = current_pose['confidences']
                            
                            # 1. Update local skeleton view
                            self.update_skeleton_view(positions, valid)
                            
                            # 2. Get active transform (post-calibrator if available)
                            postcalib_transform = self.post_calibrator.get_transform()
                            if postcalib_transform is not None:
                                active_transform = postcalib_transform
                            else:
                                active_transform = coord_transform
                            
                            # 3. Transform coordinates for VRChat
                            transformed, _ = transform_pose_to_vrchat(
                                positions,
                                active_transform,
                            )

                            
                            # 3. Convert to VRChat trackers
                            trackers = pose_to_trackers_with_rotations(
                                transformed,
                                confidences,
                                valid,
                            )
                            
                            # 4. Filter enabled trackers
                            # Use the app's _enabled_joints to filter
                            enabled_trackers = self.tracking_panel.get_enabled_trackers()
                            trackers_filtered = {k: v for k, v in trackers.items() if k in enabled_trackers}
                            
                            # 5. Send OSC
                            if self.osc_sender.send_all_trackers(trackers_filtered):
                                if time.time() - last_print > 5.0:
                                    print(f"OSC Idle: Sent T-Pose packet to {self.osc_sender.ip}:{self.osc_sender.port}")
                                    last_print = time.time()
                            
                    except Exception as e:
                        # Rate limit error printing
                        if time.time() - last_print > 5.0:
                            print(f"Idle OSC error: {e}")
                else:
                     pass
                
                time.sleep(0.033)  # ~30 FPS for smooth preview
        except Exception as e:
            print(f"CRITICAL: Idle OSC thread crashed: {e}")
    
    def run(self) -> None:
        """Run the application main loop."""
        # Ensure setup is called
        self.setup()
        
        # Auto-detect cameras on startup
        cameras = self.view.detect_cameras()
        if cameras:
            dpg.set_value("camera_count", f"{len(cameras)} camera(s) detected")
            self._update_camera_grid(cameras)
            self._start_preview(cameras)
            self.calibration_panel.set_cameras(cameras)
            self._create_pairwise_grid(cameras)
            # Start calibration capture phase (just sets step, doesn't compute yet)
            self.calibration_panel.begin_calibration()
            camera_log.info(f"Auto-detected {len(cameras)} camera(s), ready for calibration")
        
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        
        self._cleanup_recording()
        self._stop_preview()
        dpg.destroy_context()
    
    def request_stop(self) -> None:
        """Request application shutdown."""
        # Signal tracking thread to stop via shared state
        if hasattr(self, 'view') and hasattr(self.view, 'state'):
            self.view.state.tracking_mode = TrackingMode.STOPPED

        self._cleanup_recording()
        self._stop_event.set()
        
        if self.osc_sender:
            self.osc_sender.disconnect()
            
        # Give tracking thread a moment to exit loop and release resources
        import time
        time.sleep(0.5)
            
        dpg.stop_dearpygui()
