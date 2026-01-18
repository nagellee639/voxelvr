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
from .skeleton_viewer import SkeletonViewer


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
    ):
        self.title = title
        self.width = width
        self.height = height
        
        # Core view state
        self.view = UnifiedView()
        
        # Panels for floating windows
        self.performance_panel = PerformancePanel()
        self.debug_panel = DebugPanel()
        self.skeleton_viewer = SkeletonViewer(size=(400, 400))
        
        # Calibration panel for pairwise capture logic
        from .calibration_panel import CalibrationPanel
        self.calibration_panel = CalibrationPanel()
        self.calibration_panel.add_progress_callback(self._on_calibration_progress)
        
        # ChArUco board for detection
        from ..calibration.charuco import create_charuco_board
        self._charuco_board, self._aruco_dict = create_charuco_board()
        
        # DearPyGui resources
        self._texture_registry: Optional[int] = None
        self._camera_textures: Dict[int, int] = {}
        self._skeleton_texture: Optional[int] = None
        
        # Threading
        self._stop_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None
        
        # Camera preview
        from ..capture.manager import CameraManager
        self.camera_manager: Optional[CameraManager] = None
        self.preview_thread: Optional[threading.Thread] = None
        self.preview_active = False
        
        # Recording support
        self._recording_enabled = False
        self._recording_dir: Optional[Path] = None
        self._video_writers: Dict[int, cv2.VideoWriter] = {}
        
        # External callbacks
        self._on_start_tracking: Optional[Callable] = None
        self._on_stop_tracking: Optional[Callable] = None
        
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
    
    def setup(self) -> None:
        """Initialize DearPyGui context and window."""
        dpg.create_context()
        dpg.create_viewport(title=self.title, width=self.width, height=self.height)
        
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
        """Update camera grid display."""
        dpg.delete_item("camera_grid", children_only=True)
        
        if not camera_ids:
            dpg.add_text("No cameras detected.", parent="camera_grid")
            return
        
        rows, cols = self.view.get_camera_grid_layout()
        
        for i, cam_id in enumerate(camera_ids):
            # Create texture for this camera
            texture_tag = f"cam_texture_{cam_id}"
            
            if texture_tag not in self._camera_textures:
                # Create placeholder texture
                placeholder = np.zeros((360, 480, 4), dtype=np.float32)
                placeholder[:, :, 3] = 1.0
                
                self._camera_textures[cam_id] = dpg.add_dynamic_texture(
                    width=480,
                    height=360,
                    default_value=placeholder.flatten().tolist(),
                    parent=self._texture_registry,
                    tag=texture_tag,
                )
            
            # Add image to grid
            with dpg.group(parent="camera_grid"):
                dpg.add_text(f"Camera {cam_id}")
                dpg.add_image(texture_tag)
    
    def update_camera_frame(self, camera_id: int, frame: np.ndarray) -> None:
        """Update a camera's texture with new frame."""
        texture_tag = f"cam_texture_{camera_id}"
        
        if not dpg.does_item_exist(texture_tag):
            return
        
        # Resize and convert
        frame_resized = cv2.resize(frame, (480, 360))
        frame_rgba = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)
        frame_float = frame_rgba.astype(np.float32) / 255.0
        
        dpg.set_value(texture_tag, frame_float.flatten().tolist())
    
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
    
    def _preview_loop(self) -> None:
        """Camera preview update loop with ChArUco detection."""
        from ..calibration.charuco import detect_charuco
        
        while self.preview_active and dpg.is_dearpygui_running():
            if not self.camera_manager:
                break
            
            frames = self.camera_manager.get_all_latest_frames()
            
            if frames:
                detections = {}
                
                for cam_id, frame in frames.items():
                    # Detect ChArUco board
                    result = detect_charuco(frame.image, self._charuco_board, self._aruco_dict)
                    detections[cam_id] = {
                        'success': result['success'],
                        'corners': result['corners'],
                        'ids': result['ids'],
                        'frame': frame.image
                    }
                    
                    # Use annotated frame for display
                    display_frame = result['image_with_markers']
                    self.update_camera_frame(cam_id, display_frame)
                    
                    # Update camera visibility in view state
                    self.view.update_camera_frame(
                        cam_id, frame.image,
                        board_visible=result['success']
                    )
                
                # Process detections for calibration (always active in unified mode)
                capture_result = self.calibration_panel.process_frame_detections(detections)
                
                # Handle recording if enabled
                if self._recording_enabled:
                    self._record_frames(frames)
            
            time.sleep(0.033)  # ~30 FPS
    
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
            print("Calibration complete! Ready for tracking.")
            self._on_calibration_complete()
    
    def _update_pairwise_grid_ui(self) -> None:
        """Update the pairwise progress grid display."""
        progress = self.calibration_panel.get_progress_summary()
        
        # Update per-camera intrinsic bars
        for cam_id, data in progress['cameras'].items():
            tag = f"intrinsic_{cam_id}"
            if dpg.does_item_exist(tag):
                percent = data['intrinsic_percent'] / 100.0
                dpg.set_value(tag, percent)
        
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
        
        # Update status text
        if dpg.does_item_exist("calib_status"):
            dpg.set_value("calib_status", self.view.get_calibration_status_text())
    
    def _on_calibration_complete(self) -> None:
        """Handle calibration completion - transition to tracking."""
        result = self.calibration_panel.state.extrinsics.result
        if result:
            # Save calibration
            from ..config import CONFIG_DIR
            output_path = CONFIG_DIR / "calibration" / "calibration.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
            print(f"Saved calibration to {output_path}")
    
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
        if self.view.state.tracking_mode == TrackingMode.RUNNING:
            self.view.stop_tracking()
            dpg.set_item_label("tracking_btn", "▶ Start Tracking")
        else:
            self._stop_preview()
            self.view.start_tracking()
            dpg.set_item_label("tracking_btn", "⏹ Stop Tracking")
    
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
        
        frame = self.skeleton_viewer.render(positions, valid)
        frame_float = frame.astype(np.float32) / 255.0
        dpg.set_value("skeleton_texture", frame_float.flatten().tolist())
    
    # =========================================================================
    # Compatibility shims for run_gui.py
    # =========================================================================
    
    @property
    def tracking_panel(self):
        """Compatibility shim for accessing tracking-related methods."""
        return self._TrackingPanelShim(self)
    
    class _TrackingPanelShim:
        """Minimal shim to provide tracking_panel interface."""
        def __init__(self, app: 'UnifiedVoxelVRApp'):
            self.app = app
        
        def on_tracking_error(self, message: str) -> None:
            """Handle tracking error."""
            print(f"Tracking error: {message}")
        
        @property
        def is_running(self) -> bool:
            return self.app.view.state.tracking_mode == TrackingMode.RUNNING
    
    @property 
    def state(self):
        """Compatibility shim for state access."""
        return self._StateShim(self)
    
    class _StateShim:
        """Minimal shim for state."""
        def __init__(self, app: 'UnifiedVoxelVRApp'):
            self.app = app
            self.is_running = True
    
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
            self.calibration_panel.begin_calibration()
            print(f"Auto-detected {len(cameras)} camera(s), starting calibration")
        
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        
        self._cleanup_recording()
        self._stop_preview()
        dpg.destroy_context()
    
    def request_stop(self) -> None:
        """Request application shutdown."""
        self._cleanup_recording()
        self._stop_event.set()
        dpg.stop_dearpygui()
