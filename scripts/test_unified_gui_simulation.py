#!/usr/bin/env python3
"""
Test Unified GUI with Dataset Frames

Simulates the unified GUI by loading frames from the dataset directory
and processing them through the GUI pipeline without requiring live cameras.
"""

import sys
import time
import threading
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from voxelvr.gui.unified_view import UnifiedView, CalibrationMode, TrackingMode
from voxelvr.gui.performance_panel import PerformanceMetrics
from voxelvr.calibration.skeleton_calibration import (
    SkeletonObservation, SimpleCameraIntrinsics, SimpleCameraExtrinsics,
    estimate_cameras_from_skeleton, triangulate_skeleton, compute_projection_matrix,
)


class DatasetSimulator:
    """Simulates camera input from dataset files."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.cameras: dict[int, list[np.ndarray]] = {}
        self.frame_index = 0
        self.fps = 30.0
        
    def load_session(self, session_name: str) -> list[int]:
        """Load a tracking session. Returns list of camera IDs."""
        session_path = self.dataset_path / "tracking" / session_name
        
        if not session_path.exists():
            print(f"Session not found: {session_path}")
            return []
        
        camera_dirs = sorted([d for d in session_path.iterdir() if d.is_dir() and d.name.startswith("cam_")])
        
        for cam_dir in camera_dirs:
            cam_id = int(cam_dir.name.split("_")[1])
            
            # Load all frames for this camera
            image_files = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
            frames = []
            
            for img_file in image_files:
                frame = cv2.imread(str(img_file))
                if frame is not None:
                    frames.append(frame)
            
            if frames:
                self.cameras[cam_id] = frames
                print(f"Loaded {len(frames)} frames for camera {cam_id}")
        
        return list(self.cameras.keys())
    
    def get_frame(self, camera_id: int) -> np.ndarray | None:
        """Get current frame for a camera."""
        if camera_id not in self.cameras:
            return None
        
        frames = self.cameras[camera_id]
        if not frames:
            return None
        
        idx = self.frame_index % len(frames)
        return frames[idx].copy()
    
    def get_all_frames(self) -> dict[int, np.ndarray]:
        """Get current frames from all cameras."""
        frames = {}
        for cam_id in self.cameras:
            frame = self.get_frame(cam_id)
            if frame is not None:
                frames[cam_id] = frame
        return frames
    
    def advance(self) -> None:
        """Advance to next frame."""
        self.frame_index += 1
    
    @property
    def total_frames(self) -> int:
        """Total number of frames in the session."""
        if not self.cameras:
            return 0
        return min(len(frames) for frames in self.cameras.values())


class UnifiedGUISimulationTest:
    """Test the unified GUI with simulated dataset input."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.simulator = DatasetSimulator(dataset_path)
        self.view = UnifiedView()
        
        # DearPyGui resources
        self._texture_registry = None
        self._camera_textures: dict[int, int] = {}
        
        # Pose detector
        self._detector = None
        
        # Test state
        self.running = True
        self.auto_advance = True
        self.frame_delay = 0.033  # ~30 FPS
        
        # Metrics
        self.frames_processed = 0
        self.poses_detected = 0
        self.start_time = 0.0
    
    def setup(self) -> None:
        """Setup DearPyGui context."""
        dpg.create_context()
        dpg.create_viewport(title="Unified GUI Test - Dataset Simulation", width=1400, height=900)
        
        self._texture_registry = dpg.add_texture_registry()
        
        self._setup_theme()
        self._create_window()
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def _setup_theme(self) -> None:
        """Setup theme."""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 28, 35))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 45, 55))
        
        dpg.bind_theme(global_theme)
    
    def _create_window(self) -> None:
        """Create the test window."""
        with dpg.window(label="Test", tag="main_window", no_title_bar=True):
            # Header
            with dpg.group(horizontal=True):
                dpg.add_text("Unified GUI Simulation Test", color=(80, 180, 255))
                dpg.add_spacer(width=30)
                dpg.add_text("Frame:", color=(150, 150, 150))
                dpg.add_text("0/0", tag="frame_counter")
                dpg.add_spacer(width=20)
                dpg.add_text("FPS:", color=(150, 150, 150))
                dpg.add_text("--", tag="fps_display", color=(100, 200, 100))
                dpg.add_spacer(width=20)
                dpg.add_text("Poses:", color=(150, 150, 150))
                dpg.add_text("0", tag="poses_display", color=(100, 200, 100))
            
            dpg.add_separator()
            
            # Controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Session", callback=self._on_load_session)
                dpg.add_button(label="▶ Play", tag="play_btn", callback=self._on_play_toggle)
                dpg.add_button(label="⏭ Step", callback=self._on_step)
                dpg.add_button(label="⏮ Reset", callback=self._on_reset)
                dpg.add_spacer(width=20)
                dpg.add_slider_float(
                    label="Speed",
                    tag="speed_slider",
                    default_value=1.0,
                    min_value=0.1,
                    max_value=5.0,
                    width=150,
                    callback=self._on_speed_change,
                )
            
            dpg.add_separator()
            
            # Main content
            with dpg.group(horizontal=True):
                # Camera grid
                with dpg.child_window(tag="camera_panel", width=950, height=-100):
                    dpg.add_text("Camera Feeds (from dataset)", color=(150, 150, 150))
                    with dpg.child_window(tag="camera_grid", autosize_x=True, height=-1):
                        dpg.add_text("Click 'Load Session' to begin.", tag="no_data_text")
                
                # Status panel
                with dpg.child_window(tag="status_panel", autosize_x=True, height=-100):
                    dpg.add_text("Test Status", color=(150, 200, 255))
                    dpg.add_separator()
                    
                    dpg.add_text("View State:", color=(150, 150, 150))
                    dpg.add_text("Tracking: Stopped", tag="tracking_status")
                    dpg.add_text("Calibration: ChArUco", tag="calib_status")
                    dpg.add_text("AprilTags: Disabled", tag="apriltag_status")
                    
                    dpg.add_separator()
                    
                    dpg.add_text("Pose Detection:", color=(150, 150, 150))
                    dpg.add_text("Waiting...", tag="pose_status")
                    
                    dpg.add_separator()
                    
                    dpg.add_text("Sessions:", color=(150, 150, 150))
                    sessions = self._find_sessions()
                    dpg.add_combo(
                        items=sessions,
                        tag="session_combo",
                        default_value=sessions[0] if sessions else "",
                        width=-1,
                    )
            
            # Log area
            dpg.add_separator()
            dpg.add_text("Log:", color=(150, 150, 150))
            with dpg.child_window(tag="log_window", autosize_x=True, height=80):
                dpg.add_text("Ready.", tag="log_text", wrap=0)
        
        dpg.set_primary_window("main_window", True)
    
    def _find_sessions(self) -> list[str]:
        """Find available tracking sessions."""
        tracking_path = self.dataset_path / "tracking"
        if not tracking_path.exists():
            return []
        
        sessions = [d.name for d in tracking_path.iterdir() if d.is_dir()]
        return sorted(sessions)
    
    def _log(self, message: str) -> None:
        """Add log message."""
        if dpg.does_item_exist("log_text"):
            current = dpg.get_value("log_text")
            lines = current.split("\n")[-5:]  # Keep last 5 lines
            lines.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            dpg.set_value("log_text", "\n".join(lines))
    
    def _on_load_session(self, sender=None, app_data=None) -> None:
        """Load selected session."""
        session = dpg.get_value("session_combo")
        if not session:
            self._log("No session selected")
            return
        
        self._log(f"Loading session: {session}")
        camera_ids = self.simulator.load_session(session)
        
        if not camera_ids:
            self._log("Failed to load session")
            return
        
        self._log(f"Loaded {len(camera_ids)} cameras, {self.simulator.total_frames} frames")
        
        # Setup camera textures
        self._setup_camera_grid(camera_ids)
        
        # Update view
        self.view.set_cameras(camera_ids)
        
        # Initialize pose detector
        try:
            from voxelvr.pose.detector_2d import PoseDetector2D
            self._detector = PoseDetector2D()
            self._log("Pose detector initialized")
        except Exception as e:
            self._log(f"Pose detector failed: {e}")
            self._detector = None
        
        # Reset counters
        self.simulator.frame_index = 0
        self.frames_processed = 0
        self.poses_detected = 0
        self.start_time = time.time()
        
        self._update_display()
    
    def _setup_camera_grid(self, camera_ids: list[int]) -> None:
        """Setup camera texture grid."""
        dpg.delete_item("camera_grid", children_only=True)
        
        # Get sample frame dimensions
        sample = self.simulator.get_frame(camera_ids[0])
        if sample is None:
            return
        
        h, w = sample.shape[:2]
        # Scale down for display
        display_w, display_h = 420, int(420 * h / w)
        
        for cam_id in camera_ids:
            texture_tag = f"cam_texture_{cam_id}"
            
            # Create placeholder
            placeholder = np.zeros((display_h, display_w, 4), dtype=np.float32)
            placeholder[:, :, 3] = 1.0
            
            self._camera_textures[cam_id] = dpg.add_dynamic_texture(
                width=display_w,
                height=display_h,
                default_value=placeholder.flatten().tolist(),
                parent=self._texture_registry,
                tag=texture_tag,
            )
            
            with dpg.group(parent="camera_grid", horizontal=False):
                dpg.add_text(f"Camera {cam_id}")
                dpg.add_image(texture_tag)
    
    def _update_camera_frame(self, camera_id: int, frame: np.ndarray) -> None:
        """Update camera texture."""
        texture_tag = f"cam_texture_{camera_id}"
        if not dpg.does_item_exist(texture_tag):
            return
        
        # Get display size from texture
        w, h = dpg.get_item_width(texture_tag), dpg.get_item_height(texture_tag)
        if w == 0 or h == 0:
            w, h = 420, 315
        
        frame_resized = cv2.resize(frame, (w, h))
        frame_rgba = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)
        frame_float = frame_rgba.astype(np.float32) / 255.0
        
        dpg.set_value(texture_tag, frame_float.flatten().tolist())
    
    def _on_play_toggle(self, sender=None, app_data=None) -> None:
        """Toggle auto-advance."""
        self.auto_advance = not self.auto_advance
        dpg.set_item_label("play_btn", "⏸ Pause" if self.auto_advance else "▶ Play")
        self._log("Playing" if self.auto_advance else "Paused")
    
    def _on_step(self, sender=None, app_data=None) -> None:
        """Advance one frame."""
        self._process_frame()
        self.simulator.advance()
        self._update_display()
    
    def _on_reset(self, sender=None, app_data=None) -> None:
        """Reset to first frame."""
        self.simulator.frame_index = 0
        self.frames_processed = 0
        self.poses_detected = 0
        self.start_time = time.time()
        self._update_display()
        self._log("Reset to frame 0")
    
    def _on_speed_change(self, sender, app_data) -> None:
        """Update playback speed."""
        self.frame_delay = 0.033 / app_data
    
    def _process_frame(self) -> None:
        """Process current frame through the pipeline."""
        frames = self.simulator.get_all_frames()
        if not frames:
            return
        
        poses_this_frame = 0
        
        for cam_id, frame in frames.items():
            display_frame = frame.copy()
            
            # Run pose detection if available
            if self._detector is not None:
                try:
                    result = self._detector.detect(frame)
                    if result is not None:
                        display_frame = self._detector.draw_keypoints(display_frame, result)
                        poses_this_frame += 1
                except Exception:
                    pass
            
            self._update_camera_frame(cam_id, display_frame)
        
        self.frames_processed += 1
        if poses_this_frame > 0:
            self.poses_detected += 1
    
    def _update_display(self) -> None:
        """Update display elements."""
        # Frame counter
        current = self.simulator.frame_index
        total = self.simulator.total_frames
        dpg.set_value("frame_counter", f"{current}/{total}")
        
        # FPS
        if self.frames_processed > 0 and self.start_time > 0:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                fps = self.frames_processed / elapsed
                dpg.set_value("fps_display", f"{fps:.1f}")
        
        # Poses
        dpg.set_value("poses_display", str(self.poses_detected))
        
        # View state
        dpg.set_value("tracking_status", f"Tracking: {self.view.state.tracking_mode.name}")
        dpg.set_value("calib_status", f"Calibration: {self.view.state.calibration.mode.name}")
        dpg.set_value("apriltag_status", f"AprilTags: {'Enabled' if self.view.state.apriltags_enabled else 'Disabled'}")
        
        # Pose status
        if self._detector is not None:
            dpg.set_value("pose_status", f"Detected in {self.poses_detected}/{self.frames_processed} frames")
        else:
            dpg.set_value("pose_status", "Detector not loaded")
    
    def run(self) -> None:
        """Run the test application."""
        last_frame_time = time.time()
        
        while dpg.is_dearpygui_running() and self.running:
            current_time = time.time()
            
            # Auto-advance frames
            if self.auto_advance and self.simulator.cameras:
                if current_time - last_frame_time >= self.frame_delay:
                    self._process_frame()
                    self.simulator.advance()
                    self._update_display()
                    last_frame_time = current_time
            
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()


def main():
    dataset_path = Path("/home/lee/voxelvr/dataset")
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1
    
    print("Starting Unified GUI Simulation Test...")
    print(f"Dataset: {dataset_path}")
    
    test = UnifiedGUISimulationTest(dataset_path)
    test.setup()
    test.run()
    
    print(f"\nTest complete:")
    print(f"  Frames processed: {test.frames_processed}")
    print(f"  Poses detected: {test.poses_detected}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
