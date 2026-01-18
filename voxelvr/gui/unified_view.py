"""
Unified View Module

Single unified interface for VoxelVR, combining tracking controls,
calibration status, and camera preview in one view.
"""

from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np


class CalibrationMode(Enum):
    """Active calibration mode."""
    CHARUCO = "charuco"
    SKELETON = "skeleton"  # Hidden in UI for now, kept for future use


class TrackingMode(Enum):
    """Tracking state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


@dataclass
class CameraFeedInfo:
    """Information about a camera feed."""
    camera_id: int
    is_active: bool = True
    last_frame: Optional[np.ndarray] = None
    fps: float = 0.0
    board_visible: bool = False
    apriltags_detected: int = 0


@dataclass
class CalibrationStatus:
    """Current calibration status."""
    mode: CalibrationMode = CalibrationMode.CHARUCO
    charuco_frames: int = 0
    charuco_required: int = 12
    # Skeleton mode - hidden in UI for now but kept for future improvements
    skeleton_poses: int = 0
    skeleton_required: int = 3
    is_skeleton_only: bool = False
    is_calibrated: bool = False
    reprojection_error: float = 0.0
    # Pairwise progress
    per_camera_progress: Dict[int, int] = field(default_factory=dict)  # cam_id -> intrinsic frames
    pairwise_progress: Dict[tuple, int] = field(default_factory=dict)  # (cam_a, cam_b) -> frames
    is_connected: bool = False  # True when camera chain is complete


@dataclass 
class UnifiedViewState:
    """Complete state of the unified view."""
    # Camera state
    cameras: Dict[int, CameraFeedInfo] = field(default_factory=dict)
    
    # Calibration
    calibration: CalibrationStatus = field(default_factory=CalibrationStatus)
    is_charuco_capture_active: bool = False
    
    # Tracking
    tracking_mode: TrackingMode = TrackingMode.STOPPED
    tracking_fps: float = 0.0
    active_joints: int = 0
    
    # Features
    apriltags_enabled: bool = False
    
    # OSC
    osc_ip: str = "127.0.0.1"
    osc_port: int = 9000
    osc_connected: bool = False
    
    # Floating windows
    debug_window_open: bool = False
    performance_window_open: bool = False


class UnifiedView:
    """
    Single unified interface for VoxelVR.
    
    Features:
    - Camera grid with auto-detection
    - ChArUco board calibration with pairwise progress
    - Tracking controls with live status
    - AprilTag precision toggle
    - Export buttons (ChArUco PDF, AprilTag PDF)
    - OSC configuration
    - Floating debug/performance windows
    """
    
    def __init__(
        self,
        osc_ip: str = "127.0.0.1",
        osc_port: int = 9000,
        charuco_frames_required: int = 12,
        skeleton_poses_required: int = 3,
    ):
        """
        Initialize unified view.
        
        Args:
            osc_ip: Default OSC target IP
            osc_port: Default OSC target port
            charuco_frames_required: ChArUco frames needed for calibration
            skeleton_poses_required: Skeleton poses needed for fallback calibration
        """
        self._state = UnifiedViewState(
            osc_ip=osc_ip,
            osc_port=osc_port,
        )
        self._state.calibration.charuco_required = charuco_frames_required
        self._state.calibration.skeleton_required = skeleton_poses_required
        
        # Callbacks
        self._on_detect_cameras: Optional[Callable[[], List[int]]] = None
        self._on_start_tracking: Optional[Callable[[], None]] = None
        self._on_stop_tracking: Optional[Callable[[], None]] = None
        self._on_start_charuco_capture: Optional[Callable[[], None]] = None
        self._on_stop_charuco_capture: Optional[Callable[[], None]] = None
        self._on_export_charuco: Optional[Callable[[Path], bool]] = None
        self._on_export_apriltag: Optional[Callable[[Path], bool]] = None
        self._on_osc_config_change: Optional[Callable[[str, int], None]] = None
        self._on_apriltag_toggle: Optional[Callable[[bool], None]] = None
        
        # State change callbacks
        self._state_callbacks: List[Callable[[UnifiedViewState], None]] = []
    
    @property
    def state(self) -> UnifiedViewState:
        """Get current view state."""
        return self._state
    
    def add_state_callback(self, callback: Callable[[UnifiedViewState], None]) -> None:
        """Add callback for state changes."""
        self._state_callbacks.append(callback)
    
    def _notify_state_change(self) -> None:
        """Notify callbacks of state change."""
        for callback in self._state_callbacks:
            try:
                callback(self._state)
            except Exception as e:
                print(f"State callback error: {e}")
    
    # === Callback Setters ===
    
    def set_detect_cameras_callback(self, callback: Callable[[], List[int]]) -> None:
        """Set callback for camera detection."""
        self._on_detect_cameras = callback
    
    def set_tracking_callbacks(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        """Set callbacks for tracking start/stop."""
        self._on_start_tracking = on_start
        self._on_stop_tracking = on_stop
    
    def set_charuco_callbacks(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        """Set callbacks for ChArUco capture start/stop."""
        self._on_start_charuco_capture = on_start
        self._on_stop_charuco_capture = on_stop
    
    def set_export_callbacks(
        self,
        on_export_charuco: Callable[[Path], bool],
        on_export_apriltag: Callable[[Path], bool],
    ) -> None:
        """Set callbacks for PDF export."""
        self._on_export_charuco = on_export_charuco
        self._on_export_apriltag = on_export_apriltag
    
    def set_osc_callback(self, callback: Callable[[str, int], None]) -> None:
        """Set callback for OSC config changes."""
        self._on_osc_config_change = callback
    
    def set_apriltag_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback for AprilTag toggle."""
        self._on_apriltag_toggle = callback
    
    # === Camera Methods ===
    
    def detect_cameras(self) -> List[int]:
        """Trigger camera detection."""
        if self._on_detect_cameras:
            camera_ids = self._on_detect_cameras()
            for cam_id in camera_ids:
                if cam_id not in self._state.cameras:
                    self._state.cameras[cam_id] = CameraFeedInfo(camera_id=cam_id)
            self._notify_state_change()
            return camera_ids
        return []
    
    def set_cameras(self, camera_ids: List[int]) -> None:
        """Set available camera IDs."""
        self._state.cameras = {
            cid: CameraFeedInfo(camera_id=cid) 
            for cid in camera_ids
        }
        self._notify_state_change()
    
    def update_camera_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        fps: float = 0.0,
        board_visible: bool = False,
        apriltags_detected: int = 0,
    ) -> None:
        """Update camera feed info."""
        if camera_id in self._state.cameras:
            info = self._state.cameras[camera_id]
            info.last_frame = frame
            info.fps = fps
            info.board_visible = board_visible
            info.apriltags_detected = apriltags_detected
    
    def get_camera_grid_layout(self) -> Tuple[int, int]:
        """Get grid layout (rows, cols) for camera display."""
        n = len(self._state.cameras)
        if n == 0:
            return (0, 0)
        elif n == 1:
            return (1, 1)
        elif n == 2:
            return (1, 2)
        elif n <= 4:
            return (2, 2)
        elif n <= 6:
            return (2, 3)
        elif n <= 9:
            return (3, 3)
        else:
            return (3, 4)
    
    # === Calibration Methods ===
    
    def set_calibration_mode(self, mode: CalibrationMode) -> None:
        """Set active calibration mode."""
        self._state.calibration.mode = mode
        self._notify_state_change()
    
    def start_charuco_capture(self) -> None:
        """Start ChArUco capture mode."""
        self._state.is_charuco_capture_active = True
        if self._on_start_charuco_capture:
            self._on_start_charuco_capture()
        self._notify_state_change()
    
    def stop_charuco_capture(self) -> None:
        """Stop ChArUco capture mode."""
        self._state.is_charuco_capture_active = False
        if self._on_stop_charuco_capture:
            self._on_stop_charuco_capture()
        self._notify_state_change()
    
    def update_calibration_progress(
        self,
        charuco_frames: Optional[int] = None,
        is_calibrated: Optional[bool] = None,
        reprojection_error: Optional[float] = None,
        per_camera_progress: Optional[Dict[int, int]] = None,
        pairwise_progress: Optional[Dict[tuple, int]] = None,
        is_connected: Optional[bool] = None,
    ) -> None:
        """Update calibration progress."""
        cal = self._state.calibration
        if charuco_frames is not None:
            cal.charuco_frames = charuco_frames
        if is_calibrated is not None:
            cal.is_calibrated = is_calibrated
        if reprojection_error is not None:
            cal.reprojection_error = reprojection_error
        if per_camera_progress is not None:
            cal.per_camera_progress = per_camera_progress
        if pairwise_progress is not None:
            cal.pairwise_progress = pairwise_progress
        if is_connected is not None:
            cal.is_connected = is_connected
        self._notify_state_change()
    
    def get_calibration_status_text(self) -> str:
        """Get human-readable calibration status."""
        cal = self._state.calibration
        
        if cal.is_calibrated:
            return f"Calibrated (error: {cal.reprojection_error:.3f}px)"
        elif cal.is_connected:
            return "Computing calibration..."
        else:
            n_cams = len(cal.per_camera_progress)
            return f"ChArUco: {n_cams} cameras"
    
    def get_connectivity_status(self) -> str:
        """Get connectivity status text."""
        cal = self._state.calibration
        if cal.is_connected:
            return "● Connected"
        else:
            return "○ Not Connected"
    
    # === Tracking Methods ===
    
    def start_tracking(self) -> None:
        """Start tracking."""
        if self._state.tracking_mode == TrackingMode.STOPPED:
            self._state.tracking_mode = TrackingMode.STARTING
            if self._on_start_tracking:
                self._on_start_tracking()
            self._notify_state_change()
    
    def stop_tracking(self) -> None:
        """Stop tracking."""
        if self._state.tracking_mode == TrackingMode.RUNNING:
            self._state.tracking_mode = TrackingMode.STOPPING
            if self._on_stop_tracking:
                self._on_stop_tracking()
            self._notify_state_change()
    
    def on_tracking_started(self) -> None:
        """Call when tracking has started."""
        self._state.tracking_mode = TrackingMode.RUNNING
        self._notify_state_change()
    
    def on_tracking_stopped(self) -> None:
        """Call when tracking has stopped."""
        self._state.tracking_mode = TrackingMode.STOPPED
        self._state.tracking_fps = 0.0
        self._state.active_joints = 0
        self._notify_state_change()
    
    def update_tracking_status(
        self,
        fps: Optional[float] = None,
        active_joints: Optional[int] = None,
    ) -> None:
        """Update tracking status."""
        if fps is not None:
            self._state.tracking_fps = fps
        if active_joints is not None:
            self._state.active_joints = active_joints
    
    def get_tracking_button_label(self) -> str:
        """Get label for tracking button."""
        mode = self._state.tracking_mode
        if mode == TrackingMode.STOPPED:
            return "▶ Start Tracking"
        elif mode == TrackingMode.STARTING:
            return "Starting..."
        elif mode == TrackingMode.RUNNING:
            return "◼ Stop Tracking"
        elif mode == TrackingMode.STOPPING:
            return "Stopping..."
        return ""
    
    def is_tracking_button_enabled(self) -> bool:
        """Check if tracking button should be enabled."""
        mode = self._state.tracking_mode
        return mode in (TrackingMode.STOPPED, TrackingMode.RUNNING)
    
    # === AprilTag Methods ===
    
    def toggle_apriltags(self) -> None:
        """Toggle AprilTag precision mode."""
        self._state.apriltags_enabled = not self._state.apriltags_enabled
        if self._on_apriltag_toggle:
            self._on_apriltag_toggle(self._state.apriltags_enabled)
        self._notify_state_change()
    
    def set_apriltags_enabled(self, enabled: bool) -> None:
        """Set AprilTag mode."""
        self._state.apriltags_enabled = enabled
        if self._on_apriltag_toggle:
            self._on_apriltag_toggle(enabled)
        self._notify_state_change()
    
    # === Export Methods ===
    
    def export_charuco_pdf(self, output_path: Path) -> bool:
        """Export ChArUco board PDF."""
        if self._on_export_charuco:
            return self._on_export_charuco(output_path)
        return False
    
    def export_apriltag_pdf(self, output_path: Path) -> bool:
        """Export AprilTag sheet PDF."""
        if self._on_export_apriltag:
            return self._on_export_apriltag(output_path)
        return False
    
    # === OSC Methods ===
    
    def set_osc_config(self, ip: str, port: int) -> None:
        """Update OSC configuration."""
        self._state.osc_ip = ip
        self._state.osc_port = port
        if self._on_osc_config_change:
            self._on_osc_config_change(ip, port)
        self._notify_state_change()
    
    def set_osc_connected(self, connected: bool) -> None:
        """Update OSC connection status."""
        self._state.osc_connected = connected
        self._notify_state_change()
    
    def get_osc_status_text(self) -> str:
        """Get OSC status indicator text."""
        if self._state.osc_connected:
            return "⬤ OSC Connected"
        else:
            return "○ OSC Disconnected"
    
    # === Floating Window Methods ===
    
    def toggle_debug_window(self) -> None:
        """Toggle debug window visibility."""
        self._state.debug_window_open = not self._state.debug_window_open
        self._notify_state_change()
    
    def toggle_performance_window(self) -> None:
        """Toggle performance window visibility."""
        self._state.performance_window_open = not self._state.performance_window_open
        self._notify_state_change()
    
    def set_debug_window_open(self, open: bool) -> None:
        """Set debug window visibility."""
        self._state.debug_window_open = open
        self._notify_state_change()
    
    def set_performance_window_open(self, open: bool) -> None:
        """Set performance window visibility."""
        self._state.performance_window_open = open
        self._notify_state_change()
