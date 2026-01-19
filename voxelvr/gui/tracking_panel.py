"""
Tracking Panel

Controls for starting/stopping tracking and configuring OSC output.
"""

import numpy as np
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum


class TrackingState(Enum):
    """Tracking pipeline states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TrackerConfig:
    """Configuration for individual trackers."""
    name: str
    enabled: bool = True
    display_name: str = ""
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace('_', ' ').title()


@dataclass
class TrackingStatus:
    """Current tracking status."""
    state: TrackingState = TrackingState.STOPPED
    fps: float = 0.0
    valid_joints: int = 0
    total_joints: int = 17
    trackers_sending: int = 0
    error_message: str = ""
    
    # Per-tracker status
    tracker_validity: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.tracker_validity is None:
            self.tracker_validity = {}


class TrackingPanel:
    """
    Tracking controls and status display.
    
    Features:
    - Start/Stop tracking controls
    - OSC configuration (IP, port)
    - Tracker enable/disable toggles
    - Real-time joint validity display
    - Skeleton preview
    """
    
    # Default tracker definitions
    DEFAULT_TRACKERS = [
        TrackerConfig("hip", True, "Hip"),
        TrackerConfig("chest", True, "Chest"),
        TrackerConfig("left_foot", True, "Left Foot"),
        TrackerConfig("right_foot", True, "Right Foot"),
        TrackerConfig("left_knee", True, "Left Knee"),
        TrackerConfig("right_knee", True, "Right Knee"),
        TrackerConfig("left_elbow", True, "Left Elbow"),
        TrackerConfig("right_elbow", True, "Right Elbow"),
    ]
    
    def __init__(
        self,
        osc_ip: str = "127.0.0.1",
        osc_port: int = 9000,
    ):
        """
        Initialize tracking panel.
        
        Args:
            osc_ip: Default OSC target IP
            osc_port: Default OSC target port
        """
        self.osc_ip = osc_ip
        self.osc_port = osc_port
        
        # Tracker configuration
        self._trackers = {t.name: t for t in self.DEFAULT_TRACKERS}
        
        # Status
        self._status = TrackingStatus()
        
        # Callbacks
        self._state_callbacks: List[Callable[[TrackingState], None]] = []
        self._tracker_callbacks: List[Callable[[str, bool], None]] = []
        
        # Current pose data (for display)
        # Initialize with T-Pose
        from .skeleton_viewer import get_tpose
        tpose = get_tpose()
        self._current_positions = tpose['positions']
        self._current_valid_mask = tpose['valid']
        self._current_confidences = tpose['confidences']
    
    @property
    def status(self) -> TrackingStatus:
        """Get current tracking status."""
        return self._status
    
    @property
    def is_running(self) -> bool:
        """Check if tracking is running."""
        return self._status.state == TrackingState.RUNNING
    
    def add_state_callback(self, callback: Callable[[TrackingState], None]) -> None:
        """Add callback for state changes."""
        self._state_callbacks.append(callback)
    
    def add_tracker_callback(self, callback: Callable[[str, bool], None]) -> None:
        """Add callback for tracker enable/disable changes."""
        self._tracker_callbacks.append(callback)
    
    def _notify_state_change(self) -> None:
        """Notify callbacks of state change."""
        for callback in self._state_callbacks:
            try:
                callback(self._status.state)
            except Exception as e:
                print(f"State callback error: {e}")
    
    def _notify_tracker_change(self, tracker_name: str, enabled: bool) -> None:
        """Notify callbacks of tracker change."""
        for callback in self._tracker_callbacks:
            try:
                callback(tracker_name, enabled)
            except Exception as e:
                print(f"Tracker callback error: {e}")
    
    def set_osc_config(self, ip: str, port: int) -> None:
        """Update OSC configuration."""
        self.osc_ip = ip
        self.osc_port = port
    
    def get_osc_config(self) -> Tuple[str, int]:
        """Get current OSC configuration."""
        return (self.osc_ip, self.osc_port)
    
    def set_tracker_enabled(self, tracker_name: str, enabled: bool) -> None:
        """Enable or disable a tracker."""
        if tracker_name in self._trackers:
            self._trackers[tracker_name].enabled = enabled
            self._notify_tracker_change(tracker_name, enabled)
    
    def get_tracker_enabled(self, tracker_name: str) -> bool:
        """Check if a tracker is enabled."""
        if tracker_name in self._trackers:
            return self._trackers[tracker_name].enabled
        return False
    
    def get_enabled_trackers(self) -> List[str]:
        """Get list of enabled tracker names."""
        return [name for name, t in self._trackers.items() if t.enabled]
    
    def get_all_trackers(self) -> Dict[str, TrackerConfig]:
        """Get all tracker configurations."""
        return self._trackers.copy()
    
    def request_start(self) -> None:
        """Request to start tracking."""
        if self._status.state == TrackingState.STOPPED:
            self._status.state = TrackingState.STARTING
            self._status.error_message = ""
            self._notify_state_change()
    
    def request_stop(self) -> None:
        """Request to stop tracking."""
        if self._status.state == TrackingState.RUNNING:
            self._status.state = TrackingState.STOPPING
            self._notify_state_change()
    
    def on_tracking_started(self) -> None:
        """Call when tracking has started successfully."""
        self._status.state = TrackingState.RUNNING
        self._notify_state_change()
    
    def on_tracking_stopped(self) -> None:
        """Call when tracking has stopped."""
        self._status.state = TrackingState.STOPPED
        self._status.fps = 0.0
        self._status.valid_joints = 0
        self._status.trackers_sending = 0
        self._notify_state_change()
    
    def on_tracking_error(self, error: str) -> None:
        """Call when a tracking error occurs."""
        self._status.state = TrackingState.ERROR
        self._status.error_message = error
        self._notify_state_change()
    
    def update_status(
        self,
        fps: Optional[float] = None,
        valid_joints: Optional[int] = None,
        trackers_sending: Optional[int] = None,
        tracker_validity: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Update tracking status metrics."""
        if fps is not None:
            self._status.fps = fps
        if valid_joints is not None:
            self._status.valid_joints = valid_joints
        if trackers_sending is not None:
            self._status.trackers_sending = trackers_sending
        if tracker_validity is not None:
            self._status.tracker_validity = tracker_validity
    
    def update_pose(
        self,
        positions: np.ndarray,
        valid_mask: np.ndarray,
        confidences: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update current pose data for display.
        
        Args:
            positions: (17, 3) joint positions
            valid_mask: (17,) boolean mask
            confidences: (17,) confidence scores
        """
        self._current_positions = positions.copy()
        self._current_valid_mask = valid_mask.copy()
        if confidences is not None:
            self._current_confidences = confidences.copy()
        else:
            self._current_confidences = np.ones(17)
    
    def get_current_pose(self) -> Optional[Dict]:
        """Get current pose data for visualization."""
        if self._current_positions is None:
            return None
        
        return {
            'positions': self._current_positions,
            'valid': self._current_valid_mask,
            'confidences': self._current_confidences,
        }
    
    def get_status_text(self) -> str:
        """Get human-readable status text."""
        state = self._status.state
        
        if state == TrackingState.STOPPED:
            return "Tracking stopped"
        elif state == TrackingState.STARTING:
            return "Starting tracking..."
        elif state == TrackingState.RUNNING:
            return (
                f"Tracking: {self._status.fps:.1f} FPS | "
                f"Joints: {self._status.valid_joints}/17 | "
                f"Trackers: {self._status.trackers_sending}"
            )
        elif state == TrackingState.STOPPING:
            return "Stopping tracking..."
        elif state == TrackingState.ERROR:
            return f"Error: {self._status.error_message}"
        
        return ""
    
    def get_joint_info(self) -> List[Dict]:
        """
        Get information about each joint for display.
        
        Returns:
            List of dicts with joint name, valid status, and confidence
        """
        # COCO keypoint names
        joint_names = [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]
        
        result = []
        for i, name in enumerate(joint_names):
            valid = False
            confidence = 0.0
            
            if self._current_valid_mask is not None and i < len(self._current_valid_mask):
                valid = self._current_valid_mask[i]
            if self._current_confidences is not None and i < len(self._current_confidences):
                confidence = self._current_confidences[i]
            
            result.append({
                'index': i,
                'name': name,
                'valid': valid,
                'confidence': confidence,
            })
        
        return result
