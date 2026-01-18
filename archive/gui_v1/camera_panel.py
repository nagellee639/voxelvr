"""
Camera Preview Panel

Displays real-time camera feeds in a flexible grid layout.
Supports any number of cameras with automatic layout adjustment.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class CameraFeedInfo:
    """Information about a camera feed."""
    camera_id: int
    name: str
    resolution: Tuple[int, int]
    fps: float
    is_connected: bool
    last_frame_time: float
    frame_count: int


class CameraPanel:
    """
    Manages camera preview display with flexible grid layout.
    
    Features:
    - Dynamic grid layout based on camera count
    - Real-time frame updates via textures
    - Camera status indicators
    - Click-to-expand functionality
    - Frame rate display per camera
    """
    
    def __init__(
        self,
        max_columns: int = 3,
        preview_size: Tuple[int, int] = (320, 240),
        show_fps: bool = True,
        show_status: bool = True,
    ):
        """
        Initialize camera panel.
        
        Args:
            max_columns: Maximum columns in grid layout
            preview_size: Size for each camera preview (width, height)
            show_fps: Whether to show FPS overlay
            show_status: Whether to show connection status
        """
        self.max_columns = max_columns
        self.preview_size = preview_size
        self.show_fps = show_fps
        self.show_status = show_status
        
        # Camera tracking
        self._cameras: Dict[int, CameraFeedInfo] = {}
        self._frame_data: Dict[int, np.ndarray] = {}
        self._texture_ids: Dict[int, int] = {}
        
        # DearPyGui references (set during setup)
        self._panel_id: Optional[int] = None
        self._texture_registry: Optional[int] = None
        
        # Layout
        self._columns = 1
        self._rows = 1
        
        # Expanded view
        self._expanded_camera: Optional[int] = None
        
        # Click callbacks
        self._click_callbacks: List[Callable[[int], None]] = []
        
        # FPS tracking
        self._fps_history: Dict[int, List[float]] = {}
        self._last_frame_times: Dict[int, float] = {}
    
    def add_camera(
        self,
        camera_id: int,
        name: Optional[str] = None,
        resolution: Tuple[int, int] = (1280, 720),
    ) -> None:
        """
        Add a camera to the panel.
        
        Args:
            camera_id: Unique camera identifier
            name: Optional display name
            resolution: Camera resolution
        """
        if name is None:
            name = f"Camera {camera_id}"
        
        self._cameras[camera_id] = CameraFeedInfo(
            camera_id=camera_id,
            name=name,
            resolution=resolution,
            fps=0.0,
            is_connected=False,
            last_frame_time=0.0,
            frame_count=0,
        )
        
        self._fps_history[camera_id] = []
        self._update_layout()
    
    def remove_camera(self, camera_id: int) -> None:
        """Remove a camera from the panel."""
        if camera_id in self._cameras:
            del self._cameras[camera_id]
            if camera_id in self._frame_data:
                del self._frame_data[camera_id]
            if camera_id in self._fps_history:
                del self._fps_history[camera_id]
            self._update_layout()
    
    def _update_layout(self) -> None:
        """Recalculate grid layout based on camera count."""
        num_cameras = len(self._cameras)
        if num_cameras == 0:
            self._columns = 1
            self._rows = 1
        else:
            self._columns = min(num_cameras, self.max_columns)
            self._rows = (num_cameras + self._columns - 1) // self._columns
    
    def update_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update the frame for a camera.
        
        Args:
            camera_id: Camera to update
            frame: BGR image array
            timestamp: Frame timestamp
        """
        if camera_id not in self._cameras:
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        # Store frame data (resized to preview size)
        import cv2
        resized = cv2.resize(frame, self.preview_size)
        # Convert BGR to RGBA for DearPyGui
        rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        self._frame_data[camera_id] = rgba
        
        # Update camera info
        info = self._cameras[camera_id]
        info.is_connected = True
        info.frame_count += 1
        
        # Calculate FPS
        if camera_id in self._last_frame_times:
            dt = timestamp - self._last_frame_times[camera_id]
            if dt > 0:
                fps = 1.0 / dt
                self._fps_history[camera_id].append(fps)
                # Keep only last 30 values
                if len(self._fps_history[camera_id]) > 30:
                    self._fps_history[camera_id] = self._fps_history[camera_id][-30:]
                info.fps = np.mean(self._fps_history[camera_id])
        
        self._last_frame_times[camera_id] = timestamp
        info.last_frame_time = timestamp
    
    def mark_disconnected(self, camera_id: int) -> None:
        """Mark a camera as disconnected."""
        if camera_id in self._cameras:
            self._cameras[camera_id].is_connected = False
    
    def get_camera_ids(self) -> List[int]:
        """Get list of camera IDs."""
        return list(self._cameras.keys())
    
    def get_camera_info(self, camera_id: int) -> Optional[CameraFeedInfo]:
        """Get info for a specific camera."""
        return self._cameras.get(camera_id)
    
    def get_grid_size(self) -> Tuple[int, int]:
        """Get grid dimensions (columns, rows)."""
        return (self._columns, self._rows)
    
    def get_total_size(self) -> Tuple[int, int]:
        """Get total panel size in pixels."""
        return (
            self._columns * self.preview_size[0],
            self._rows * self.preview_size[1],
        )
    
    def add_click_callback(self, callback: Callable[[int], None]) -> None:
        """Add callback for camera click events."""
        self._click_callbacks.append(callback)
    
    def toggle_expand(self, camera_id: int) -> None:
        """Toggle expanded view for a camera."""
        if self._expanded_camera == camera_id:
            self._expanded_camera = None
        else:
            self._expanded_camera = camera_id
    
    @property
    def expanded_camera(self) -> Optional[int]:
        """Get currently expanded camera ID, or None."""
        return self._expanded_camera
    
    def get_frame_rgba(self, camera_id: int) -> Optional[np.ndarray]:
        """Get RGBA frame data for a camera (for texture updates)."""
        return self._frame_data.get(camera_id)
    
    def create_placeholder_frame(self, camera_id: int) -> np.ndarray:
        """Create a placeholder frame for disconnected camera."""
        import cv2
        
        placeholder = np.zeros(
            (self.preview_size[1], self.preview_size[0], 4),
            dtype=np.uint8
        )
        placeholder[:, :, 3] = 255  # Full alpha
        placeholder[:, :, :3] = 30  # Dark gray
        
        # Add text
        name = self._cameras.get(camera_id, CameraFeedInfo(camera_id, f"Camera {camera_id}", (0, 0), 0, False, 0, 0)).name
        cv2.putText(
            placeholder,
            name,
            (10, self.preview_size[1] // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 128, 128, 255),
            1,
        )
        cv2.putText(
            placeholder,
            "No Signal",
            (10, self.preview_size[1] // 2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 200, 255),
            1,
        )
        
        return placeholder
    
    def get_dpg_frames_flat(self) -> Dict[int, np.ndarray]:
        """
        Get frames flattened for DearPyGui texture updates.
        
        Returns:
            Dict mapping camera_id to flattened RGBA array (0-1 float range)
        """
        result = {}
        for cam_id in self._cameras:
            if cam_id in self._frame_data:
                # Convert to float 0-1 and flatten
                frame = self._frame_data[cam_id].astype(np.float32) / 255.0
                result[cam_id] = frame.flatten()
            else:
                # Placeholder
                placeholder = self.create_placeholder_frame(cam_id)
                frame = placeholder.astype(np.float32) / 255.0
                result[cam_id] = frame.flatten()
        return result
