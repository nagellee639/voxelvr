"""
Multi-Camera Manager

Coordinates multiple cameras for synchronized capture,
handling timing alignment and calibration data.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .camera import Camera, CameraFrame
from ..config import CameraConfig, CameraIntrinsics, CameraExtrinsics, MultiCameraCalibration


class CameraManager:
    """
    Manages multiple cameras with synchronized capture.
    
    Provides methods to start/stop all cameras and get
    approximately synchronized frames from all views.
    """
    
    def __init__(self, camera_configs: Optional[List[CameraConfig]] = None):
        """
        Initialize camera manager.
        
        Args:
            camera_configs: List of camera configurations
        """
        self.cameras: Dict[int, Camera] = {}
        self.configs: Dict[int, CameraConfig] = {}
        
        if camera_configs:
            for config in camera_configs:
                self.add_camera(config)
    
    def add_camera(self, config: CameraConfig) -> bool:
        """
        Add a camera to the manager.
        
        Args:
            config: Camera configuration
            
        Returns:
            True if camera was added successfully
        """
        camera = Camera(
            camera_id=config.id,
            resolution=config.resolution,
            fps=config.fps,
        )
        self.cameras[config.id] = camera
        self.configs[config.id] = config
        return True
    
    @staticmethod
    def detect_cameras(max_cameras: int = 10) -> List[int]:
        """
        Auto-detect available cameras.
        
        Uses /dev/video* globbing on Linux to avoid OpenCV error spam.
        
        Args:
            max_cameras: Maximum number of cameras to check
            
        Returns:
            List of available camera IDs
        """
        import sys
        import glob
        import os
        
        available = []
        candidates = []
        
        # On Linux, check /dev/video* first to identify potential cameras
        if sys.platform.startswith('linux'):
            # Find all video devices
            devices = glob.glob('/dev/video*')
            for dev in devices:
                # Extract index
                try:
                    idx = int(dev.replace('/dev/video', ''))
                    candidates.append(idx)
                except ValueError:
                    pass
            candidates.sort()
            
            # Limit to max_count if specified (though valid indices can be high)
            if not candidates:
                # Fallback to naive iteration if glob fails
                candidates = list(range(max_cameras))
        else:
            candidates = list(range(max_cameras))
            
        # Suppress OpenCV errors during probing
        # We can't easily suppress C++ stderr from Python without pipes,
        # but we can try to rely on valid candidates to minimize failures.
        
        for i in candidates:
            # Skip if we already found enough?
            # No, let's find all available within reason.
            
            # Only check if index < max_cameras to prevent excessive probing of virtual devices?
            # Users might have plugged in cameras at high indices.
            
            # Use CAP_V4L2 backend explicitly on Linux if needed, or default.
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Read a frame to verify it's a real camera (some meta devices open but don't stream)
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
                
        return sorted(available)
    
    def start_all(self) -> bool:
        """Start all cameras."""
        success = True
        for camera in self.cameras.values():
            if not camera.start():
                print(f"Failed to start camera {camera.camera_id}")
                success = False
        return success
    
    def stop_all(self) -> None:
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop()
    
    def load_calibration(self, calibration: MultiCameraCalibration) -> None:
        """
        Apply calibration data to cameras.
        
        Args:
            calibration: Multi-camera calibration data
        """
        for cam_id, data in calibration.cameras.items():
            if cam_id in self.cameras:
                intr = data.get('intrinsics', {})
                camera_matrix = np.array(intr.get('camera_matrix', []))
                dist_coeffs = np.array(intr.get('distortion_coeffs', []))
                
                if camera_matrix.size > 0 and dist_coeffs.size > 0:
                    self.cameras[cam_id].set_calibration(camera_matrix, dist_coeffs)
    
    def get_synchronized_frames(
        self,
        max_time_diff: float = 0.05,  # 50ms max difference
        timeout: float = 0.1,
    ) -> Optional[Dict[int, CameraFrame]]:
        """
        Get approximately synchronized frames from all cameras.
        
        Args:
            max_time_diff: Maximum allowed timestamp difference in seconds
            timeout: Timeout for waiting for frames
            
        Returns:
            Dict mapping camera_id to CameraFrame, or None if sync failed
        """
        frames = {}
        timestamps = []
        
        # Collect latest frames from all cameras
        for cam_id, camera in self.cameras.items():
            frame = camera.get_frame(timeout=timeout)
            if frame is None:
                return None
            frames[cam_id] = frame
            timestamps.append(frame.timestamp)
        
        # Check if frames are synchronized enough
        if timestamps:
            time_spread = max(timestamps) - min(timestamps)
            if time_spread > max_time_diff:
                # Frames too far apart, retry with latest
                for cam_id, camera in self.cameras.items():
                    frame = camera.get_latest_frame()
                    if frame:
                        frames[cam_id] = frame
        
        return frames if len(frames) == len(self.cameras) else None
    
    def get_all_latest_frames(self) -> Dict[int, CameraFrame]:
        """Get the latest frame from each camera (non-blocking)."""
        frames = {}
        for cam_id, camera in self.cameras.items():
            frame = camera.get_latest_frame()
            if frame:
                frames[cam_id] = frame
        return frames
    
    @property
    def camera_ids(self) -> List[int]:
        """Get list of camera IDs."""
        return list(self.cameras.keys())
    
    @property
    def num_cameras(self) -> int:
        """Get number of cameras."""
        return len(self.cameras)
    
    def __enter__(self):
        self.start_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()


# Import cv2 at module level for detect_cameras
import cv2
