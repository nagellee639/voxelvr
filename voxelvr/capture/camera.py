"""
Threaded Camera Capture

Provides non-blocking camera capture with frame buffering
for consistent multi-camera synchronization.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple
from dataclasses import dataclass
from queue import Queue, Empty


@dataclass
class CameraFrame:
    """A captured frame with metadata."""
    image: np.ndarray
    timestamp: float  # Time of capture
    frame_number: int
    camera_id: int


class Camera:
    """
    Threaded camera capture with frame buffering.
    
    Runs capture in background thread for consistent timing
    and provides latest frame on demand.
    """
    
    def __init__(
        self,
        camera_id: int,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        buffer_size: int = 2,
    ):
        """
        Initialize camera.
        
        Args:
            camera_id: OpenCV camera index
            resolution: (width, height) tuple
            fps: Target frames per second
            buffer_size: Number of frames to buffer
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.target_fps = fps
        self.buffer_size = buffer_size
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_queue: Queue[CameraFrame] = Queue(maxsize=buffer_size)
        self._latest_frame: Optional[CameraFrame] = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0
        
        # Calibration data (set after calibration)
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion_coeffs: Optional[np.ndarray] = None
        self.undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
    def start(self) -> bool:
        """Start the camera capture thread."""
        if self._running:
            return True
        
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            return False
        
        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Reduce buffering for lower latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual settings
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera {self.camera_id}: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop the camera capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None
    
    def _capture_loop(self) -> None:
        """Background capture loop."""
        while self._running:
            if self._cap is None:
                break
            
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            timestamp = time.time()
            self._frame_count += 1
            
            # Apply undistortion if calibrated
            if self.undistort_maps is not None:
                frame = cv2.remap(
                    frame, 
                    self.undistort_maps[0], 
                    self.undistort_maps[1], 
                    cv2.INTER_LINEAR
                )
            
            camera_frame = CameraFrame(
                image=frame,
                timestamp=timestamp,
                frame_number=self._frame_count,
                camera_id=self.camera_id,
            )
            
            # Update latest frame
            with self._frame_lock:
                self._latest_frame = camera_frame
            
            # Try to add to queue (non-blocking)
            try:
                # Remove old frame if queue is full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass
                self._frame_queue.put_nowait(camera_frame)
            except:
                pass
    
    def get_frame(self, timeout: float = 0.1) -> Optional[CameraFrame]:
        """
        Get the latest frame (blocks up to timeout).
        
        Args:
            timeout: Max seconds to wait for a frame
            
        Returns:
            CameraFrame or None if no frame available
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            # Return cached latest frame if available
            with self._frame_lock:
                return self._latest_frame
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get the most recent frame immediately (non-blocking)."""
        with self._frame_lock:
            return self._latest_frame
    
    def set_calibration(
        self,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
    ) -> None:
        """
        Set calibration data for undistortion.
        
        Args:
            camera_matrix: 3x3 intrinsic matrix
            distortion_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        
        # Precompute undistortion maps
        self.undistort_maps = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion_coeffs,
            None,
            camera_matrix,
            self.resolution,
            cv2.CV_32FC1,
        )
    
    @property
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running
    
    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
