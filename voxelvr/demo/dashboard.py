"""
Tracking Dashboard

Multi-panel dashboard showing:
- Camera feeds with 2D keypoint overlays
- Performance metrics (FPS, latency)
- Tracking quality indicators
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from ..pose.detector_2d import Keypoints2D, COCO_SKELETON


@dataclass
class PerformanceMetrics:
    """Tracking performance metrics."""
    capture_fps: float = 0.0
    detection_fps: float = 0.0
    triangulation_fps: float = 0.0
    total_fps: float = 0.0
    
    capture_latency_ms: float = 0.0
    detection_latency_ms: float = 0.0
    triangulation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    num_valid_joints: int = 0
    avg_confidence: float = 0.0


class TrackingDashboard:
    """
    Real-time dashboard for monitoring tracking performance.
    
    Shows multiple camera feeds with overlays and performance metrics.
    """
    
    def __init__(
        self,
        camera_ids: List[int],
        window_name: str = "VoxelVR Dashboard",
        preview_size: Tuple[int, int] = (640, 360),
        max_columns: int = 3,
    ):
        """
        Initialize the dashboard.
        
        Args:
            camera_ids: List of camera IDs to display
            window_name: OpenCV window name
            preview_size: Size for each camera preview
            max_columns: Maximum columns in grid layout
        """
        self.camera_ids = camera_ids
        self.window_name = window_name
        self.preview_size = preview_size
        self.max_columns = max_columns
        
        # Calculate grid layout
        self.num_cameras = len(camera_ids)
        self.columns = min(self.num_cameras, max_columns)
        self.rows = (self.num_cameras + self.columns - 1) // self.columns
        
        # Performance tracking
        self.fps_history: deque = deque(maxlen=60)
        self.latency_history: deque = deque(maxlen=60)
        self.last_update_time = time.time()
        
        # Current metrics
        self.metrics = PerformanceMetrics()
        
        # Recording state
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
    
    def update(
        self,
        frames: Dict[int, np.ndarray],
        keypoints: Optional[Dict[int, Keypoints2D]] = None,
        metrics: Optional[PerformanceMetrics] = None,
    ) -> np.ndarray:
        """
        Update the dashboard with new frames and data.
        
        Args:
            frames: Dictionary mapping camera_id to BGR image
            keypoints: Optional 2D keypoints for each camera
            metrics: Optional performance metrics
            
        Returns:
            Dashboard image
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        
        if metrics:
            self.metrics = metrics
        
        # Create camera previews
        previews = []
        for cam_id in self.camera_ids:
            if cam_id in frames:
                preview = self._create_camera_preview(
                    frames[cam_id],
                    cam_id,
                    keypoints.get(cam_id) if keypoints else None,
                )
            else:
                preview = self._create_placeholder(cam_id)
            previews.append(preview)
        
        # Arrange in grid
        dashboard = self._create_grid(previews)
        
        # Add metrics panel
        dashboard = self._add_metrics_panel(dashboard)
        
        # Add recording indicator
        if self.is_recording:
            cv2.circle(dashboard, (dashboard.shape[1] - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(dashboard, "REC", (dashboard.shape[1] - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Record frame if recording
        if self.is_recording and self.video_writer:
            self.video_writer.write(dashboard)
        
        return dashboard
    
    def _create_camera_preview(
        self,
        frame: np.ndarray,
        camera_id: int,
        keypoints: Optional[Keypoints2D],
    ) -> np.ndarray:
        """Create a camera preview with overlays."""
        # Resize frame
        preview = cv2.resize(frame, self.preview_size)
        
        # Draw keypoints if available
        if keypoints:
            scale_x = self.preview_size[0] / keypoints.image_width
            scale_y = self.preview_size[1] / keypoints.image_height
            
            # Draw skeleton
            for start_idx, end_idx in COCO_SKELETON:
                if (keypoints.confidences[start_idx] >= keypoints.threshold and
                    keypoints.confidences[end_idx] >= keypoints.threshold):
                    pt1 = (
                        int(keypoints.positions[start_idx, 0] * scale_x),
                        int(keypoints.positions[start_idx, 1] * scale_y),
                    )
                    pt2 = (
                        int(keypoints.positions[end_idx, 0] * scale_x),
                        int(keypoints.positions[end_idx, 1] * scale_y),
                    )
                    cv2.line(preview, pt1, pt2, (0, 255, 0), 2)
            
            # Draw joints
            for i, (pos, conf) in enumerate(zip(keypoints.positions, keypoints.confidences)):
                if conf >= keypoints.threshold:
                    pt = (int(pos[0] * scale_x), int(pos[1] * scale_y))
                    color_intensity = int(min(255, conf * 255))
                    color = (0, color_intensity, 255 - color_intensity)
                    cv2.circle(preview, pt, 4, color, -1)
        
        # Add camera label
        cv2.rectangle(preview, (0, 0), (120, 30), (0, 0, 0), -1)
        cv2.putText(preview, f"Camera {camera_id}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add confidence indicator if keypoints available
        if keypoints:
            valid_count = np.sum(keypoints.confidences >= keypoints.threshold)
            avg_conf = np.mean(keypoints.confidences[keypoints.confidences >= keypoints.threshold])
            status = f"{valid_count}/17 | {avg_conf:.2f}"
            cv2.putText(preview, status, (5, self.preview_size[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return preview
    
    def _create_placeholder(self, camera_id: int) -> np.ndarray:
        """Create a placeholder for missing camera."""
        placeholder = np.zeros((self.preview_size[1], self.preview_size[0], 3), dtype=np.uint8)
        placeholder[:] = (30, 30, 30)
        
        cv2.putText(placeholder, f"Camera {camera_id}", 
                   (self.preview_size[0] // 2 - 50, self.preview_size[1] // 2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        cv2.putText(placeholder, "No Signal",
                   (self.preview_size[0] // 2 - 40, self.preview_size[1] // 2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        
        return placeholder
    
    def _create_grid(self, previews: List[np.ndarray]) -> np.ndarray:
        """Arrange previews in a grid."""
        # Pad to fill grid
        while len(previews) < self.rows * self.columns:
            placeholder = np.zeros((self.preview_size[1], self.preview_size[0], 3), dtype=np.uint8)
            previews.append(placeholder)
        
        rows = []
        for r in range(self.rows):
            row_images = previews[r * self.columns:(r + 1) * self.columns]
            rows.append(np.hstack(row_images))
        
        return np.vstack(rows)
    
    def _add_metrics_panel(self, dashboard: np.ndarray) -> np.ndarray:
        """Add metrics panel to bottom of dashboard."""
        panel_height = 80
        panel = np.zeros((panel_height, dashboard.shape[1], 3), dtype=np.uint8)
        panel[:] = (25, 25, 35)
        
        # Calculate average FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Draw metrics
        y = 25
        metrics_text = [
            f"Dashboard FPS: {avg_fps:.1f}",
            f"Tracking FPS: {self.metrics.total_fps:.1f}",
            f"Latency: {self.metrics.total_latency_ms:.1f}ms",
            f"Valid Joints: {self.metrics.num_valid_joints}/17",
            f"Confidence: {self.metrics.avg_confidence:.2f}",
        ]
        
        x = 10
        for text in metrics_text:
            cv2.putText(panel, text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            x += 200
            if x > dashboard.shape[1] - 200:
                x = 10
                y += 25
        
        # Draw FPS graph
        graph_x = dashboard.shape[1] - 150
        graph_y = 10
        graph_w = 140
        graph_h = 60
        
        cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                     (50, 50, 60), -1)
        cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                     (80, 80, 90), 1)
        
        if len(self.fps_history) > 1:
            fps_list = list(self.fps_history)
            max_fps = max(fps_list) or 1
            
            points = []
            for i, fps in enumerate(fps_list):
                px = graph_x + int(i * graph_w / len(fps_list))
                py = graph_y + graph_h - int(fps / max_fps * (graph_h - 10))
                points.append((px, py))
            
            for i in range(len(points) - 1):
                cv2.line(panel, points[i], points[i + 1], (0, 200, 100), 1)
        
        cv2.putText(panel, "FPS", (graph_x + 5, graph_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        return np.vstack([dashboard, panel])
    
    def show(
        self,
        frames: Dict[int, np.ndarray],
        keypoints: Optional[Dict[int, Keypoints2D]] = None,
        metrics: Optional[PerformanceMetrics] = None,
    ) -> int:
        """
        Show the dashboard and return key pressed.
        
        Returns:
            Key code pressed, or -1 if none
        """
        dashboard = self.update(frames, keypoints, metrics)
        cv2.imshow(self.window_name, dashboard)
        return cv2.waitKey(1) & 0xFF
    
    def start_recording(self, output_path: str, fps: float = 30.0) -> bool:
        """Start recording the dashboard."""
        if self.is_recording:
            return False
        
        total_width = self.columns * self.preview_size[0]
        total_height = self.rows * self.preview_size[1] + 80  # +80 for metrics
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (total_width, total_height)
        )
        
        if self.video_writer.isOpened():
            self.is_recording = True
            print(f"Recording started: {output_path}")
            return True
        return False
    
    def stop_recording(self) -> None:
        """Stop recording."""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("Recording stopped")
    
    def close(self) -> None:
        """Close the dashboard window."""
        self.stop_recording()
        cv2.destroyWindow(self.window_name)
