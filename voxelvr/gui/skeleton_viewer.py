"""
3D Skeleton Viewer for VoxelVR Tracking

Renders a rotating 3D skeleton using matplotlib for real-time visualization
of tracked poses in the GUI.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple
import time


# COCO 17-keypoint skeleton connections
SKELETON_CONNECTIONS = [
    # Torso
    (5, 6),   # Shoulders
    (5, 11),  # Left shoulder to left hip
    (6, 12),  # Right shoulder to right hip
    (11, 12), # Hips
    
    # Left arm
    (5, 7), (7, 9),  # Shoulder -> elbow -> wrist
    
    # Right arm
    (6, 8), (8, 10),
    
    # Left leg
    (11, 13), (13, 15),  # Hip -> knee -> ankle
    
    # Right leg
    (12, 14), (14, 16),
    
    # Nose to shoulders (approximation)
    (0, 5), (0, 6),
]

# Joint names for COCO 17-keypoint model
JOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]


def get_tpose() -> dict:
    """
    Get a standard T-Pose for COCO 17-keypoint skeleton.
    
    Returns:
        Dictionary with 'positions', 'confidences', and 'valid' mask.
    """
    positions = np.zeros((17, 3), dtype=np.float32)
    # Hips (y=1.0)
    positions[11] = [-0.1, 1.0, 0.0]
    positions[12] = [ 0.1, 1.0, 0.0]
    # Legs
    positions[13] = [-0.1, 0.5, 0.0]
    positions[14] = [ 0.1, 0.5, 0.0]
    positions[15] = [-0.1, 0.0, 0.0]
    positions[16] = [ 0.1, 0.0, 0.0]
    # Torso (y=1.5)
    positions[5] = [-0.2, 1.5, 0.0]
    positions[6] = [ 0.2, 1.5, 0.0]
    # Arms
    positions[7] = [-0.45, 1.5, 0.0]
    positions[8] = [ 0.45, 1.5, 0.0]
    positions[9] = [-0.7, 1.5, 0.0]
    positions[10] = [ 0.7, 1.5, 0.0]
    # Head
    positions[0] = [0.0, 1.6, 0.0]
    positions[1] = [0.03, 1.65, 0.0]
    positions[2] = [-0.03, 1.65, 0.0]
    
    return {
        'positions': positions,
        'confidences': np.ones(17, dtype=np.float32),
        'valid': np.ones(17, dtype=bool)
    }


class SkeletonViewer:
    """3D skeleton visualization with automatic rotation."""
    
    def __init__(self, size: Tuple[int, int] = (400, 400)):
        """
        Initialize skeleton viewer.
        
        Args:
            size: (width, height) in pixels
        """
        self.size = size
        self.fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # View parameters
        self.rotation_angle = 0
        self.rotation_speed = 1.0  # degrees per frame
        self.auto_rotate = True
        self.elevation = 15  # degrees
        self.distance = 2.0  # meters
        
        # Skeleton display parameters
        self.show_joints = True
        self.show_bones = True
        self.show_axes = True
        self.show_floor = True
        
        # Last update time for frame rate control
        self._last_update = time.time()
        self._target_fps = 30.0
    
    def set_auto_rotate(self, enabled: bool) -> None:
        """Enable or disable automatic rotation."""
        self.auto_rotate = enabled
    
    def set_rotation_speed(self, speed: float) -> None:
        """
        Set rotation speed.
        
        Args:
            speed: Degrees per frame (0.1 to 5.0)
        """
        self.rotation_speed = np.clip(speed, 0.1, 5.0)
    
    def set_elevation(self, elevation: float) -> None:
        """
        Set viewing elevation.
        
        Args:
            elevation: Angle in degrees (-90 to 90)
        """
        self.elevation = np.clip(elevation, -90, 90)
    
    def set_distance(self, distance: float) -> None:
        """
        Set viewing distance.
        
        Args:
            distance: Distance in meters (0.5 to 5.0)
        """
        self.distance = np.clip(distance, 0.5, 5.0)
    
    def reset_view(self) -> None:
        """Reset view to default parameters."""
        self.rotation_angle = 0
        self.elevation = 15
        self.distance = 2.0
        self.rotation_speed = 1.0
    
    def _confidence_to_color(self, confidence: float) -> Tuple[float, float, float]:
        """
        Convert confidence score to color.
        
        Args:
            confidence: 0.0 to 1.0
            
        Returns:
            RGB tuple (0-1 scale)
        """
        if confidence > 0.7:
            return (0.2, 0.8, 0.2)  # Green - high confidence
        elif confidence > 0.3:
            return (0.9, 0.9, 0.2)  # Yellow - medium confidence
        else:
            return (0.9, 0.2, 0.2)  # Red - low confidence
    
    def render_skeleton(
        self,
        keypoints_3d: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render 3D skeleton and return as numpy array for DearPyGUI.
        
        Args:
            keypoints_3d: (17, 3) array of 3D joint positions in meters
            confidences: (17,) array of confidence scores (0-1)
            
        Returns:
            RGBA image as numpy array (H, W, 4) with values 0-1
        """
        # Frame rate limiting
        now = time.time()
        elapsed = now - self._last_update
        if elapsed < 1.0 / self._target_fps:
            time.sleep(1.0 / self._target_fps - elapsed)
        self._last_update = time.time()
        
        # Default confidences if not provided
        if confidences is None:
            confidences = np.ones(17)
        
        self.ax.clear()
        
        # Convert Y-up (Unity/VoxelVR) to Z-up (Matplotlib) for visualization
        # Input: (x, y, z) where Y is up
        # Output: (x, z, y) where Z is up (mapped to Y input)
        plot_points = np.zeros_like(keypoints_3d)
        plot_points[:, 0] = keypoints_3d[:, 0]  # X -> X
        plot_points[:, 1] = keypoints_3d[:, 2]  # Z -> Y (Depth)
        plot_points[:, 2] = keypoints_3d[:, 1]  # Y -> Z (Up)
        
        # Set view parameters
        self.ax.view_init(elev=self.elevation, azim=self.rotation_angle)
        
        # Plot joints
        if self.show_joints:
            for i, (point, conf) in enumerate(zip(plot_points, confidences)):
                if conf > 0.1:  # Only show if confident
                    color = self._confidence_to_color(conf)
                    self.ax.scatter(*point, c=[color], s=50, marker='o', edgecolors='black', linewidth=1)
        
        # Plot bones
        if self.show_bones:
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if confidences[start_idx] > 0.1 and confidences[end_idx] > 0.1:
                    points = np.array([plot_points[start_idx], plot_points[end_idx]])
                    
                    # Color based on average confidence
                    avg_conf = (confidences[start_idx] + confidences[end_idx]) / 2
                    color = self._confidence_to_color(avg_conf)
                    
                    self.ax.plot3D(*points.T, color=color, linewidth=2)
        
        # Plot floor grid (optional)
        if self.show_floor:
            grid_size = 2.0
            grid_res = 0.5
            x = np.arange(-grid_size, grid_size + grid_res, grid_res)
            y = np.arange(-grid_size, grid_size + grid_res, grid_res)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            self.ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.2, linewidth=0.5)
        
        # Set labels
        if self.show_axes:
            self.ax.set_xlabel('X (m)', fontsize=8)
            self.ax.set_ylabel('Depth (m)', fontsize=8)
            self.ax.set_zlabel('Height (m)', fontsize=8)
        else:
            self.ax.set_xlabel('')
            self.ax.set_ylabel('')
            self.ax.set_zlabel('')
        
        # Set equal aspect ratio and limits
        max_range = self.distance
        center = plot_points[confidences > 0.5].mean(axis=0) if np.any(confidences > 0.5) else np.zeros(3)
        
        # For Z-up, we generally want Z to start at 0
        self.ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
        self.ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
        self.ax.set_zlim([0, max_range])
        
        # Force aspect ratio to be equal (cubic) so it doesn't look distorted
        self.ax.set_box_aspect((1, 1, 1))
        
        # Update rotation
        if self.auto_rotate:
            self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360
        
        # Update rotation
        if self.auto_rotate:
            self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360
            
        # Render to image
        return self._fig_to_array()
    
    def render_placeholder(self, message: str = "No tracking data") -> np.ndarray:
        """
        Render a placeholder image when no skeleton data is available.
        
        Args:
            message: Message to display
            
        Returns:
            RGBA image as numpy array
        """
        self.ax.clear()
        self.ax.text(0.5, 0.5, 0.5, message,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12,
                    color='gray')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.set_zlim([0, 1])
        self.ax.axis('off')
        
        return self._fig_to_array()
    
    def _fig_to_array(self) -> np.ndarray:
        """
        Convert matplotlib figure to numpy array.
        
        Returns:
            RGBA array with shape (H, W, 4) and values 0-1
        """
        buf = BytesIO()
        self.fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf).convert('RGBA')
        
        # Resize to target size
        img = img.resize(self.size, Image.Resampling.LANCZOS)
        
        arr = np.array(img, dtype=np.float32) / 255.0
        buf.close()
        
        return arr
    
    def close(self) -> None:
        """Clean up matplotlib resources."""
        plt.close(self.fig)


if __name__ == "__main__":
    # Test the viewer
    import time
    
    viewer = SkeletonViewer(size=(600, 600))
    
    # Create a simple standing pose
    keypoints = np.array([
        [0.0, 0.0, 1.7],  # Nose
        [-0.05, -0.05, 1.72],  # Left Eye
        [0.05, -0.05, 1.72],   # Right Eye
        [-0.1, -0.05, 1.7],    # Left Ear
        [0.1, -0.05, 1.7],     # Right Ear
        [-0.2, 0.0, 1.4],      # Left Shoulder
        [0.2, 0.0, 1.4],       # Right Shoulder
        [-0.3, 0.0, 1.0],      # Left Elbow
        [0.3, 0.0, 1.0],       # Right Elbow
        [-0.35, 0.0, 0.7],     # Left Wrist
        [0.35, 0.0, 0.7],      # Right Wrist
        [-0.15, 0.0, 0.9],     # Left Hip
        [0.15, 0.0, 0.9],      # Right Hip
        [-0.15, 0.0, 0.45],    # Left Knee
        [0.15, 0.0, 0.45],     # Right Knee
        [-0.15, 0.0, 0.0],     # Left Ankle
        [0.15, 0.0, 0.0],      # Right Ankle
    ])
    
    confidences = np.random.uniform(0.8, 1.0, 17)
    
    # Render a few frames to test rotation
    for i in range(10):
        img = viewer.render_skeleton(keypoints, confidences)
        # Check if image is all white/transparent
        non_zero = np.count_nonzero(img)
        mean_val = np.mean(img)
        print(f"Frame {i}: Shape {img.shape}, Range [{img.min():.3f}, {img.max():.3f}], Mean: {mean_val:.3f}, Non-zero: {non_zero}")
        
        # Simulate rotation
        time.sleep(0.1)
    
    viewer.close()
    print("Test complete!")
