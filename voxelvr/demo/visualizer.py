"""
3D Skeleton Visualizer

Real-time 3D visualization of tracked skeleton using PyVista or Open3D.
Provides immediate visual feedback for testing and debugging.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import time


# COCO skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Colors for different body parts
BODY_PART_COLORS = {
    "head": (0.8, 0.8, 0.2),  # Yellow
    "torso": (0.2, 0.8, 0.2),  # Green
    "left_arm": (0.2, 0.2, 0.8),  # Blue
    "right_arm": (0.8, 0.2, 0.2),  # Red
    "left_leg": (0.2, 0.6, 0.8),  # Cyan
    "right_leg": (0.8, 0.4, 0.2),  # Orange
}

# Map skeleton segments to body parts
SEGMENT_TO_PART = {
    (0, 1): "head", (0, 2): "head", (1, 3): "head", (2, 4): "head",
    (5, 6): "torso", (5, 11): "torso", (6, 12): "torso", (11, 12): "torso",
    (5, 7): "left_arm", (7, 9): "left_arm",
    (6, 8): "right_arm", (8, 10): "right_arm",
    (11, 13): "left_leg", (13, 15): "left_leg",
    (12, 14): "right_leg", (14, 16): "right_leg",
}


def create_skeleton_mesh(
    positions: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    joint_radius: float = 0.03,
    bone_radius: float = 0.015,
) -> dict:
    """
    Create mesh data for skeleton visualization.
    
    Args:
        positions: (17, 3) joint positions
        valid_mask: (17,) boolean mask for valid joints
        joint_radius: Radius of joint spheres
        bone_radius: Radius of bone cylinders
        
    Returns:
        Dictionary with 'joints' and 'bones' mesh data
    """
    if valid_mask is None:
        valid_mask = np.ones(len(positions), dtype=bool)
    
    mesh_data = {
        'joints': [],  # List of (position, color, radius)
        'bones': [],   # List of (start, end, color, radius)
    }
    
    # Create joint spheres
    for i, (pos, valid) in enumerate(zip(positions, valid_mask)):
        if valid:
            # Determine color based on joint index
            if i <= 4:
                color = BODY_PART_COLORS["head"]
            elif i in [5, 6, 11, 12]:
                color = BODY_PART_COLORS["torso"]
            elif i in [7, 9]:
                color = BODY_PART_COLORS["left_arm"]
            elif i in [8, 10]:
                color = BODY_PART_COLORS["right_arm"]
            elif i in [13, 15]:
                color = BODY_PART_COLORS["left_leg"]
            elif i in [14, 16]:
                color = BODY_PART_COLORS["right_leg"]
            else:
                color = (0.5, 0.5, 0.5)
            
            mesh_data['joints'].append({
                'position': pos.copy(),
                'color': color,
                'radius': joint_radius,
            })
    
    # Create bone cylinders
    for start_idx, end_idx in COCO_SKELETON:
        if valid_mask[start_idx] and valid_mask[end_idx]:
            part = SEGMENT_TO_PART.get((start_idx, end_idx), "torso")
            color = BODY_PART_COLORS.get(part, (0.5, 0.5, 0.5))
            
            mesh_data['bones'].append({
                'start': positions[start_idx].copy(),
                'end': positions[end_idx].copy(),
                'color': color,
                'radius': bone_radius,
            })
    
    return mesh_data


class SkeletonVisualizer:
    """
    Real-time 3D skeleton visualization using PyVista.
    
    Provides an interactive 3D view of the tracked skeleton
    for testing and debugging without VRChat.
    """
    
    def __init__(
        self,
        window_size: Tuple[int, int] = (1024, 768),
        background_color: str = "#1a1a2e",
        show_floor: bool = True,
        show_axes: bool = True,
    ):
        """
        Initialize the visualizer.
        
        Args:
            window_size: Window dimensions (width, height)
            background_color: Background color as hex string
            show_floor: Show a floor grid
            show_axes: Show coordinate axes
        """
        self.window_size = window_size
        self.background_color = background_color
        self.show_floor = show_floor
        self.show_axes = show_axes
        
        self.plotter = None
        self.joint_actors = []
        self.bone_actors = []
        self.initialized = False
        
        # Performance tracking
        self.frame_times = []
        self.last_update_time = 0
    
    def initialize(self) -> bool:
        """
        Initialize the PyVista plotter.
        
        Returns:
            True if initialization successful
        """
        try:
            import pyvista as pv
        except ImportError:
            print("PyVista not installed. Run: pip install pyvista")
            return False
        
        # Create plotter with off_screen=False for interactive display
        self.plotter = pv.Plotter(
            window_size=self.window_size,
            title="VoxelVR - Skeleton Visualizer",
        )
        
        # Set background
        self.plotter.set_background(self.background_color)
        
        # Add floor grid
        if self.show_floor:
            floor = pv.Plane(
                center=(0, 0, 0),
                direction=(0, 1, 0),
                i_size=4, j_size=4,
                i_resolution=20, j_resolution=20,
            )
            self.plotter.add_mesh(
                floor, 
                color='#2d2d44',
                opacity=0.5,
                show_edges=True,
                edge_color='#3d3d54',
            )
        
        # Add axes
        if self.show_axes:
            self.plotter.add_axes(
                line_width=2,
                labels_off=False,
            )
        
        # Set camera position (looking at origin from front-right-above)
        self.plotter.camera_position = [
            (3, 2, 3),  # Camera position
            (0, 0.8, 0),  # Focal point (center of person)
            (0, 1, 0),  # Up vector
        ]
        
        self.initialized = True
        return True
    
    def update(
        self,
        positions: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        joint_radius: float = 0.03,
        bone_radius: float = 0.015,
    ) -> None:
        """
        Update the skeleton visualization.
        
        Args:
            positions: (17, 3) joint positions
            valid_mask: (17,) boolean mask for valid joints
            joint_radius: Radius of joint spheres
            bone_radius: Radius of bone cylinders
        """
        if not self.initialized:
            if not self.initialize():
                return
        
        import pyvista as pv
        
        # Track timing
        current_time = time.time()
        if self.last_update_time > 0:
            self.frame_times.append(current_time - self.last_update_time)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)
        self.last_update_time = current_time
        
        if valid_mask is None:
            valid_mask = np.ones(len(positions), dtype=bool)
        
        # Clear previous actors
        for actor in self.joint_actors + self.bone_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self.joint_actors = []
        self.bone_actors = []
        
        # Create mesh data
        mesh_data = create_skeleton_mesh(positions, valid_mask, joint_radius, bone_radius)
        
        # Add joints
        for joint in mesh_data['joints']:
            sphere = pv.Sphere(
                radius=joint['radius'],
                center=joint['position'],
            )
            actor = self.plotter.add_mesh(
                sphere,
                color=joint['color'],
                smooth_shading=True,
            )
            self.joint_actors.append(actor)
        
        # Add bones
        for bone in mesh_data['bones']:
            # Create cylinder between points
            start = bone['start']
            end = bone['end']
            direction = end - start
            length = np.linalg.norm(direction)
            
            if length > 0.001:
                center = (start + end) / 2
                cylinder = pv.Cylinder(
                    center=center,
                    direction=direction,
                    radius=bone['radius'],
                    height=length,
                )
                actor = self.plotter.add_mesh(
                    cylinder,
                    color=bone['color'],
                    smooth_shading=True,
                )
                self.bone_actors.append(actor)
        
        # Update the display
        self.plotter.update()
    
    def get_fps(self) -> float:
        """Get current frames per second."""
        if len(self.frame_times) < 2:
            return 0.0
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def show(self) -> None:
        """Show the interactive visualization window."""
        if self.plotter:
            self.plotter.show(interactive=True)
    
    def show_non_blocking(self) -> None:
        """Show window without blocking (for integration with other code)."""
        if self.plotter:
            self.plotter.show(interactive=False, auto_close=False)
    
    def close(self) -> None:
        """Close the visualization window."""
        if self.plotter:
            self.plotter.close()
            self.plotter = None
            self.initialized = False


class SimpleOpenCVVisualizer:
    """
    Fallback 2D visualization using OpenCV when PyVista is not available.
    
    Shows orthographic projections of the 3D skeleton.
    """
    
    def __init__(
        self,
        window_size: Tuple[int, int] = (800, 600),
        scale: float = 200.0,  # pixels per meter
    ):
        self.window_size = window_size
        self.scale = scale
        self.center = (window_size[0] // 2, window_size[1] // 2)
    
    def update(
        self,
        positions: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create a 2D visualization image.
        
        Returns:
            BGR image showing the skeleton
        """
        import cv2
        
        if valid_mask is None:
            valid_mask = np.ones(len(positions), dtype=bool)
        
        # Create blank image
        img = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        img[:] = (30, 30, 46)  # Dark background
        
        # Project to 2D (XZ plane, looking from above)
        def project(pos_3d):
            x = int(self.center[0] + pos_3d[0] * self.scale)
            y = int(self.center[1] - pos_3d[2] * self.scale)  # Flip Z
            return (x, y)
        
        # Draw bones
        for start_idx, end_idx in COCO_SKELETON:
            if valid_mask[start_idx] and valid_mask[end_idx]:
                pt1 = project(positions[start_idx])
                pt2 = project(positions[end_idx])
                cv2.line(img, pt1, pt2, (0, 255, 100), 3)
        
        # Draw joints
        for i, (pos, valid) in enumerate(zip(positions, valid_mask)):
            if valid:
                pt = project(pos)
                cv2.circle(img, pt, 8, (0, 200, 255), -1)
                cv2.circle(img, pt, 8, (0, 100, 128), 2)
        
        # Add labels
        cv2.putText(img, "Top View (XZ)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        return img
    
    def show(self, positions: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> None:
        """Show the visualization in a window."""
        import cv2
        img = self.update(positions, valid_mask)
        cv2.imshow("VoxelVR - Skeleton (Top View)", img)
        cv2.waitKey(1)
