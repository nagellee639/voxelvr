"""
Coordinate System Transformation

Transforms poses from the camera/world coordinate system
to VRChat's Unity coordinate system.

Camera World Space (typical CV conventions):
- Often Z-forward, Y-down, X-right
- Meters as unit

Unity/VRChat Space:
- Y-up (positive Y is up)
- Left-handed coordinate system
- 1 unit = 1 meter
- Euler angles applied in order Z, X, Y
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CoordinateTransform:
    """
    Defines the transformation from camera world space to VRChat space.
    
    This includes:
    - Rotation to align axes
    - Scale factor
    - Origin offset
    """
    
    # Rotation matrix (3x3) from camera world to VRChat
    rotation: np.ndarray
    
    # Scale factor (typically 1.0 if both use meters)
    scale: float = 1.0
    
    # Offset to apply after rotation/scale (in VRChat space)
    offset: np.ndarray = None
    
    def __post_init__(self):
        if self.offset is None:
            self.offset = np.zeros(3)
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a single 3D point."""
        return self.rotation @ (point * self.scale) + self.offset
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple 3D points (N, 3)."""
        scaled = points * self.scale
        rotated = (self.rotation @ scaled.T).T
        return rotated + self.offset
    
    def transform_direction(self, direction: np.ndarray) -> np.ndarray:
        """Transform a direction vector (no translation)."""
        return self.rotation @ direction


def create_default_transform() -> CoordinateTransform:
    """
    Create a default coordinate transform.
    
    Assumes camera world has:
    - X-right, Y-down, Z-forward (typical OpenCV convention)
    
    Unity/VRChat needs:
    - X-right, Y-up, Z-forward (left-handed)
    
    So we flip Y axis.
    """
    rotation = np.array([
        [1, 0, 0],
        [0, -1, 0],  # Flip Y
        [0, 0, 1],
    ], dtype=np.float64)
    
    return CoordinateTransform(rotation=rotation)


def create_transform_from_calibration(
    floor_height: float = 0.0,
    forward_direction: Optional[np.ndarray] = None,
) -> CoordinateTransform:
    """
    Create coordinate transform from calibration data.
    
    Args:
        floor_height: Height of the floor in camera world coordinates
        forward_direction: Direction vector that should map to VRChat forward (+Z)
        
    Returns:
        CoordinateTransform for camera world to VRChat
    """
    # Default: assume camera world has Y-down, Z-forward
    rotation = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    
    # If forward direction specified, compute rotation to align it with +Z
    if forward_direction is not None:
        forward = forward_direction / np.linalg.norm(forward_direction)
        target_forward = np.array([0, 0, 1])
        
        # Compute rotation from forward to target_forward
        v = np.cross(forward, target_forward)
        c = np.dot(forward, target_forward)
        
        if np.linalg.norm(v) > 1e-6:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R_align = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
            rotation = R_align @ rotation
    
    # Offset to place floor at Y=0
    offset = np.array([0, -floor_height, 0])
    
    return CoordinateTransform(rotation=rotation, offset=offset)


def transform_pose_to_vrchat(
    positions_3d: np.ndarray,
    transform: CoordinateTransform,
    head_position: Optional[np.ndarray] = None,
    head_rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Transform full pose from camera world to VRChat space.
    
    Args:
        positions_3d: (N, 3) joint positions in camera world
        transform: Coordinate transform to apply
        head_position: Optional head position from VR headset (for alignment)
        head_rotation: Optional head rotation from VR headset
        
    Returns:
        Tuple of (transformed_positions, alignment_offset)
        alignment_offset can be added to future transforms for drift correction
    """
    # Apply basic transformation
    transformed = transform.transform_points(positions_3d)
    
    alignment_offset = None
    
    # If head position provided, we can compute alignment
    # VRChat aligns the OSC tracking space based on head data
    if head_position is not None:
        # Find the head position in our transformed data
        # COCO keypoint 0 is "nose", which is close to head
        our_head = transformed[0]  # Using nose as head proxy
        
        # Compute offset needed to align our head with VRChat head
        alignment_offset = head_position - our_head
        
        # Apply alignment
        transformed = transformed + alignment_offset
    
    return transformed, alignment_offset


def euler_to_rotation_matrix(euler_degrees: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    VRChat applies Euler angles in order: Z, X, Y
    
    Args:
        euler_degrees: (pitch, yaw, roll) or (x, y, z) in degrees
        
    Returns:
        3x3 rotation matrix
    """
    x, y, z = np.radians(euler_degrees)
    
    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    # VRChat order: Z, X, Y
    return Ry @ Rx @ Rz


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles.
    
    Returns angles in VRChat's Z, X, Y order.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        (x, y, z) Euler angles in degrees
    """
    # Extract Euler angles for Z, X, Y order
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.degrees(np.array([x, y, z]))


def estimate_body_orientation(
    positions_3d: np.ndarray,
    valid_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Estimate overall body facing direction from pose.
    
    Uses shoulder and hip positions to determine forward direction.
    
    Args:
        positions_3d: (17, 3) joint positions
        valid_mask: (17,) validity mask
        
    Returns:
        3D unit vector for forward direction, or None if insufficient data
    """
    # Try to use shoulders (indices 5, 6)
    if valid_mask[5] and valid_mask[6]:
        left_shoulder = positions_3d[5]
        right_shoulder = positions_3d[6]
        
        # Cross product of shoulder line with up vector gives forward
        shoulder_vec = right_shoulder - left_shoulder
        up = np.array([0, 1, 0])  # Assuming Y-up after transform
        
        forward = np.cross(up, shoulder_vec)
        norm = np.linalg.norm(forward)
        
        if norm > 1e-6:
            return forward / norm
    
    # Fallback: try hips (indices 11, 12)
    if valid_mask[11] and valid_mask[12]:
        left_hip = positions_3d[11]
        right_hip = positions_3d[12]
        
        hip_vec = right_hip - left_hip
        up = np.array([0, 1, 0])
        
        forward = np.cross(up, hip_vec)
        norm = np.linalg.norm(forward)
        
        if norm > 1e-6:
            return forward / norm
    
    return None
