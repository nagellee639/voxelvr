"""
Tracker Rotation Estimation

Estimates orientations for VRChat trackers based on bone directions.
Each tracker's rotation is derived from the positions of connected joints.

VRChat expects Euler angles in degrees, applied in Z-X-Y order.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# COCO keypoint indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, returning zero vector if length is too small."""
    length = np.linalg.norm(v)
    if length < 1e-6:
        return np.zeros_like(v)
    return v / length


def safe_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product with normalization."""
    result = np.cross(a, b)
    return normalize(result)


def rotation_matrix_to_euler_zxy(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles in Z-X-Y order (VRChat convention).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        (x, y, z) Euler angles in degrees
    """
    # For Z-X-Y order:
    # R = Ry @ Rx @ Rz
    
    # Extract angles
    # From the rotation matrix for Z-X-Y:
    # R[1,0] = cos(x)*sin(z) + sin(x)*sin(y)*cos(z)
    # R[1,1] = cos(x)*cos(z) - sin(x)*sin(y)*sin(z)  
    # R[1,2] = -sin(x)*cos(y)
    # R[0,2] = cos(x)*sin(y)*cos(z) + sin(x)*sin(z)
    # R[2,2] = cos(x)*cos(y)
    
    # Check for gimbal lock
    sin_x = -R[1, 2]
    sin_x = np.clip(sin_x, -1.0, 1.0)
    
    if abs(sin_x) > 0.9999:
        # Gimbal lock - x is ±90°
        x = np.arcsin(sin_x)
        y = 0
        z = np.arctan2(-R[0, 1], R[0, 0])
    else:
        x = np.arcsin(sin_x)
        y = np.arctan2(R[0, 2], R[2, 2])
        z = np.arctan2(R[1, 0], R[1, 1])
    
    return np.degrees(np.array([x, y, z]))


def build_rotation_matrix(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build a rotation matrix from forward and up vectors.
    
    Uses Unity's convention: +Z forward, +Y up, +X right (left-handed).
    
    Args:
        forward: Direction the tracker is "facing"
        up: Direction pointing "up" for the tracker
        
    Returns:
        3x3 rotation matrix
    """
    forward = normalize(forward)
    up = normalize(up)
    
    # Ensure orthogonality
    right = safe_cross(up, forward)
    if np.linalg.norm(right) < 1e-6:
        # forward and up are parallel, pick arbitrary right
        right = np.array([1, 0, 0]) if abs(forward[0]) < 0.9 else np.array([0, 1, 0])
        right = normalize(right - np.dot(right, forward) * forward)
    
    # Recalculate up to ensure orthogonality
    up = safe_cross(forward, right)
    
    # Build rotation matrix (columns are right, up, forward)
    R = np.column_stack([right, up, forward])
    
    return R


def estimate_hip_rotation(
    positions: np.ndarray,
    valid_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Estimate hip (pelvis) rotation.
    
    Forward is perpendicular to the hip line, pointing the direction the body faces.
    Up is world up (+Y).
    """
    if not (valid_mask[LEFT_HIP] and valid_mask[RIGHT_HIP]):
        return None
    
    left_hip = positions[LEFT_HIP]
    right_hip = positions[RIGHT_HIP]
    
    # Hip direction (left to right)
    hip_vec = normalize(right_hip - left_hip)
    
    # World up
    world_up = np.array([0, 1, 0])
    
    # Forward is perpendicular to hip line and world up
    forward = safe_cross(hip_vec, world_up)
    
    if np.linalg.norm(forward) < 1e-6:
        return np.array([0, 0, 0])
    
    R = build_rotation_matrix(forward, world_up)
    return rotation_matrix_to_euler_zxy(R)


def estimate_chest_rotation(
    positions: np.ndarray,
    valid_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Estimate chest (spine) rotation.
    
    Similar to hip but uses shoulder line.
    Optionally tilts based on shoulder-hip angle.
    """
    if not (valid_mask[LEFT_SHOULDER] and valid_mask[RIGHT_SHOULDER]):
        return None
    
    left_shoulder = positions[LEFT_SHOULDER]
    right_shoulder = positions[RIGHT_SHOULDER]
    
    # Shoulder direction
    shoulder_vec = normalize(right_shoulder - left_shoulder)
    
    # Calculate up direction
    # If we have hip data, use spine direction
    if valid_mask[LEFT_HIP] and valid_mask[RIGHT_HIP]:
        hip_center = (positions[LEFT_HIP] + positions[RIGHT_HIP]) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        spine_up = normalize(shoulder_center - hip_center)
    else:
        spine_up = np.array([0, 1, 0])
    
    # Forward is perpendicular to shoulder line
    forward = safe_cross(shoulder_vec, spine_up)
    
    if np.linalg.norm(forward) < 1e-6:
        return np.array([0, 0, 0])
    
    R = build_rotation_matrix(forward, spine_up)
    return rotation_matrix_to_euler_zxy(R)


def estimate_foot_rotation(
    ankle_pos: np.ndarray,
    knee_pos: np.ndarray,
    is_left: bool,
) -> np.ndarray:
    """
    Estimate foot (ankle) rotation.
    
    Up points along the lower leg (toward knee).
    Forward is perpendicular, in the plane of the leg movement.
    """
    # Lower leg direction (up for the foot)
    lower_leg = normalize(knee_pos - ankle_pos)
    
    # World up for reference
    world_up = np.array([0, 1, 0])
    
    # Right is perpendicular to lower leg and world up
    # This puts forward in the sagittal plane (forward/back movement)
    right = safe_cross(lower_leg, world_up)
    
    if np.linalg.norm(right) < 1e-6:
        # Leg is vertical, use world forward
        right = np.array([1, 0, 0])
    
    # Forward is perpendicular to lower leg and right
    forward = safe_cross(right, lower_leg)
    
    # Flip for left foot to maintain symmetry
    if is_left:
        right = -right
    
    R = build_rotation_matrix(forward, lower_leg)
    return rotation_matrix_to_euler_zxy(R)


def estimate_knee_rotation(
    hip_pos: np.ndarray,
    knee_pos: np.ndarray,
    ankle_pos: np.ndarray,
    is_left: bool,
) -> np.ndarray:
    """
    Estimate knee rotation.
    
    Forward points in the direction the knee bends (perpendicular to leg plane).
    Up is along the thigh direction.
    """
    # Upper and lower leg vectors
    upper_leg = normalize(knee_pos - hip_pos)
    lower_leg = normalize(ankle_pos - knee_pos)
    
    # Up is blend of upper/lower leg (pointing up the leg)
    up = normalize(-upper_leg - lower_leg)
    
    # Forward is perpendicular to the leg plane (direction knee bends)
    leg_plane_normal = safe_cross(upper_leg, lower_leg)
    
    if np.linalg.norm(leg_plane_normal) < 1e-6:
        # Leg is straight, estimate from world
        world_forward = np.array([0, 0, 1])
        leg_plane_normal = safe_cross(upper_leg, world_forward)
    
    forward = leg_plane_normal if not is_left else -leg_plane_normal
    
    R = build_rotation_matrix(forward, up)
    return rotation_matrix_to_euler_zxy(R)


def estimate_elbow_rotation(
    shoulder_pos: np.ndarray,
    elbow_pos: np.ndarray,
    wrist_pos: np.ndarray,
    is_left: bool,
) -> np.ndarray:
    """
    Estimate elbow rotation.
    
    Similar to knee - forward is the bend direction.
    """
    # Upper and lower arm vectors
    upper_arm = normalize(elbow_pos - shoulder_pos)
    lower_arm = normalize(wrist_pos - elbow_pos)
    
    # Up points along the arm (toward shoulder)
    up = normalize(-upper_arm)
    
    # Forward is perpendicular to arm plane
    arm_plane_normal = safe_cross(upper_arm, lower_arm)
    
    if np.linalg.norm(arm_plane_normal) < 1e-6:
        # Arm is straight, estimate from world up
        world_up = np.array([0, 1, 0])
        arm_plane_normal = safe_cross(upper_arm, world_up)
    
    forward = arm_plane_normal if is_left else -arm_plane_normal
    
    R = build_rotation_matrix(forward, up)
    return rotation_matrix_to_euler_zxy(R)


@dataclass
class TrackerRotations:
    """Container for all tracker rotations."""
    hip: Optional[np.ndarray] = None
    chest: Optional[np.ndarray] = None
    left_foot: Optional[np.ndarray] = None
    right_foot: Optional[np.ndarray] = None
    left_knee: Optional[np.ndarray] = None
    right_knee: Optional[np.ndarray] = None
    left_elbow: Optional[np.ndarray] = None
    right_elbow: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Tuple[float, float, float]]:
        """Convert to dictionary with tuples."""
        result = {}
        for name in ['hip', 'chest', 'left_foot', 'right_foot', 
                     'left_knee', 'right_knee', 'left_elbow', 'right_elbow']:
            rot = getattr(self, name)
            if rot is not None:
                result[name] = tuple(rot)
            else:
                result[name] = (0.0, 0.0, 0.0)
        return result


def estimate_all_rotations(
    positions: np.ndarray,
    valid_mask: np.ndarray,
) -> TrackerRotations:
    """
    Estimate rotations for all VRChat trackers.
    
    Args:
        positions: (17, 3) joint positions in VRChat coordinate space
        valid_mask: (17,) boolean mask for valid joints
        
    Returns:
        TrackerRotations with Euler angles for each tracker
    """
    rotations = TrackerRotations()
    
    # Hip
    rotations.hip = estimate_hip_rotation(positions, valid_mask)
    
    # Chest
    rotations.chest = estimate_chest_rotation(positions, valid_mask)
    
    # Left foot
    if valid_mask[LEFT_ANKLE] and valid_mask[LEFT_KNEE]:
        rotations.left_foot = estimate_foot_rotation(
            positions[LEFT_ANKLE],
            positions[LEFT_KNEE],
            is_left=True,
        )
    
    # Right foot
    if valid_mask[RIGHT_ANKLE] and valid_mask[RIGHT_KNEE]:
        rotations.right_foot = estimate_foot_rotation(
            positions[RIGHT_ANKLE],
            positions[RIGHT_KNEE],
            is_left=False,
        )
    
    # Left knee
    if valid_mask[LEFT_HIP] and valid_mask[LEFT_KNEE] and valid_mask[LEFT_ANKLE]:
        rotations.left_knee = estimate_knee_rotation(
            positions[LEFT_HIP],
            positions[LEFT_KNEE],
            positions[LEFT_ANKLE],
            is_left=True,
        )
    
    # Right knee
    if valid_mask[RIGHT_HIP] and valid_mask[RIGHT_KNEE] and valid_mask[RIGHT_ANKLE]:
        rotations.right_knee = estimate_knee_rotation(
            positions[RIGHT_HIP],
            positions[RIGHT_KNEE],
            positions[RIGHT_ANKLE],
            is_left=False,
        )
    
    # Left elbow
    if valid_mask[LEFT_SHOULDER] and valid_mask[LEFT_ELBOW] and valid_mask[LEFT_WRIST]:
        rotations.left_elbow = estimate_elbow_rotation(
            positions[LEFT_SHOULDER],
            positions[LEFT_ELBOW],
            positions[LEFT_WRIST],
            is_left=True,
        )
    
    # Right elbow
    if valid_mask[RIGHT_SHOULDER] and valid_mask[RIGHT_ELBOW] and valid_mask[RIGHT_WRIST]:
        rotations.right_elbow = estimate_elbow_rotation(
            positions[RIGHT_SHOULDER],
            positions[RIGHT_ELBOW],
            positions[RIGHT_WRIST],
            is_left=False,
        )
    
    return rotations


class RotationFilter:
    """
    Temporal filter for rotations to prevent jittery spinning.
    
    Uses exponential smoothing on Euler angles with wrap-around handling.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize rotation filter.
        
        Args:
            alpha: Smoothing factor (0-1). Lower = more smoothing.
        """
        self.alpha = alpha
        self.prev_rotations: Dict[str, np.ndarray] = {}
    
    def reset(self) -> None:
        """Reset filter state."""
        self.prev_rotations = {}
    
    def filter(self, rotations: TrackerRotations) -> TrackerRotations:
        """
        Apply temporal filtering to rotations.
        
        Args:
            rotations: Current frame rotations
            
        Returns:
            Filtered rotations
        """
        filtered = TrackerRotations()
        
        for name in ['hip', 'chest', 'left_foot', 'right_foot',
                     'left_knee', 'right_knee', 'left_elbow', 'right_elbow']:
            current = getattr(rotations, name)
            
            if current is None:
                setattr(filtered, name, None)
                continue
            
            current = np.array(current)
            
            if name in self.prev_rotations:
                prev = self.prev_rotations[name]
                
                # Handle angle wrap-around (e.g., 179° to -179°)
                diff = current - prev
                diff = np.where(diff > 180, diff - 360, diff)
                diff = np.where(diff < -180, diff + 360, diff)
                
                # Apply smoothing
                smoothed = prev + self.alpha * diff
                
                # Normalize to -180 to 180
                smoothed = np.where(smoothed > 180, smoothed - 360, smoothed)
                smoothed = np.where(smoothed < -180, smoothed + 360, smoothed)
            else:
                smoothed = current
            
            self.prev_rotations[name] = smoothed
            setattr(filtered, name, smoothed)
        
        return filtered
