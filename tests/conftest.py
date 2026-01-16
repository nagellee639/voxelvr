"""
Shared test fixtures for VoxelVR testing.

Provides synthetic data, mock cameras, and performance measurement utilities.
"""

import pytest
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Generator
from dataclasses import dataclass
import json
import sys


# ============================================================================
# Synthetic Pose Data
# ============================================================================

def generate_t_pose() -> np.ndarray:
    """Generate a standard T-pose (17 COCO keypoints)."""
    return np.array([
        [0.0, 1.70, 0.0],      # 0: nose
        [-0.05, 1.72, 0.0],    # 1: left_eye
        [0.05, 1.72, 0.0],     # 2: right_eye
        [-0.10, 1.70, 0.0],    # 3: left_ear
        [0.10, 1.70, 0.0],     # 4: right_ear
        [-0.20, 1.50, 0.0],    # 5: left_shoulder
        [0.20, 1.50, 0.0],     # 6: right_shoulder
        [-0.50, 1.50, 0.0],    # 7: left_elbow (arms out)
        [0.50, 1.50, 0.0],     # 8: right_elbow
        [-0.80, 1.50, 0.0],    # 9: left_wrist
        [0.80, 1.50, 0.0],     # 10: right_wrist
        [-0.15, 1.00, 0.0],    # 11: left_hip
        [0.15, 1.00, 0.0],     # 12: right_hip
        [-0.15, 0.50, 0.0],    # 13: left_knee
        [0.15, 0.50, 0.0],     # 14: right_knee
        [-0.15, 0.05, 0.0],    # 15: left_ankle
        [0.15, 0.05, 0.0],     # 16: right_ankle
    ], dtype=np.float32)


def generate_walking_pose(t: float) -> np.ndarray:
    """Generate a walking pose at time t."""
    base = generate_t_pose()
    
    # Arms down at sides
    base[7] = [-0.25, 1.30, 0.0]   # left_elbow
    base[8] = [0.25, 1.30, 0.0]    # right_elbow
    base[9] = [-0.25, 1.10, 0.0]   # left_wrist
    base[10] = [0.25, 1.10, 0.0]   # right_wrist
    
    # Animate walking
    arm_swing = 0.15 * np.sin(t * 4)
    leg_swing = 0.20 * np.sin(t * 4)
    
    # Arm swing (opposite to legs)
    base[7, 2] = arm_swing
    base[9, 2] = arm_swing * 0.8
    base[8, 2] = -arm_swing
    base[10, 2] = -arm_swing * 0.8
    
    # Leg swing
    base[13, 2] = leg_swing
    base[15, 2] = leg_swing * 1.2
    base[14, 2] = -leg_swing
    base[16, 2] = -leg_swing * 1.2
    
    # Hip sway
    base[11:13, 0] += 0.02 * np.sin(t * 8)
    
    # Vertical bounce
    base[:, 1] += 0.02 * abs(np.sin(t * 8))
    
    return base


def generate_pose_sequence(
    duration: float = 5.0,
    fps: float = 30.0,
    pose_type: str = "walking"
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """Generate a sequence of poses."""
    num_frames = int(duration * fps)
    for i in range(num_frames):
        t = i / fps
        if pose_type == "walking":
            yield t, generate_walking_pose(t)
        elif pose_type == "stationary":
            yield t, generate_t_pose()
        else:
            yield t, generate_t_pose()


# ============================================================================
# Synthetic Camera Data
# ============================================================================

def generate_camera_intrinsics(
    width: int = 1280,
    height: int = 720,
    fov_degrees: float = 60.0
) -> np.ndarray:
    """Generate realistic camera intrinsic matrix."""
    fx = width / (2 * np.tan(np.radians(fov_degrees) / 2))
    fy = fx  # Square pixels
    cx = width / 2
    cy = height / 2
    
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)


def generate_camera_extrinsics(
    position: np.ndarray,
    look_at: np.ndarray = None
) -> np.ndarray:
    """
    Generate camera extrinsic matrix (4x4, camera-to-world).
    
    The matrix transforms points from camera space to world space.
    Uses OpenCV camera convention: X right, Y down, Z forward (into scene).
    
    Args:
        position: Camera position in world space
        look_at: Point the camera is looking at (default: origin at y=1)
    """
    if look_at is None:
        look_at = np.array([0.0, 1.0, 0.0])
    
    # Camera Z axis points FROM camera TOWARD target (into the scene)
    z_axis = look_at - position
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # World up vector
    world_up = np.array([0.0, 1.0, 0.0])
    
    # Handle case when camera is looking straight up/down
    if abs(np.dot(z_axis, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])
    
    # Camera X axis (right)
    x_axis = np.cross(z_axis, world_up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Camera Y axis (down in OpenCV, but we'll use up for simplicity)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Build rotation matrix: columns are camera axes in world coordinates
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # Build 4x4 camera-to-world transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    
    return T


def generate_multi_camera_setup(
    num_cameras: int = 3,
    radius: float = 3.0,
    height: float = 1.5
) -> List[Dict]:
    """
    Generate a ring of cameras around the origin.
    
    Returns list of camera configs with intrinsics and extrinsics.
    """
    cameras = []
    
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        position = np.array([x, height, z])
        
        intrinsics = generate_camera_intrinsics()
        extrinsics = generate_camera_extrinsics(position)
        
        cameras.append({
            'id': i,
            'position': position,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
        })
    
    return cameras


def project_3d_to_2d(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    add_noise: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) world coordinates
        intrinsics: (3, 3) camera matrix
        extrinsics: (4, 4) camera-to-world transform
        add_noise: Standard deviation of Gaussian noise in pixels
        
    Returns:
        Tuple of (points_2d, visibility_mask)
    """
    # World to camera transform
    T_world_to_cam = np.linalg.inv(extrinsics)
    R = T_world_to_cam[:3, :3]
    t = T_world_to_cam[:3, 3]
    
    # Transform to camera space
    points_cam = (R @ points_3d.T).T + t
    
    # Check visibility (in front of camera)
    visibility = points_cam[:, 2] > 0.1
    
    # Project to image
    points_2d = np.zeros((len(points_3d), 2))
    
    for i, (pc, vis) in enumerate(zip(points_cam, visibility)):
        if vis:
            projected = intrinsics @ pc
            points_2d[i] = projected[:2] / projected[2]
            
            if add_noise > 0:
                points_2d[i] += np.random.normal(0, add_noise, 2)
    
    return points_2d, visibility


# ============================================================================
# Performance Measurement
# ============================================================================

@dataclass
class PerformanceResult:
    """Result of a performance measurement."""
    name: str
    min_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float
    median_ms: float
    fps: float
    num_iterations: int
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'median_ms': self.median_ms,
            'fps': self.fps,
            'num_iterations': self.num_iterations,
        }


def measure_performance(
    func,
    *args,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    name: str = "operation",
    **kwargs
) -> PerformanceResult:
    """
    Measure the performance of a function.
    
    Returns timing statistics.
    """
    # Warmup
    for _ in range(warmup_iterations):
        func(*args, **kwargs)
    
    # Measure
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return PerformanceResult(
        name=name,
        min_ms=float(np.min(times)),
        max_ms=float(np.max(times)),
        mean_ms=float(np.mean(times)),
        std_ms=float(np.std(times)),
        median_ms=float(np.median(times)),
        fps=1000.0 / float(np.mean(times)),
        num_iterations=num_iterations,
    )


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def t_pose():
    """Provide a T-pose skeleton."""
    return generate_t_pose()


@pytest.fixture
def walking_sequence():
    """Provide a walking animation sequence."""
    return list(generate_pose_sequence(duration=2.0, fps=30.0, pose_type="walking"))


@pytest.fixture
def camera_setup():
    """Provide a 3-camera setup."""
    return generate_multi_camera_setup(num_cameras=3)


@pytest.fixture
def multi_view_keypoints(t_pose, camera_setup):
    """
    Provide synthetic 2D keypoints from multiple cameras.
    
    Returns dict mapping camera_id to (points_2d, visibility).
    """
    result = {}
    for cam in camera_setup:
        points_2d, visibility = project_3d_to_2d(
            t_pose,
            cam['intrinsics'],
            cam['extrinsics'],
            add_noise=2.0  # 2 pixel noise
        )
        result[cam['id']] = {
            'points_2d': points_2d,
            'visibility': visibility,
            'confidences': visibility.astype(np.float32) * 0.9,  # High confidence where visible
        }
    return result


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


# ============================================================================
# System Info
# ============================================================================

def get_system_info() -> Dict:
    """Get system information for test reporting."""
    import platform
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
    }
    
    # Check for GPU
    try:
        import onnxruntime as ort
        info['onnxruntime_version'] = ort.__version__
        info['onnxruntime_providers'] = ort.get_available_providers()
    except ImportError:
        info['onnxruntime_version'] = 'not installed'
        info['onnxruntime_providers'] = []
    
    # Check for CUDA
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device'] = torch.cuda.get_device_name(0)
    except ImportError:
        info['torch_version'] = 'not installed'
        info['cuda_available'] = False
    
    return info
