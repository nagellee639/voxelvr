"""
VoxelVR Configuration Management

Handles loading/saving of calibration data, user preferences, and runtime settings.
Uses Pydantic for validation and YAML for human-readable config files.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml
import json


# Default paths
CONFIG_DIR = Path.home() / ".voxelvr"
CALIBRATION_DIR = CONFIG_DIR / "calibration"
RECORDINGS_DIR = CONFIG_DIR / "recordings"


class CameraConfig(BaseModel):
    """Configuration for a single camera."""
    id: int  # OpenCV camera index or device ID
    name: str = ""  # User-friendly name (e.g., "Front Left")
    resolution: tuple[int, int] = (1280, 720)
    fps: int = 30
    
    # Intrinsic calibration (set after calibration)
    intrinsic_path: Optional[Path] = None
    
    
class CalibrationConfig(BaseModel):
    """Calibration settings."""
    # ChArUco board parameters
    charuco_squares_x: int = 7
    charuco_squares_y: int = 5
    charuco_square_length: float = 0.04  # meters
    charuco_marker_length: float = 0.03  # meters
    charuco_dict: str = "DICT_6X6_250"
    
    # Capture settings for calibration
    intrinsic_frames_required: int = 20
    extrinsic_frames_required: int = 30


class TrackingConfig(BaseModel):
    """Pose estimation and tracking settings."""
    # Model selection
    pose_2d_model: str = "rtmpose-m"  # Options: rtmpose-s, rtmpose-m, rtmpose-l
    use_voxelpose: bool = True  # If False, fall back to triangulation
    
    # Voxel grid settings
    voxel_size: tuple[float, float, float] = (2.0, 2.0, 2.0)  # meters
    voxel_resolution: tuple[int, int, int] = (64, 64, 64)
    
    # Temporal filtering (One-Euro filter parameters)
    filter_enabled: bool = True
    filter_min_cutoff: float = 1.0  # Lower = more smoothing
    filter_beta: float = 0.5  # Higher = less lag during fast motion
    filter_d_cutoff: float = 1.0
    
    # Confidence-based view filtering
    confidence_threshold: float = 0.3  # Minimum confidence for valid detection
    confidence_grace_period_frames: int = 7  # Frames before requiring reactivation
    confidence_reactivation_frames: int = 3  # Consecutive frames to reactivate
    freeze_unconfident_joints: bool = True  # Freeze joints when all views unconfident
    
    # Performance
    target_fps: int = 30
    max_persons: int = 1  # For now, single person tracking


class OSCConfig(BaseModel):
    """VRChat OSC output settings."""
    enabled: bool = True
    ip: str = "127.0.0.1"
    port: int = 9000
    
    # Which trackers to send (VRChat supports up to 8 + head)
    send_hip: bool = True
    send_chest: bool = True
    send_feet: bool = True
    send_knees: bool = True
    send_elbows: bool = True
    send_head: bool = True  # For alignment with Quest


class DemoConfig(BaseModel):
    """Demo visualizer settings."""
    show_3d_skeleton: bool = True
    show_camera_feeds: bool = True
    show_metrics: bool = True
    skeleton_color: str = "#00ff88"
    joint_size: float = 0.03  # meters
    bone_width: float = 0.01  # meters


class VoxelVRConfig(BaseSettings):
    """Main configuration container."""
    cameras: List[CameraConfig] = Field(default_factory=list)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    osc: OSCConfig = Field(default_factory=OSCConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)
    
    # Paths
    config_dir: Path = CONFIG_DIR
    calibration_dir: Path = CALIBRATION_DIR
    recordings_dir: Path = RECORDINGS_DIR
    
    class Config:
        env_prefix = "VOXELVR_"
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "VoxelVRConfig":
        """Load configuration from YAML file."""
        if path is None:
            path = CONFIG_DIR / "config.yaml"
        
        if path.exists():
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        if path is None:
            path = CONFIG_DIR / "config.yaml"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(mode='json'), f, default_flow_style=False)
    
    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)


class CameraIntrinsics(BaseModel):
    """Intrinsic calibration data for a camera."""
    camera_id: int
    camera_name: str
    resolution: tuple[int, int]
    
    # Camera matrix (3x3)
    camera_matrix: List[List[float]]
    
    # Distortion coefficients (typically 5 or 8 values)
    distortion_coeffs: List[float]
    
    # Calibration quality metrics
    reprojection_error: float
    calibration_date: str
    
    def save(self, path: Path) -> None:
        """Save intrinsics to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "CameraIntrinsics":
        """Load intrinsics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class CameraExtrinsics(BaseModel):
    """Extrinsic calibration data (camera pose in world space)."""
    camera_id: int
    camera_name: str
    
    # 4x4 transformation matrix (world to camera)
    transform_matrix: List[List[float]]
    
    # Convenience: rotation and translation decomposed
    rotation_matrix: List[List[float]]  # 3x3
    translation_vector: List[float]  # 3x1
    
    # Calibration quality
    reprojection_error: float
    calibration_date: str
    
    def save(self, path: Path) -> None:
        """Save extrinsics to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "CameraExtrinsics":
        """Load extrinsics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MultiCameraCalibration(BaseModel):
    """Complete calibration data for the multi-camera system."""
    cameras: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    
    # World origin definition
    origin_description: str = "Center of room, floor level"
    
    def add_camera(
        self, 
        intrinsics: CameraIntrinsics, 
        extrinsics: CameraExtrinsics
    ) -> None:
        """Add a calibrated camera to the system."""
        self.cameras[intrinsics.camera_id] = {
            "intrinsics": intrinsics.model_dump(),
            "extrinsics": extrinsics.model_dump(),
        }
    
    def save(self, path: Path) -> None:
        """Save complete calibration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "MultiCameraCalibration":
        """Load calibration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Skeleton definition (COCO format keypoints)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# VRChat tracker mapping
VRCHAT_TRACKER_MAP = {
    "hip": 1,          # Midpoint of left_hip and right_hip
    "chest": 2,        # Midpoint of left_shoulder and right_shoulder
    "left_foot": 3,    # left_ankle
    "right_foot": 4,   # right_ankle
    "left_knee": 5,    # left_knee
    "right_knee": 6,   # right_knee
    "left_elbow": 7,   # left_elbow
    "right_elbow": 8,  # right_elbow
}
