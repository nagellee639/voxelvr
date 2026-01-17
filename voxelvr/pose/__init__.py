"""Pose estimation module."""

# Core modules that don't require OpenCV
from .triangulation import triangulate_points, triangulate_pose
from .filter import OneEuroFilter, PoseFilter
from .rotation import estimate_all_rotations, TrackerRotations, RotationFilter
from .confidence_filter import ConfidenceFilter, create_tpose

# OpenCV-dependent modules (optional import)
try:
    from .detector_2d import PoseDetector2D, Keypoints2D
except ImportError:
    PoseDetector2D = None
    Keypoints2D = None

__all__ = [
    "PoseDetector2D",
    "Keypoints2D", 
    "triangulate_points",
    "triangulate_pose",
    "OneEuroFilter",
    "PoseFilter",
    "estimate_all_rotations",
    "TrackerRotations",
    "RotationFilter",
    "ConfidenceFilter",
    "create_tpose",
]

