"""
Confidence-Based View Filtering with Hysteresis

Implements intelligent filtering of multi-view pose detections based on
confidence scores with a grace period mechanism:

- Immediately responds to loss of confidence (view becomes INACTIVE_GRACE)
- Allows instant recovery within grace period (< 7 frames)
- Requires proof (3 consecutive confident frames) after prolonged absence
- Freezes joints when all views are unconfident
- Initializes with T-pose before first valid tracking
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass

if TYPE_CHECKING:
    from .detector_2d import Keypoints2D


# T-Pose definition in 3D space (COCO 17-keypoint format)
# Coordinates in meters, Y-up coordinate system
# Origin at pelvis center, person facing +Z direction
TPOSE_3D = np.array([
    [0.00,  1.50,  0.00],  # 0: nose
    [-0.05, 1.55,  0.05],  # 1: left_eye
    [0.05,  1.55,  0.05],  # 2: right_eye
    [-0.10, 1.50,  0.10],  # 3: left_ear
    [0.10,  1.50,  0.10],  # 4: right_ear
    [-0.20, 1.30,  0.00],  # 5: left_shoulder
    [0.20,  1.30,  0.00],  # 6: right_shoulder
    [-0.50, 1.30,  0.00],  # 7: left_elbow (arm extended horizontally)
    [0.50,  1.30,  0.00],  # 8: right_elbow
    [-0.80, 1.30,  0.00],  # 9: left_wrist
    [0.80,  1.30,  0.00],  # 10: right_wrist
    [-0.15, 0.85,  0.00],  # 11: left_hip
    [0.15,  0.85,  0.00],  # 12: right_hip
    [-0.15, 0.45,  0.00],  # 13: left_knee
    [0.15,  0.45,  0.00],  # 14: right_knee
    [-0.15, 0.05,  0.00],  # 15: left_ankle
    [0.15,  0.05,  0.00],  # 16: right_ankle
], dtype=np.float32)


def create_tpose() -> np.ndarray:
    """
    Create a standard T-pose.
    
    Returns:
        (17, 3) array of 3D joint positions in meters
    """
    return TPOSE_3D.copy()


class ViewConfidenceState(Enum):
    """State of a view's confidence for triangulation."""
    ACTIVE = "active"                    # Currently using for triangulation
    INACTIVE_GRACE = "inactive_grace"    # Recently unconfident (< grace_period)
    INACTIVE_DEEP = "inactive_deep"      # Unconfident for extended period


@dataclass
class ViewState:
    """
    Tracks confidence state for a single (camera_id, joint_idx) pair.
    
    Implements grace period hysteresis:
    - ACTIVE -> INACTIVE_GRACE on first unconfident frame
    - INACTIVE_GRACE -> ACTIVE immediately on confident frame (fast recovery)
    - INACTIVE_GRACE -> INACTIVE_DEEP after grace_period_frames
    - INACTIVE_DEEP -> ACTIVE after reactivation_frames consecutive confident frames
    """
    state: ViewConfidenceState = ViewConfidenceState.INACTIVE_DEEP
    consecutive_unconfident: int = 0
    consecutive_confident: int = 0
    
    def update(
        self,
        is_confident: bool,
        grace_period_frames: int,
        reactivation_frames: int,
    ) -> None:
        """
        Update state based on current confidence.
        
        Args:
            is_confident: Whether current detection is confident
            grace_period_frames: Frames before requiring reactivation proof
            reactivation_frames: Consecutive confident frames needed after grace period
        """
        if is_confident:
            self.consecutive_confident += 1
            self.consecutive_unconfident = 0
        else:
            self.consecutive_unconfident += 1
            self.consecutive_confident = 0
        
        # State transitions
        if self.state == ViewConfidenceState.ACTIVE:
            if not is_confident:
                # Immediately go to grace period
                self.state = ViewConfidenceState.INACTIVE_GRACE
        
        elif self.state == ViewConfidenceState.INACTIVE_GRACE:
            if is_confident:
                # Fast recovery within grace period!
                self.state = ViewConfidenceState.ACTIVE
            elif self.consecutive_unconfident >= grace_period_frames:
                # Grace period expired, need proof to reactivate
                self.state = ViewConfidenceState.INACTIVE_DEEP
        
        elif self.state == ViewConfidenceState.INACTIVE_DEEP:
            if self.consecutive_confident >= reactivation_frames:
                # Proven confident, reactivate
                self.state = ViewConfidenceState.ACTIVE
    
    def is_active(self) -> bool:
        """Check if view is currently active for triangulation."""
        return self.state == ViewConfidenceState.ACTIVE


class ConfidenceFilter:
    """
    Filters multi-view pose detections based on confidence with hysteresis.
    
    Features:
    - Grace period for fast recovery from brief occlusions
    - Requires proof after prolonged absence to prevent jitter
    - Freezes joints when all views are unconfident
    - Initializes with T-pose before first valid tracking
    """
    
    def __init__(
        self,
        num_joints: int = 17,
        confidence_threshold: float = 0.3,
        grace_period_frames: int = 7,
        reactivation_frames: int = 3,
    ):
        """
        Initialize confidence filter.
        
        Args:
            num_joints: Number of joints (default 17 for COCO)
            confidence_threshold: Minimum confidence to consider valid
            grace_period_frames: Frames before requiring reactivation proof
            reactivation_frames: Consecutive confident frames after grace period
        """
        self.num_joints = num_joints
        self.confidence_threshold = confidence_threshold
        self.grace_period_frames = grace_period_frames
        self.reactivation_frames = reactivation_frames
        
        # Per-joint, per-view state tracking
        # view_states[joint_idx][camera_id] -> ViewState
        self.view_states: Dict[int, Dict[int, ViewState]] = {
            i: {} for i in range(num_joints)
        }
        
        # Last known confident 3D positions for each joint
        # Initialized to T-pose
        self.last_confident_positions = create_tpose()
        
        # Track whether we've received any valid tracking yet
        self.has_tracking_history = False
    
    def update(
        self,
        keypoints_list: List,  # List[Keypoints2D]
    ) -> Tuple[List, Dict[str, np.ndarray]]:
        """
        Filter keypoints based on confidence and return active views.
        
        Args:
            keypoints_list: List of Keypoints2D from all cameras
        
        Returns:
            Tuple of:
            - filtered_keypoints: List of Keypoints2D with only active views
            - diagnostics: Dict with 'active_views_per_joint' (num_joints,)
        """
        filtered_keypoints = []
        active_views_per_joint = np.zeros(self.num_joints, dtype=int)
        
        # Update state for each view and joint
        for kp in keypoints_list:
            camera_id = kp.camera_id
            is_active_for_any_joint = False
            
            for joint_idx in range(min(self.num_joints, len(kp.confidences))):
                # Get or create state for this view
                if camera_id not in self.view_states[joint_idx]:
                    self.view_states[joint_idx][camera_id] = ViewState()
                
                state = self.view_states[joint_idx][camera_id]
                
                # Update state based on confidence
                is_confident = kp.confidences[joint_idx] >= self.confidence_threshold
                state.update(
                    is_confident,
                    self.grace_period_frames,
                    self.reactivation_frames,
                )
                
                # Track active views
                if state.is_active():
                    active_views_per_joint[joint_idx] += 1
                    is_active_for_any_joint = True
            
            # Only include keypoints that are active for at least some joints
            if is_active_for_any_joint:
                filtered_keypoints.append(kp)
        
        diagnostics = {
            'active_views_per_joint': active_views_per_joint,
        }
        
        return filtered_keypoints, diagnostics
    
    def apply_freezing(
        self,
        positions_3d: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply joint freezing for unconfident joints.
        
        Args:
            positions_3d: (num_joints, 3) 3D positions from triangulation
            valid_mask: (num_joints,) boolean mask of valid positions
        
        Returns:
            (num_joints, 3) positions with frozen joints filled in
        """
        result = positions_3d.copy()
        
        for i in range(min(self.num_joints, len(valid_mask))):
            if valid_mask[i]:
                # Update last known position
                self.last_confident_positions[i] = positions_3d[i]
                self.has_tracking_history = True
            else:
                # Freeze: use last known position (or T-pose if no history)
                result[i] = self.last_confident_positions[i]
        
        return result
    
    def get_diagnostics_string(self, active_views: np.ndarray) -> str:
        """
        Get human-readable diagnostics string.
        
        Args:
            active_views: (num_joints,) array of active view counts
        
        Returns:
            Formatted string with joint names and view counts
        """
        from .detector_2d import COCO_KEYPOINT_NAMES
        
        # Group by view count for compact display
        by_count: Dict[int, List[str]] = {}
        for i, count in enumerate(active_views):
            if i < len(COCO_KEYPOINT_NAMES):
                name = COCO_KEYPOINT_NAMES[i]
                if count not in by_count:
                    by_count[count] = []
                by_count[count].append(name)
        
        parts = []
        for count in sorted(by_count.keys(), reverse=True):
            joints = by_count[count]
            if len(joints) <= 3:
                parts.append(f"{count}v: {', '.join(joints)}")
            else:
                parts.append(f"{count}v: {len(joints)} joints")
        
        return " | ".join(parts) if parts else "No active joints"
    
    def reset(self) -> None:
        """Reset all state (useful for testing or restarting tracking)."""
        self.view_states = {i: {} for i in range(self.num_joints)}
        self.last_confident_positions = create_tpose()
        self.has_tracking_history = False
