"""
Post-Calibration for VRChat OSC

Handles the "post-calibration" process that:
1. Captures 3 seconds of pose data (after a 3-second countdown)
2. Computes a world transform to center origin at user's feet
3. Aligns axes: Y-up through spine, X-right from shoulders, Z-forward
4. Supports manual Y-axis rotation offset for VRChat alignment
"""

import numpy as np
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
from .coordinate import CoordinateTransform


class PostCalibrationState(Enum):
    """States for post-calibration process."""
    IDLE = "idle"
    COUNTDOWN = "countdown"
    CAPTURING = "capturing"
    COMPLETE = "complete"


@dataclass
class PostCalibrationResult:
    """Result of post-calibration capture."""
    origin: np.ndarray  # (3,) point between feet
    up_axis: np.ndarray  # (3,) up through spine
    right_axis: np.ndarray  # (3,) user's right direction
    forward_axis: np.ndarray  # (3,) facing direction
    transform: CoordinateTransform  # Base transform (no yaw offset)
    
    def get_transform_with_yaw(self, yaw_degrees: float) -> CoordinateTransform:
        """
        Get transform with additional Y-axis rotation.
        
        Args:
            yaw_degrees: Rotation around Y axis in degrees
            
        Returns:
            New CoordinateTransform with yaw applied
        """
        if abs(yaw_degrees) < 0.01:
            return self.transform
        
        # Compute yaw rotation matrix (around Y axis)
        yaw_rad = np.radians(yaw_degrees)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ], dtype=np.float64)
        
        # Combine: first apply base rotation, then yaw
        combined_rotation = R_yaw @ self.transform.rotation
        
        # Rotate the offset as well
        rotated_offset = R_yaw @ self.transform.offset
        
        return CoordinateTransform(
            rotation=combined_rotation,
            scale=self.transform.scale,
            offset=rotated_offset,
        )


class PostCalibrator:
    """
    Handles post-calibration capture and transform computation.
    
    Usage:
        calibrator = PostCalibrator()
        calibrator.start()  # Begin countdown
        
        # In tracking loop:
        calibrator.update(pose_3d, timestamp)
        
        if calibrator.state == PostCalibrationState.COMPLETE:
            transform = calibrator.get_transform(yaw_offset)
    """
    
    def __init__(
        self,
        countdown_duration: float = 3.0,
        capture_duration: float = 3.0,
    ):
        """
        Initialize post-calibrator.
        
        Args:
            countdown_duration: Seconds to countdown before capture
            capture_duration: Seconds to capture poses
        """
        self.countdown_duration = countdown_duration
        self.capture_duration = capture_duration
        
        self._state = PostCalibrationState.IDLE
        self._start_time: Optional[float] = None
        self._captured_poses: List[np.ndarray] = []
        self._result: Optional[PostCalibrationResult] = None
        self._yaw_offset: float = 0.0
    
    @property
    def state(self) -> PostCalibrationState:
        """Current calibration state."""
        return self._state
    
    @property
    def result(self) -> Optional[PostCalibrationResult]:
        """Calibration result, if complete."""
        return self._result
    
    @property
    def yaw_offset(self) -> float:
        """Current Y-axis rotation offset in degrees."""
        return self._yaw_offset
    
    @yaw_offset.setter
    def yaw_offset(self, value: float):
        """Set Y-axis rotation offset."""
        self._yaw_offset = value % 360
        if self._yaw_offset > 180:
            self._yaw_offset -= 360
    
    @property
    def countdown_remaining(self) -> float:
        """Seconds remaining in countdown (0 if not in countdown)."""
        if self._state != PostCalibrationState.COUNTDOWN or self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return max(0.0, self.countdown_duration - elapsed)
    
    @property
    def capture_progress(self) -> float:
        """Capture progress 0.0 to 1.0 (0 if not capturing)."""
        if self._state != PostCalibrationState.CAPTURING or self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time - self.countdown_duration
        return min(1.0, max(0.0, elapsed / self.capture_duration))
    
    def start(self) -> None:
        """Start the post-calibration process (begins countdown)."""
        self._state = PostCalibrationState.COUNTDOWN
        self._start_time = time.time()
        self._captured_poses = []
        self._result = None
        print(f"Post-calibration started: {self.countdown_duration}s countdown...")
    
    def reset(self) -> None:
        """Reset to idle state, keeping yaw offset."""
        self._state = PostCalibrationState.IDLE
        self._start_time = None
        self._captured_poses = []
        self._result = None
    
    def adjust_yaw(self, delta_degrees: float) -> None:
        """Adjust yaw offset by delta."""
        self.yaw_offset = self._yaw_offset + delta_degrees
        print(f"Yaw offset: {self._yaw_offset:.1f}°")
    
    def reset_yaw(self) -> None:
        """Reset yaw offset to 0."""
        self._yaw_offset = 0.0
        print("Yaw offset reset to 0°")
    
    def update(self, pose_3d: np.ndarray, valid_mask: np.ndarray) -> None:
        """
        Update with current pose data.
        
        Args:
            pose_3d: (17, 3) joint positions
            valid_mask: (17,) boolean validity mask
        """
        if self._start_time is None:
            return
        
        current_time = time.time()
        elapsed = current_time - self._start_time
        
        # Check state transitions
        if self._state == PostCalibrationState.COUNTDOWN:
            if elapsed >= self.countdown_duration:
                self._state = PostCalibrationState.CAPTURING
                print("Post-calibration: CAPTURING (stand still in T-pose)...")
        
        elif self._state == PostCalibrationState.CAPTURING:
            # Capture pose if enough joints are valid
            # Key joints: ankles (15, 16), hips (11, 12), shoulders (5, 6)
            key_joints = [5, 6, 11, 12, 15, 16]
            if all(valid_mask[j] for j in key_joints):
                self._captured_poses.append(pose_3d.copy())
            
            # Check if capture complete
            if elapsed >= self.countdown_duration + self.capture_duration:
                self._compute_transform()
    
    def _compute_transform(self) -> None:
        """Compute the world transform from captured poses."""
        if not self._captured_poses:
            print("Post-calibration FAILED: No valid poses captured")
            self._state = PostCalibrationState.IDLE
            return
        
        # Average all captured poses
        avg_pose = np.mean(self._captured_poses, axis=0)
        
        # Extract key points (COCO indices)
        left_ankle = avg_pose[15]
        right_ankle = avg_pose[16]
        left_hip = avg_pose[11]
        right_hip = avg_pose[12]
        left_shoulder = avg_pose[5]
        right_shoulder = avg_pose[6]
        
        # Origin: midpoint between ankles
        origin = (left_ankle + right_ankle) / 2
        
        # Up axis: from hip midpoint to shoulder midpoint
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        up_axis = shoulder_center - hip_center
        up_axis = up_axis / np.linalg.norm(up_axis)
        
        # Right axis: from left shoulder/hip to right (user's right)
        # Use average of shoulder and hip lines for stability
        shoulder_right = right_shoulder - left_shoulder
        hip_right = right_hip - left_hip
        right_axis = (shoulder_right + hip_right) / 2
        right_axis = right_axis / np.linalg.norm(right_axis)
        
        # Forward axis: cross product of up × right (right-hand rule gives forward)
        forward_axis = np.cross(up_axis, right_axis)
        forward_axis = forward_axis / np.linalg.norm(forward_axis)
        
        # Re-orthogonalize right axis
        right_axis = np.cross(forward_axis, up_axis)
        right_axis = right_axis / np.linalg.norm(right_axis)
        
        # Build rotation matrix: columns are new X, Y, Z axes
        # We want to transform FROM world TO VRChat (Y-up) space
        # Input coordinates use our computed axes
        # Output should be X-right, Y-up, Z-forward
        rotation = np.array([
            right_axis,    # X axis
            up_axis,       # Y axis  
            forward_axis,  # Z axis
        ], dtype=np.float64)
        
        # Offset: translate so origin is at feet
        offset = -rotation @ origin
        
        transform = CoordinateTransform(
            rotation=rotation,
            scale=1.0,
            offset=offset,
        )
        
        self._result = PostCalibrationResult(
            origin=origin,
            up_axis=up_axis,
            right_axis=right_axis,
            forward_axis=forward_axis,
            transform=transform,
        )
        
        self._state = PostCalibrationState.COMPLETE
        print(f"Post-calibration COMPLETE: {len(self._captured_poses)} poses averaged")
        print(f"  Origin: {origin}")
        print(f"  Up: {up_axis}")
        print(f"  Forward: {forward_axis}")
    
    def get_transform(self) -> Optional[CoordinateTransform]:
        """
        Get the final transform including yaw offset.
        
        Returns:
            CoordinateTransform if calibration complete, else None
        """
        if self._result is None:
            return None
        return self._result.get_transform_with_yaw(self._yaw_offset)
    
    def get_status_text(self) -> str:
        """Get human-readable status for UI display."""
        if self._state == PostCalibrationState.IDLE:
            return "Click to calibrate"
        elif self._state == PostCalibrationState.COUNTDOWN:
            remaining = self.countdown_remaining
            return f"Get ready... {remaining:.0f}s"
        elif self._state == PostCalibrationState.CAPTURING:
            progress = self.capture_progress * 100
            return f"Capturing: {progress:.0f}%"
        elif self._state == PostCalibrationState.COMPLETE:
            return f"✓ Calibrated (Yaw: {self._yaw_offset:.0f}°)"
        return ""
