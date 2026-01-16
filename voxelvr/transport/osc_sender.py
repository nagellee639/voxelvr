"""
VRChat OSC Sender

Transmits tracker data to VRChat using the OSC protocol.
VRChat supports up to 8 body trackers plus head position/rotation.

OSC Addresses:
- /tracking/trackers/1/position (hip)
- /tracking/trackers/1/rotation
- /tracking/trackers/2/position (chest)
- ... up to /tracking/trackers/8
- /tracking/trackers/head/position
- /tracking/trackers/head/rotation

All positions are Vector3 (x, y, z) in meters.
All rotations are Euler angles in degrees (applied Z -> X -> Y).
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
import time


@dataclass
class VRChatTracker:
    """
    Tracker data ready for VRChat transmission.
    
    Attributes:
        position: (x, y, z) in meters, Unity coordinate system
        rotation: (pitch, yaw, roll) in degrees, Euler angles
        confidence: Tracking confidence [0-1]
    """
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    confidence: float = 1.0


# VRChat tracker IDs and their body part mappings
VRCHAT_TRACKER_IDS = {
    "hip": 1,
    "chest": 2,
    "left_foot": 3,
    "right_foot": 4,
    "left_knee": 5,
    "right_knee": 6,
    "left_elbow": 7,
    "right_elbow": 8,
}

# Mapping from COCO keypoint indices to VRChat trackers
COCO_TO_TRACKER = {
    # Hip is midpoint of left_hip (11) and right_hip (12)
    "hip": (11, 12),
    # Chest is midpoint of left_shoulder (5) and right_shoulder (6)
    "chest": (5, 6),
    # Feet
    "left_foot": 15,  # left_ankle
    "right_foot": 16,  # right_ankle
    # Knees
    "left_knee": 13,
    "right_knee": 14,
    # Elbows
    "left_elbow": 7,
    "right_elbow": 8,
}


class OSCSender:
    """
    Sends tracking data to VRChat via OSC.
    
    Handles connection, message formatting, and transmission
    at the appropriate rate for smooth tracking.
    """
    
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 9000,
        send_rate: float = 60.0,  # Hz
    ):
        """
        Initialize OSC sender.
        
        Args:
            ip: VRChat OSC receiver IP address
            port: VRChat OSC receiver port (default 9000)
            send_rate: Target send rate in Hz
        """
        self.ip = ip
        self.port = port
        self.send_rate = send_rate
        self.min_interval = 1.0 / send_rate if send_rate > 0 else 0
        
        self.client: Optional[udp_client.SimpleUDPClient] = None
        self.last_send_time = 0.0
        
        # Track what trackers are enabled
        self.enabled_trackers = set(VRCHAT_TRACKER_IDS.keys())
        self.send_head = True
        
    def connect(self) -> bool:
        """
        Initialize the OSC client connection.
        
        Returns:
            True if connection successful
        """
        try:
            self.client = udp_client.SimpleUDPClient(self.ip, self.port)
            print(f"OSC client ready: {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to create OSC client: {e}")
            return False
    
    def disconnect(self) -> None:
        """Clean up OSC client."""
        self.client = None
    
    def send_tracker(
        self,
        tracker_name: str,
        tracker: VRChatTracker,
    ) -> bool:
        """
        Send a single tracker's data.
        
        Args:
            tracker_name: Name of tracker (e.g., "hip", "left_foot")
            tracker: Tracker data to send
            
        Returns:
            True if sent successfully
        """
        if self.client is None:
            return False
        
        if tracker_name == "head":
            return self._send_head(tracker)
        
        if tracker_name not in VRCHAT_TRACKER_IDS:
            return False
        
        tracker_id = VRCHAT_TRACKER_IDS[tracker_name]
        
        try:
            # Send position
            pos_addr = f"/tracking/trackers/{tracker_id}/position"
            self.client.send_message(pos_addr, list(tracker.position))
            
            # Send rotation
            rot_addr = f"/tracking/trackers/{tracker_id}/rotation"
            self.client.send_message(rot_addr, list(tracker.rotation))
            
            return True
        except Exception as e:
            print(f"Failed to send tracker {tracker_name}: {e}")
            return False
    
    def _send_head(self, tracker: VRChatTracker) -> bool:
        """Send head tracking data (used for alignment)."""
        if self.client is None:
            return False
        
        try:
            self.client.send_message(
                "/tracking/trackers/head/position",
                list(tracker.position)
            )
            self.client.send_message(
                "/tracking/trackers/head/rotation",
                list(tracker.rotation)
            )
            return True
        except Exception as e:
            print(f"Failed to send head tracker: {e}")
            return False
    
    def send_all_trackers(
        self,
        trackers: Dict[str, VRChatTracker],
        respect_rate_limit: bool = True,
    ) -> bool:
        """
        Send all tracker data in a single batch.
        
        Args:
            trackers: Dictionary mapping tracker names to data
            respect_rate_limit: If True, skip send if called too frequently
            
        Returns:
            True if sent successfully
        """
        current_time = time.time()
        
        if respect_rate_limit:
            if current_time - self.last_send_time < self.min_interval:
                return True  # Skipped, not an error
        
        if self.client is None:
            if not self.connect():
                return False
        
        success = True
        for name, tracker in trackers.items():
            if name in self.enabled_trackers or name == "head":
                if not self.send_tracker(name, tracker):
                    success = False
        
        self.last_send_time = current_time
        return success
    
    def enable_tracker(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a specific tracker."""
        if enabled:
            self.enabled_trackers.add(name)
        else:
            self.enabled_trackers.discard(name)


def pose_to_trackers(
    positions_3d: np.ndarray,
    confidences: np.ndarray,
    valid_mask: np.ndarray,
    confidence_threshold: float = 0.3,
    rotations: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Dict[str, VRChatTracker]:
    """
    Convert 3D pose to VRChat tracker format.
    
    Args:
        positions_3d: (17, 3) joint positions in world space
        confidences: (17,) confidence scores
        valid_mask: (17,) boolean mask for valid joints
        confidence_threshold: Minimum confidence to include tracker
        rotations: Optional pre-computed rotations dict from estimate_all_rotations
        
    Returns:
        Dictionary of VRChatTracker objects
    """
    trackers = {}
    
    # Use provided rotations or default to zeros
    if rotations is None:
        rotations = {}
    
    for tracker_name, indices in COCO_TO_TRACKER.items():
        # Get rotation for this tracker (default to zero)
        rotation = rotations.get(tracker_name, (0.0, 0.0, 0.0))
        
        if isinstance(indices, tuple):
            # Midpoint of two joints (hip, chest)
            idx1, idx2 = indices
            
            if valid_mask[idx1] and valid_mask[idx2]:
                if confidences[idx1] >= confidence_threshold and \
                   confidences[idx2] >= confidence_threshold:
                    position = (positions_3d[idx1] + positions_3d[idx2]) / 2
                    confidence = (confidences[idx1] + confidences[idx2]) / 2
                    
                    trackers[tracker_name] = VRChatTracker(
                        position=tuple(position),
                        rotation=rotation,
                        confidence=confidence,
                    )
        else:
            # Single joint
            idx = indices
            
            if valid_mask[idx] and confidences[idx] >= confidence_threshold:
                trackers[tracker_name] = VRChatTracker(
                    position=tuple(positions_3d[idx]),
                    rotation=rotation,
                    confidence=confidences[idx],
                )
    
    return trackers


def pose_to_trackers_with_rotations(
    positions_3d: np.ndarray,
    confidences: np.ndarray,
    valid_mask: np.ndarray,
    confidence_threshold: float = 0.3,
) -> Dict[str, VRChatTracker]:
    """
    Convert 3D pose to VRChat trackers with automatic rotation estimation.
    
    This is a convenience function that combines position and rotation
    estimation in one call.
    
    Args:
        positions_3d: (17, 3) joint positions in VRChat coordinate space
        confidences: (17,) confidence scores
        valid_mask: (17,) boolean mask for valid joints
        confidence_threshold: Minimum confidence to include tracker
        
    Returns:
        Dictionary of VRChatTracker objects with positions and rotations
    """
    # Import rotation estimation
    from ..pose.rotation import estimate_all_rotations
    
    # Estimate rotations from positions
    rotations = estimate_all_rotations(positions_3d, valid_mask)
    rotation_dict = rotations.to_dict()
    
    # Create trackers with rotations
    return pose_to_trackers(
        positions_3d,
        confidences,
        valid_mask,
        confidence_threshold,
        rotation_dict,
    )

