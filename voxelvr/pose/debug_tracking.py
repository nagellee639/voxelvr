"""
Debug Utility for Tracking Pipeline

Provides functions for logging, validation, and error analysis
of the tracking pipeline components.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoxelVR.Debug")

@dataclass
class TriangulationDebugInfo:
    """Debug info for a single triangulation step."""
    frame_id: int
    num_cameras: int
    num_keypoints_detected: Dict[int, int]  # camera_id -> count
    triangulation_success: bool
    num_valid_joints: int
    avg_reprojection_error: float
    outlier_cameras: List[int]

class TrackingDebugger:
    """
    Central debugger for the tracking system.
    """
    
    def __init__(self):
        self.frame_history: List[TriangulationDebugInfo] = []
        self.max_history = 1000
    
    def log_triangulation(
        self, 
        frame_id: int, 
        keypoints_2d: Dict[int, Any], 
        pose_3d: Optional[Dict[str, np.ndarray]],
        errors: Optional[np.ndarray] = None
    ):
        """
        Log details of a triangulation attempt.
        """
        num_cams = len(keypoints_2d)
        kp_counts = {cid: len(kp.positions) if hasattr(kp, 'positions') else 0 
                     for cid, kp in keypoints_2d.items()}
        
        success = pose_3d is not None
        valid_joints = np.sum(pose_3d['valid']) if success and pose_3d else 0
        
        avg_error = 0.0
        if errors is not None and len(errors) > 0:
            avg_error = np.mean(errors)
            
        info = TriangulationDebugInfo(
            frame_id=frame_id,
            num_cameras=num_cams,
            num_keypoints_detected=kp_counts,
            triangulation_success=success,
            num_valid_joints=valid_joints,
            avg_reprojection_error=avg_error,
            outlier_cameras=[] # TODO: Implement outlier detection
        )
        
        self.frame_history.append(info)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
            
        # Log potential issues
        if success and valid_joints < 5:
            logger.warning(f"[Frame {frame_id}] Low valid joint count: {valid_joints}/17. Safe minimum is 5.")
        
        if success and avg_error > 0.1: # 10cm error? or pixels? usually pixels in 2d, meters in 3d. 
            # In triangulation.py errors are reprojection errors in pixels.
            logger.warning(f"[Frame {frame_id}] High reprojection error: {avg_error:.2f} px")

    def analyze_calibration(self, camera_transforms: Dict[int, np.ndarray]):
        """
        Analyze camera calibration for sanity.
        """
        logger.info("Analyzing camera calibration...")
        
        positions = {cid: T[:3, 3] for cid, T in camera_transforms.items()}
        
        # Check for cameras placed extremely far away
        for cid, pos in positions.items():
            dist = np.linalg.norm(pos)
            if dist > 10.0:
                logger.warning(f"Camera {cid} is {dist:.1f}m from origin. This seems suspicious.")
            logger.info(f"Camera {cid} position: {pos}")

        # Check for cameras that are too close (overlapping?)
        cids = list(positions.keys())
        for i in range(len(cids)):
            for j in range(i+1, len(cids)):
                cid1, cid2 = cids[i], cids[j]
                dist = np.linalg.norm(positions[cid1] - positions[cid2])
                if dist < 0.1: # 10cm
                    logger.warning(f"Camera {cid1} and {cid2} are ver close ({dist:.3f}m). Duplicate or error?")

def print_matrix(name: str, mat: np.ndarray):
    """Pretty print a matrix."""
    print(f"{name}:")
    print(np.array2string(mat, precision=3, suppress_small=True))
