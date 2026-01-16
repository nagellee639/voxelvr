"""
Multi-View Triangulation

Reconstructs 3D joint positions from multiple 2D views using
camera calibration data and geometric triangulation.

This serves as both a baseline method and a fallback when
VoxelPose is too slow or unavailable.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING

# Optional imports for type hints
if TYPE_CHECKING:
    from .detector_2d import Keypoints2D
    from ..config import CameraIntrinsics, CameraExtrinsics


def triangulate_points(
    points_2d: Dict[int, np.ndarray],  # camera_id -> (N, 2) points
    projection_matrices: Dict[int, np.ndarray],  # camera_id -> (3, 4) matrix
    min_views: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from multiple 2D observations.
    
    Uses Direct Linear Transform (DLT) method with SVD.
    
    Args:
        points_2d: Dictionary mapping camera_id to (N, 2) 2D points
        projection_matrices: Dictionary mapping camera_id to (3, 4) projection matrices
        min_views: Minimum number of views required for triangulation
        
    Returns:
        Tuple of:
        - points_3d: (N, 3) triangulated 3D points
        - reprojection_errors: (N,) reprojection error for each point
    """
    camera_ids = list(points_2d.keys())
    num_views = len(camera_ids)
    
    if num_views < min_views:
        return np.array([]), np.array([])
    
    # Get number of points (assume all cameras see same points)
    num_points = len(points_2d[camera_ids[0]])
    
    points_3d = np.zeros((num_points, 3))
    errors = np.zeros(num_points)
    
    for i in range(num_points):
        # Build the DLT matrix A for this point
        A = []
        
        for cam_id in camera_ids:
            P = projection_matrices[cam_id]
            x, y = points_2d[cam_id][i]
            
            # Each point gives 2 equations
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = np.array(A)
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]  # Last row of V^T is the solution
        
        # Convert from homogeneous coordinates
        X = X[:3] / X[3]
        points_3d[i] = X
        
        # Compute reprojection error
        total_error = 0
        for cam_id in camera_ids:
            P = projection_matrices[cam_id]
            projected = P @ np.append(X, 1)
            projected = projected[:2] / projected[2]
            
            original = points_2d[cam_id][i]
            error = np.linalg.norm(projected - original)
            total_error += error
        
        errors[i] = total_error / num_views
    
    return points_3d, errors


def triangulate_pose(
    keypoints_list: List[Any],  # List[Keypoints2D]
    projection_matrices: Dict[int, np.ndarray],
    confidence_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulate a full pose from multiple camera views.
    
    Args:
        keypoints_list: List of Keypoints2D from different cameras
        projection_matrices: Projection matrix for each camera
        confidence_threshold: Minimum confidence for a keypoint to be used
        
    Returns:
        Tuple of:
        - positions_3d: (17, 3) 3D joint positions
        - confidences: (17,) combined confidence scores
        - valid_mask: (17,) boolean mask for successfully triangulated joints
    """
    num_joints = 17  # COCO format
    positions_3d = np.zeros((num_joints, 3))
    confidences = np.zeros(num_joints)
    valid_mask = np.zeros(num_joints, dtype=bool)
    
    for joint_idx in range(num_joints):
        # Collect 2D observations of this joint from all cameras
        points_2d = {}
        joint_confidences = []
        
        for kp in keypoints_list:
            if kp.confidences[joint_idx] >= confidence_threshold:
                points_2d[kp.camera_id] = kp.positions[joint_idx:joint_idx+1]
                joint_confidences.append(kp.confidences[joint_idx])
        
        if len(points_2d) >= 2:
            # Triangulate this joint
            proj_mats = {cid: projection_matrices[cid] for cid in points_2d.keys()}
            pos_3d, errors = triangulate_points(points_2d, proj_mats)
            
            if len(pos_3d) > 0:
                positions_3d[joint_idx] = pos_3d[0]
                confidences[joint_idx] = np.mean(joint_confidences)
                valid_mask[joint_idx] = True
    
    return positions_3d, confidences, valid_mask


def compute_projection_matrices(
    intrinsics_list: List[Any],  # List[CameraIntrinsics]
    extrinsics_list: List[Any],  # List[CameraExtrinsics]
) -> Dict[int, np.ndarray]:
    """
    Compute projection matrices from calibration data.
    
    Args:
        intrinsics_list: Camera intrinsics for each camera
        extrinsics_list: Camera extrinsics for each camera
        
    Returns:
        Dictionary mapping camera_id to (3, 4) projection matrix
    """
    projection_matrices = {}
    
    intrinsics_map = {intr.camera_id: intr for intr in intrinsics_list}
    
    for ext in extrinsics_list:
        cam_id = ext.camera_id
        if cam_id not in intrinsics_map:
            continue
        
        intr = intrinsics_map[cam_id]
        K = np.array(intr.camera_matrix)
        
        # Extrinsics stores camera-to-world, we need world-to-camera
        T_cam_to_world = np.array(ext.transform_matrix)
        T_world_to_cam = np.linalg.inv(T_cam_to_world)
        
        R = T_world_to_cam[:3, :3]
        t = T_world_to_cam[:3, 3]
        
        # P = K [R | t]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        
        projection_matrices[cam_id] = P
    
    return projection_matrices


class TriangulationPipeline:
    """
    Complete triangulation pipeline for 3D pose estimation.
    
    Combines 2D detection and multi-view triangulation
    for a simpler, faster alternative to VoxelPose.
    """
    
    def __init__(
        self,
        projection_matrices: Dict[int, np.ndarray],
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize the triangulation pipeline.
        
        Args:
            projection_matrices: Pre-computed projection matrices
            confidence_threshold: Minimum keypoint confidence
        """
        self.projection_matrices = projection_matrices
        self.confidence_threshold = confidence_threshold
    
    def process(
        self,
        keypoints_list: List[Any],  # List[Keypoints2D]
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Process 2D keypoints from multiple views to get 3D pose.
        
        Args:
            keypoints_list: 2D keypoints from each camera
            
        Returns:
            Dictionary with:
            - 'positions': (17, 3) joint positions in world space
            - 'confidences': (17,) confidence scores
            - 'valid': (17,) boolean mask
            Or None if triangulation failed
        """
        if len(keypoints_list) < 2:
            return None
        
        # Filter to only cameras we have projection matrices for
        valid_keypoints = [
            kp for kp in keypoints_list 
            if kp.camera_id in self.projection_matrices
        ]
        
        if len(valid_keypoints) < 2:
            return None
        
        positions, confidences, valid_mask = triangulate_pose(
            valid_keypoints,
            self.projection_matrices,
            self.confidence_threshold,
        )
        
        # Check if we got enough valid joints
        if np.sum(valid_mask) < 5:
            return None
        
        return {
            'positions': positions,
            'confidences': confidences,
            'valid': valid_mask,
        }


def ransac_triangulate(
    points_2d: Dict[int, np.ndarray],
    projection_matrices: Dict[int, np.ndarray],
    iterations: int = 100,
    threshold: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    RANSAC-based triangulation for robust outlier rejection.
    
    Useful when some cameras may have incorrect detections.
    
    Args:
        points_2d: 2D observations from each camera
        projection_matrices: Projection matrices
        iterations: Number of RANSAC iterations
        threshold: Inlier threshold in pixels
        
    Returns:
        Tuple of (best_point_3d, best_error, inlier_camera_ids)
    """
    camera_ids = list(points_2d.keys())
    num_cameras = len(camera_ids)
    
    if num_cameras < 2:
        return np.array([0, 0, 0]), np.inf, []
    
    best_point = None
    best_error = np.inf
    best_inliers = []
    
    for _ in range(iterations):
        # Randomly sample 2 cameras
        sample_ids = np.random.choice(camera_ids, size=2, replace=False)
        
        # Triangulate using these 2 cameras
        sample_points = {cid: points_2d[cid] for cid in sample_ids}
        sample_projs = {cid: projection_matrices[cid] for cid in sample_ids}
        
        pts_3d, _ = triangulate_points(sample_points, sample_projs)
        if len(pts_3d) == 0:
            continue
        
        point_3d = pts_3d[0]
        
        # Count inliers and compute error
        inliers = []
        total_error = 0
        
        for cam_id in camera_ids:
            P = projection_matrices[cam_id]
            projected = P @ np.append(point_3d, 1)
            projected = projected[:2] / projected[2]
            
            error = np.linalg.norm(projected - points_2d[cam_id][0])
            
            if error < threshold:
                inliers.append(cam_id)
                total_error += error
        
        if len(inliers) >= 2:
            avg_error = total_error / len(inliers)
            
            # Prefer more inliers, then lower error
            score = -len(inliers) * 1000 + avg_error
            
            if len(inliers) > len(best_inliers) or \
               (len(inliers) == len(best_inliers) and avg_error < best_error):
                best_point = point_3d
                best_error = avg_error
                best_inliers = inliers
    
    if best_point is None:
        best_point = np.array([0, 0, 0])
        best_error = np.inf
    
    return best_point, best_error, best_inliers
