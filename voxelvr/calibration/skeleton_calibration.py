"""
Skeleton-Based Camera Calibration

Estimates camera positions from triangulated skeleton poses when ChArUco
is unavailable. Uses bone length constraints for scale recovery and
bundle adjustment for refinement.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from scipy.optimize import least_squares


# Standard adult bone lengths in meters (from anthropometric data)
BONE_LENGTHS = {
    'shoulder': 0.35,      # Left shoulder to right shoulder
    'hip_width': 0.28,     # Left hip to right hip
    'upper_arm': 0.30,     # Shoulder to elbow
    'forearm': 0.26,       # Elbow to wrist
    'thigh': 0.43,         # Hip to knee
    'shin': 0.42,          # Knee to ankle
    'torso': 0.50,         # Mid-shoulder to mid-hip
}

# Bone length tolerances (min, max) as fraction of reference
BONE_TOLERANCE = (0.7, 1.4)

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

# Bone definitions: (joint1, joint2, expected_length_key)
BONES = [
    (LEFT_SHOULDER, RIGHT_SHOULDER, 'shoulder'),
    (LEFT_HIP, RIGHT_HIP, 'hip_width'),
    (LEFT_SHOULDER, LEFT_ELBOW, 'upper_arm'),
    (RIGHT_SHOULDER, RIGHT_ELBOW, 'upper_arm'),
    (LEFT_ELBOW, LEFT_WRIST, 'forearm'),
    (RIGHT_ELBOW, RIGHT_WRIST, 'forearm'),
    (LEFT_HIP, LEFT_KNEE, 'thigh'),
    (RIGHT_HIP, RIGHT_KNEE, 'thigh'),
    (LEFT_KNEE, LEFT_ANKLE, 'shin'),
    (RIGHT_KNEE, RIGHT_ANKLE, 'shin'),
]


@dataclass
class SimpleCameraIntrinsics:
    """Simplified camera intrinsics for skeleton calibration."""
    camera_id: int
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    
    @classmethod
    def from_pydantic(cls, intr: Any) -> 'SimpleCameraIntrinsics':
        """Create from pydantic CameraIntrinsics."""
        cam_matrix = np.array(intr.camera_matrix)
        return cls(
            camera_id=intr.camera_id,
            fx=cam_matrix[0, 0],
            fy=cam_matrix[1, 1],
            cx=cam_matrix[0, 2],
            cy=cam_matrix[1, 2],
            width=intr.resolution[0],
            height=intr.resolution[1],
            distortion_coeffs=np.array(intr.distortion_coeffs),
        )


@dataclass
class SimpleCameraExtrinsics:
    """Simplified camera extrinsics for skeleton calibration."""
    camera_id: int
    rotation_matrix: np.ndarray  # (3, 3)
    translation: np.ndarray      # (3,)
    
    @classmethod
    def from_pydantic(cls, ext: Any) -> 'SimpleCameraExtrinsics':
        """Create from pydantic CameraExtrinsics."""
        return cls(
            camera_id=ext.camera_id,
            rotation_matrix=np.array(ext.rotation_matrix),
            translation=np.array(ext.translation_vector),
        )
    
    def to_pydantic(self) -> Any:
        """Convert to pydantic CameraExtrinsics."""
        from ..config import CameraExtrinsics
        
        # Build 4x4 transform matrix
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.translation
        
        return CameraExtrinsics(
            camera_id=self.camera_id,
            camera_name=f"Camera {self.camera_id}",
            transform_matrix=T.tolist(),
            rotation_matrix=self.rotation_matrix.tolist(),
            translation_vector=self.translation.tolist(),
            reprojection_error=0.0,
            calibration_date=datetime.now().isoformat(),
        )


# Type aliases for compatibility
CameraIntrinsics = SimpleCameraIntrinsics
CameraExtrinsics = SimpleCameraExtrinsics


@dataclass
class SkeletonObservation:
    """A single skeleton observation across multiple cameras."""
    keypoints_2d: Dict[int, np.ndarray]  # camera_id -> (17, 2) keypoints
    confidences: Dict[int, np.ndarray]   # camera_id -> (17,) confidences
    timestamp: float = 0.0


@dataclass
class SkeletonCalibrationResult:
    """Result of skeleton-based calibration."""
    cameras: Dict[int, SimpleCameraExtrinsics]
    scale_factor: float
    reprojection_error: float
    bone_length_error: float
    num_observations: int
    is_skeleton_only: bool  # True if no ChArUco was used


def triangulate_point(
    points_2d: Dict[int, np.ndarray],
    projection_matrices: Dict[int, np.ndarray],
) -> Optional[np.ndarray]:
    """
    Triangulate a single 3D point from multiple 2D observations.
    Uses SVD-based DLT method.
    """
    if len(points_2d) < 2:
        return None
    
    # Build the system of equations
    A = []
    for cam_id, pt in points_2d.items():
        if cam_id not in projection_matrices:
            continue
        P = projection_matrices[cam_id]
        x, y = pt[0], pt[1]
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    
    if len(A) < 4:
        return None
    
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    if abs(X[3]) < 1e-10:
        return None
    
    return X[:3] / X[3]


def compute_projection_matrix(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    """Compute 3x4 projection matrix from calibration."""
    K = np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1]
    ])
    R = extrinsics.rotation_matrix
    t = extrinsics.translation.reshape(3, 1)
    return K @ np.hstack([R, t])


def estimate_initial_cameras(
    observations: List[SkeletonObservation],
    intrinsics: Dict[int, CameraIntrinsics],
    min_confidence: float = 0.3,
) -> Dict[int, CameraExtrinsics]:
    """
    Estimate initial camera positions from skeleton observations.
    
    Uses a simple approach:
    1. Assume cameras face the center of the tracking volume
    2. Estimate distance based on field of view and skeleton size
    3. Arrange cameras in a rough circle based on which joints they see best
    """
    camera_ids = list(intrinsics.keys())
    if len(camera_ids) == 0:
        return {}
    
    # Find the average position of skeleton center (mid-point of shoulders)
    # for each camera to estimate relative positioning
    camera_angles = {}
    camera_distances = {}
    
    for cam_id in camera_ids:
        left_shoulder_xs = []
        right_shoulder_xs = []
        skeleton_heights = []
        
        for obs in observations:
            if cam_id not in obs.keypoints_2d:
                continue
            kpts = obs.keypoints_2d[cam_id]
            conf = obs.confidences.get(cam_id, np.ones(17))
            
            if conf[LEFT_SHOULDER] > min_confidence and conf[RIGHT_SHOULDER] > min_confidence:
                left_shoulder_xs.append(kpts[LEFT_SHOULDER, 0])
                right_shoulder_xs.append(kpts[RIGHT_SHOULDER, 0])
                
            # Estimate skeleton height in pixels (nose to ankles)
            if conf[NOSE] > min_confidence:
                y_top = kpts[NOSE, 1]
                y_bottom = None
                if conf[LEFT_ANKLE] > min_confidence:
                    y_bottom = kpts[LEFT_ANKLE, 1]
                elif conf[RIGHT_ANKLE] > min_confidence:
                    y_bottom = kpts[RIGHT_ANKLE, 1]
                if y_bottom is not None:
                    skeleton_heights.append(abs(y_bottom - y_top))
        
        if len(left_shoulder_xs) > 0:
            # Which shoulder appears on the left side indicates viewing angle
            avg_left = np.mean(left_shoulder_xs)
            avg_right = np.mean(right_shoulder_xs)
            cx = intrinsics[cam_id].cx
            
            # If left shoulder is to the left of right shoulder, we're viewing from front
            # The offset from center indicates angle
            mid_x = (avg_left + avg_right) / 2
            offset = (mid_x - cx) / cx  # Normalized offset
            camera_angles[cam_id] = offset * np.pi / 2  # Map to angle
        else:
            camera_angles[cam_id] = 0
        
        if len(skeleton_heights) > 0:
            avg_height_px = np.mean(skeleton_heights)
            # Assume human is ~1.7m tall
            # Distance = (focal_length * real_height) / pixel_height
            fy = intrinsics[cam_id].fy
            camera_distances[cam_id] = (fy * 1.7) / max(avg_height_px, 100)
        else:
            camera_distances[cam_id] = 2.5  # Default 2.5m
    
    # Create camera extrinsics
    cameras = {}
    n_cams = len(camera_ids)
    
    for i, cam_id in enumerate(camera_ids):
        # Distribute cameras around the origin
        base_angle = (2 * np.pi * i) / n_cams + camera_angles.get(cam_id, 0)
        distance = camera_distances.get(cam_id, 2.5)
        
        # Camera position (looking at origin)
        x = distance * np.sin(base_angle)
        z = distance * np.cos(base_angle)
        y = 1.2  # Camera height ~1.2m
        
        # Rotation matrix: camera looks at origin
        forward = -np.array([x, y - 0.9, z])  # Look at waist height
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Rotation matrix (world to camera)
        R = np.stack([right, -up, forward], axis=0)
        t = -R @ np.array([x, y, z])
        
        cameras[cam_id] = CameraExtrinsics(
            camera_id=cam_id,
            rotation_matrix=R,
            translation=t,
        )
    
    return cameras


def compute_bone_length_error(
    positions_3d: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute error between observed bone lengths and expected lengths.
    Returns total error and per-bone errors.
    """
    errors = {}
    total_error = 0.0
    count = 0
    
    for j1, j2, bone_name in BONES:
        if valid_mask[j1] and valid_mask[j2]:
            length = np.linalg.norm(positions_3d[j1] - positions_3d[j2])
            expected = BONE_LENGTHS[bone_name]
            error = abs(length - expected) / expected
            errors[bone_name] = error
            total_error += error
            count += 1
    
    avg_error = total_error / max(count, 1)
    return avg_error, errors


def estimate_scale_from_bones(
    positions_3d: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """
    Estimate scale factor from observed bone lengths.
    Returns factor to multiply positions by.
    """
    ratios = []
    
    for j1, j2, bone_name in BONES:
        if valid_mask[j1] and valid_mask[j2]:
            length = np.linalg.norm(positions_3d[j1] - positions_3d[j2])
            if length > 0.01:  # Min 1cm to avoid division issues
                expected = BONE_LENGTHS[bone_name]
                ratios.append(expected / length)
    
    if len(ratios) == 0:
        return 1.0
    
    return np.median(ratios)


def triangulate_skeleton(
    observation: SkeletonObservation,
    projection_matrices: Dict[int, np.ndarray],
    min_confidence: float = 0.3,
    min_views: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate a full skeleton from multi-view 2D keypoints.
    
    Returns:
        positions_3d: (17, 3) joint positions
        valid_mask: (17,) boolean mask
    """
    positions_3d = np.zeros((17, 3))
    valid_mask = np.zeros(17, dtype=bool)
    
    for joint_idx in range(17):
        points_2d = {}
        for cam_id, kpts in observation.keypoints_2d.items():
            conf = observation.confidences.get(cam_id, np.ones(17))
            if conf[joint_idx] >= min_confidence and cam_id in projection_matrices:
                points_2d[cam_id] = kpts[joint_idx]
        
        if len(points_2d) >= min_views:
            pt = triangulate_point(points_2d, projection_matrices)
            if pt is not None:
                positions_3d[joint_idx] = pt
                valid_mask[joint_idx] = True
    
    return positions_3d, valid_mask


def refine_cameras_bundle_adjustment(
    initial_cameras: Dict[int, CameraExtrinsics],
    observations_skeleton: List[SkeletonObservation],
    observations_charuco: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],  # (cam_id, corners, ids, obj_pts)
    intrinsics: Dict[int, CameraIntrinsics],
    charuco_weight: float = 1.0,
    skeleton_weight: float = 0.1,
    max_iterations: int = 100,
) -> Dict[int, CameraExtrinsics]:
    """
    Refine camera positions using bundle adjustment.
    
    ChArUco observations are weighted more heavily than skeleton observations
    (default 10:1 ratio).
    
    Args:
        initial_cameras: Initial camera extrinsics estimate
        observations_skeleton: Skeleton observations (multi-camera 2D keypoints)
        observations_charuco: ChArUco observations (cam_id, corners, ids, object_points)
        intrinsics: Camera intrinsics
        charuco_weight: Weight for ChArUco observations (default 1.0)
        skeleton_weight: Weight for skeleton observations (default 0.1)
        
    Returns:
        Refined camera extrinsics
    """
    camera_ids = sorted(initial_cameras.keys())
    n_cams = len(camera_ids)
    
    if n_cams == 0:
        return initial_cameras
    
    # Pack camera parameters: 6 params per camera (rodrigues + translation)
    # First camera is reference (fixed at origin looking down Z)
    def pack_cameras(cameras: Dict[int, CameraExtrinsics]) -> np.ndarray:
        params = []
        for cam_id in camera_ids[1:]:  # Skip reference camera
            ext = cameras[cam_id]
            rvec, _ = cv2.Rodrigues(ext.rotation_matrix)
            params.extend(rvec.flatten())
            params.extend(ext.translation.flatten())
        return np.array(params)
    
    def unpack_cameras(params: np.ndarray) -> Dict[int, CameraExtrinsics]:
        cameras = {}
        # Reference camera at origin
        cameras[camera_ids[0]] = CameraExtrinsics(
            camera_id=camera_ids[0],
            rotation_matrix=np.eye(3),
            translation=np.zeros(3),
        )
        
        idx = 0
        for cam_id in camera_ids[1:]:
            rvec = params[idx:idx+3]
            tvec = params[idx+3:idx+6]
            R, _ = cv2.Rodrigues(rvec)
            cameras[cam_id] = CameraExtrinsics(
                camera_id=cam_id,
                rotation_matrix=R,
                translation=tvec,
            )
            idx += 6
        return cameras
    
    def project_point(pt_3d: np.ndarray, cam_id: int, cameras: Dict[int, CameraExtrinsics]) -> np.ndarray:
        ext = cameras[cam_id]
        intr = intrinsics[cam_id]
        
        # World to camera transform
        pt_cam = ext.rotation_matrix @ pt_3d + ext.translation
        if pt_cam[2] <= 0:
            return np.array([np.nan, np.nan])
        
        # Project
        x = (intr.fx * pt_cam[0] / pt_cam[2]) + intr.cx
        y = (intr.fy * pt_cam[1] / pt_cam[2]) + intr.cy
        return np.array([x, y])
    
    def residuals(params: np.ndarray) -> np.ndarray:
        cameras = unpack_cameras(params)
        proj_mats = {cid: compute_projection_matrix(intrinsics[cid], cameras[cid]) 
                     for cid in camera_ids}
        
        residuals_list = []
        
        # Skeleton reprojection residuals
        for obs in observations_skeleton:
            # Triangulate with current cameras
            pos_3d, valid = triangulate_skeleton(obs, proj_mats)
            
            # Reproject and compute error
            for cam_id, kpts_2d in obs.keypoints_2d.items():
                if cam_id not in cameras:
                    continue
                conf = obs.confidences.get(cam_id, np.ones(17))
                
                for j in range(17):
                    if valid[j] and conf[j] > 0.3:
                        proj = project_point(pos_3d[j], cam_id, cameras)
                        if not np.isnan(proj[0]):
                            err = (proj - kpts_2d[j]) * skeleton_weight * conf[j]
                            residuals_list.extend(err)
        
        # ChArUco reprojection residuals (weighted higher)
        for cam_id, corners, ids, obj_pts in observations_charuco:
            if cam_id not in cameras:
                continue
            
            for i, pt_3d in enumerate(obj_pts):
                proj = project_point(pt_3d, cam_id, cameras)
                if not np.isnan(proj[0]):
                    err = (proj - corners[i].flatten()) * charuco_weight
                    residuals_list.extend(err)
        
        if len(residuals_list) == 0:
            return np.array([0.0])
        
        return np.array(residuals_list)
    
    # Need OpenCV for Rodrigues
    import cv2
    
    x0 = pack_cameras(initial_cameras)
    
    if len(x0) == 0:
        return initial_cameras
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=max_iterations)
        return unpack_cameras(result.x)
    except Exception as e:
        print(f"Bundle adjustment failed: {e}")
        return initial_cameras


def estimate_cameras_from_skeleton(
    observations: List[SkeletonObservation],
    intrinsics: Dict[int, CameraIntrinsics],
    min_confidence: float = 0.3,
) -> SkeletonCalibrationResult:
    """
    Estimate camera positions from skeleton observations alone.
    
    This is a fallback when ChArUco calibration is not available.
    The result will have is_skeleton_only=True to indicate lower accuracy.
    
    Args:
        observations: List of skeleton observations across cameras
        intrinsics: Camera intrinsics for each camera
        min_confidence: Minimum keypoint confidence threshold
        
    Returns:
        SkeletonCalibrationResult with estimated cameras
    """
    if len(observations) == 0:
        return SkeletonCalibrationResult(
            cameras={},
            scale_factor=1.0,
            reprojection_error=float('inf'),
            bone_length_error=float('inf'),
            num_observations=0,
            is_skeleton_only=True,
        )
    
    # Step 1: Estimate initial camera positions
    cameras = estimate_initial_cameras(observations, intrinsics, min_confidence)
    
    if len(cameras) == 0:
        return SkeletonCalibrationResult(
            cameras={},
            scale_factor=1.0,
            reprojection_error=float('inf'),
            bone_length_error=float('inf'),
            num_observations=0,
            is_skeleton_only=True,
        )
    
    # Step 2: Compute projection matrices
    proj_mats = {cid: compute_projection_matrix(intrinsics[cid], cameras[cid]) 
                 for cid in cameras}
    
    # Step 3: Triangulate skeletons and estimate scale
    scale_factors = []
    bone_errors = []
    
    for obs in observations:
        pos_3d, valid = triangulate_skeleton(obs, proj_mats, min_confidence)
        if np.sum(valid) >= 6:  # Need at least 6 joints
            scale = estimate_scale_from_bones(pos_3d, valid)
            scale_factors.append(scale)
            
            # Apply scale and compute bone error
            pos_3d_scaled = pos_3d * scale
            err, _ = compute_bone_length_error(pos_3d_scaled, valid)
            bone_errors.append(err)
    
    if len(scale_factors) > 0:
        final_scale = np.median(scale_factors)
        avg_bone_error = np.mean(bone_errors)
    else:
        final_scale = 1.0
        avg_bone_error = float('inf')
    
    # Step 4: Refine cameras with bundle adjustment (skeleton only)
    cameras = refine_cameras_bundle_adjustment(
        cameras,
        observations_skeleton=observations,
        observations_charuco=[],
        intrinsics=intrinsics,
        skeleton_weight=1.0,  # Full weight since skeleton-only
    )
    
    # Compute final reprojection error
    proj_mats = {cid: compute_projection_matrix(intrinsics[cid], cameras[cid]) 
                 for cid in cameras}
    
    total_reproj_error = 0.0
    reproj_count = 0
    
    for obs in observations:
        pos_3d, valid = triangulate_skeleton(obs, proj_mats, min_confidence)
        
        for cam_id, kpts_2d in obs.keypoints_2d.items():
            if cam_id not in cameras:
                continue
            
            for j in range(17):
                if valid[j]:
                    ext = cameras[cam_id]
                    intr = intrinsics[cam_id]
                    pt_cam = ext.rotation_matrix @ (pos_3d[j] * final_scale) + ext.translation
                    if pt_cam[2] > 0:
                        proj_x = (intr.fx * pt_cam[0] / pt_cam[2]) + intr.cx
                        proj_y = (intr.fy * pt_cam[1] / pt_cam[2]) + intr.cy
                        err = np.sqrt((proj_x - kpts_2d[j, 0])**2 + (proj_y - kpts_2d[j, 1])**2)
                        total_reproj_error += err
                        reproj_count += 1
    
    avg_reproj_error = total_reproj_error / max(reproj_count, 1)
    
    return SkeletonCalibrationResult(
        cameras=cameras,
        scale_factor=final_scale,
        reprojection_error=avg_reproj_error,
        bone_length_error=avg_bone_error,
        num_observations=len(observations),
        is_skeleton_only=True,
    )


def refine_cameras_from_poses(
    initial_cameras: Dict[int, CameraExtrinsics],
    observations_charuco: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    observations_skeleton: List[SkeletonObservation],
    intrinsics: Dict[int, CameraIntrinsics],
    charuco_weight: float = 1.0,
    skeleton_weight: float = 0.1,
) -> SkeletonCalibrationResult:
    """
    Refine camera positions using both ChArUco and skeleton observations.
    
    ChArUco observations are weighted much higher (default 10:1 ratio).
    
    Args:
        initial_cameras: Initial camera extrinsics
        observations_charuco: ChArUco detections (cam_id, corners, ids, obj_pts)
        observations_skeleton: Skeleton observations
        intrinsics: Camera intrinsics
        charuco_weight: Weight for ChArUco (default 1.0)
        skeleton_weight: Weight for skeleton (default 0.1)
        
    Returns:
        SkeletonCalibrationResult with refined cameras
    """
    is_skeleton_only = len(observations_charuco) == 0
    
    cameras = refine_cameras_bundle_adjustment(
        initial_cameras,
        observations_skeleton=observations_skeleton,
        observations_charuco=observations_charuco,
        intrinsics=intrinsics,
        charuco_weight=charuco_weight,
        skeleton_weight=skeleton_weight,
    )
    
    # Compute metrics
    proj_mats = {cid: compute_projection_matrix(intrinsics[cid], cameras[cid]) 
                 for cid in cameras}
    
    total_reproj_error = 0.0
    reproj_count = 0
    bone_errors = []
    
    for obs in observations_skeleton:
        pos_3d, valid = triangulate_skeleton(obs, proj_mats)
        if np.sum(valid) >= 6:
            err, _ = compute_bone_length_error(pos_3d, valid)
            bone_errors.append(err)
        
        for cam_id, kpts_2d in obs.keypoints_2d.items():
            if cam_id not in cameras:
                continue
            
            for j in range(17):
                if valid[j]:
                    ext = cameras[cam_id]
                    intr = intrinsics[cam_id]
                    pt_cam = ext.rotation_matrix @ pos_3d[j] + ext.translation
                    if pt_cam[2] > 0:
                        proj_x = (intr.fx * pt_cam[0] / pt_cam[2]) + intr.cx
                        proj_y = (intr.fy * pt_cam[1] / pt_cam[2]) + intr.cy
                        err = np.sqrt((proj_x - kpts_2d[j, 0])**2 + (proj_y - kpts_2d[j, 1])**2)
                        total_reproj_error += err
                        reproj_count += 1
    
    return SkeletonCalibrationResult(
        cameras=cameras,
        scale_factor=1.0,  # ChArUco provides absolute scale
        reprojection_error=total_reproj_error / max(reproj_count, 1),
        bone_length_error=np.mean(bone_errors) if bone_errors else 0.0,
        num_observations=len(observations_skeleton) + len(observations_charuco),
        is_skeleton_only=is_skeleton_only,
    )


def is_bone_length_valid(
    positions_3d: np.ndarray,
    valid_mask: np.ndarray,
) -> bool:
    """
    Check if triangulated skeleton has plausible bone lengths.
    
    Returns True if bone lengths are within human range.
    """
    for j1, j2, bone_name in BONES:
        if valid_mask[j1] and valid_mask[j2]:
            length = np.linalg.norm(positions_3d[j1] - positions_3d[j2])
            expected = BONE_LENGTHS[bone_name]
            min_len = expected * BONE_TOLERANCE[0]
            max_len = expected * BONE_TOLERANCE[1]
            
            if length < min_len or length > max_len:
                return False
    
    return True
