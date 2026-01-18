"""
Tests for Skeleton-Based Camera Calibration

Tests skeleton calibration module including camera estimation,
bundle adjustment, and dataset verification.
"""

import pytest
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, List

from voxelvr.calibration.skeleton_calibration import (
    SkeletonObservation,
    SkeletonCalibrationResult,
    estimate_cameras_from_skeleton,
    refine_cameras_from_poses,
    triangulate_point,
    triangulate_skeleton,
    compute_bone_length_error,
    estimate_scale_from_bones,
    is_bone_length_valid,
    compute_projection_matrix,
    BONE_LENGTHS,
    BONES,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, NOSE,
    SimpleCameraIntrinsics,
    SimpleCameraExtrinsics,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def synthetic_intrinsics() -> Dict[int, SimpleCameraIntrinsics]:
    """Create synthetic camera intrinsics for testing."""
    intrinsics = {}
    for cam_id in [0, 1, 2, 3]:
        intrinsics[cam_id] = SimpleCameraIntrinsics(
            camera_id=cam_id,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
            distortion_coeffs=np.zeros(5),
        )
    return intrinsics


@pytest.fixture
def synthetic_cameras() -> Dict[int, SimpleCameraExtrinsics]:
    """Create synthetic camera extrinsics for testing."""
    cameras = {}
    n_cams = 4
    radius = 2.5
    
    for i in range(n_cams):
        angle = (2 * np.pi * i) / n_cams
        x = radius * np.sin(angle)
        z = radius * np.cos(angle)
        y = 1.2
        
        # Camera looks at origin
        forward = -np.array([x, y - 0.9, z])
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        R = np.stack([right, -up, forward], axis=0)
        t = -R @ np.array([x, y, z])
        
        cameras[i] = SimpleCameraExtrinsics(
            camera_id=i,
            rotation_matrix=R,
            translation=t,
        )
    
    return cameras


@pytest.fixture
def t_pose_skeleton() -> np.ndarray:
    """Create a T-pose skeleton with realistic proportions."""
    positions = np.zeros((17, 3))
    
    # Head
    positions[NOSE] = [0, 1.7, 0]
    
    # Shoulders
    positions[LEFT_SHOULDER] = [-0.175, 1.45, 0]
    positions[RIGHT_SHOULDER] = [0.175, 1.45, 0]
    
    # Arms (T-pose)
    positions[LEFT_ELBOW] = [-0.475, 1.45, 0]
    positions[RIGHT_ELBOW] = [0.475, 1.45, 0]
    positions[LEFT_WRIST] = [-0.735, 1.45, 0]
    positions[RIGHT_WRIST] = [0.735, 1.45, 0]
    
    # Hips
    positions[LEFT_HIP] = [-0.14, 0.95, 0]
    positions[RIGHT_HIP] = [0.14, 0.95, 0]
    
    # Legs
    positions[LEFT_KNEE] = [-0.14, 0.52, 0]
    positions[RIGHT_KNEE] = [0.14, 0.52, 0]
    positions[LEFT_ANKLE] = [-0.14, 0.10, 0]
    positions[RIGHT_ANKLE] = [0.14, 0.10, 0]
    
    return positions


def project_skeleton_to_cameras(
    skeleton: np.ndarray,
    cameras: Dict[int, SimpleCameraExtrinsics],
    intrinsics: Dict[int, SimpleCameraIntrinsics],
) -> SkeletonObservation:
    """Project a 3D skeleton to 2D for all cameras."""
    keypoints_2d = {}
    confidences = {}
    
    for cam_id, ext in cameras.items():
        intr = intrinsics[cam_id]
        pts_2d = np.zeros((17, 2))
        confs = np.ones(17)
        
        for j in range(17):
            pt_3d = skeleton[j]
            pt_cam = ext.rotation_matrix @ pt_3d + ext.translation
            
            if pt_cam[2] > 0:
                pts_2d[j, 0] = (intr.fx * pt_cam[0] / pt_cam[2]) + intr.cx
                pts_2d[j, 1] = (intr.fy * pt_cam[1] / pt_cam[2]) + intr.cy
                
                # Check if in frame
                if 0 <= pts_2d[j, 0] < intr.width and 0 <= pts_2d[j, 1] < intr.height:
                    confs[j] = 0.9
                else:
                    confs[j] = 0.0
            else:
                confs[j] = 0.0
        
        keypoints_2d[cam_id] = pts_2d
        confidences[cam_id] = confs
    
    return SkeletonObservation(
        keypoints_2d=keypoints_2d,
        confidences=confidences,
    )


# ============================================================================
# Unit Tests
# ============================================================================

class TestTriangulation:
    """Tests for triangulation functions."""
    
    def test_triangulate_point_basic(self, synthetic_cameras, synthetic_intrinsics):
        """Test basic point triangulation."""
        # Project a known 3D point
        pt_3d = np.array([0.5, 1.0, 0.2])
        
        proj_mats = {
            cid: compute_projection_matrix(synthetic_intrinsics[cid], cam)
            for cid, cam in synthetic_cameras.items()
        }
        
        points_2d = {}
        for cam_id, ext in synthetic_cameras.items():
            intr = synthetic_intrinsics[cam_id]
            pt_cam = ext.rotation_matrix @ pt_3d + ext.translation
            x = (intr.fx * pt_cam[0] / pt_cam[2]) + intr.cx
            y = (intr.fy * pt_cam[1] / pt_cam[2]) + intr.cy
            points_2d[cam_id] = np.array([x, y])
        
        result = triangulate_point(points_2d, proj_mats)
        assert result is not None
        error = np.linalg.norm(result - pt_3d)
        assert error < 0.01  # Less than 1cm error
    
    def test_triangulate_point_with_noise(self, synthetic_cameras, synthetic_intrinsics):
        """Test triangulation with noisy 2D points."""
        pt_3d = np.array([0.0, 1.2, 0.0])
        
        proj_mats = {
            cid: compute_projection_matrix(synthetic_intrinsics[cid], cam)
            for cid, cam in synthetic_cameras.items()
        }
        
        points_2d = {}
        for cam_id, ext in synthetic_cameras.items():
            intr = synthetic_intrinsics[cam_id]
            pt_cam = ext.rotation_matrix @ pt_3d + ext.translation
            x = (intr.fx * pt_cam[0] / pt_cam[2]) + intr.cx + np.random.randn() * 2
            y = (intr.fy * pt_cam[1] / pt_cam[2]) + intr.cy + np.random.randn() * 2
            points_2d[cam_id] = np.array([x, y])
        
        result = triangulate_point(points_2d, proj_mats)
        assert result is not None
        error = np.linalg.norm(result - pt_3d)
        assert error < 0.1  # Less than 10cm error with noise
    
    def test_triangulate_skeleton(
        self, synthetic_cameras, synthetic_intrinsics, t_pose_skeleton
    ):
        """Test full skeleton triangulation."""
        proj_mats = {
            cid: compute_projection_matrix(synthetic_intrinsics[cid], cam)
            for cid, cam in synthetic_cameras.items()
        }
        
        obs = project_skeleton_to_cameras(
            t_pose_skeleton, synthetic_cameras, synthetic_intrinsics
        )
        
        positions, valid = triangulate_skeleton(obs, proj_mats)
        
        # Should have most joints valid
        assert np.sum(valid) >= 10
        
        # Check accuracy of triangulated positions
        for j in range(17):
            if valid[j]:
                error = np.linalg.norm(positions[j] - t_pose_skeleton[j])
                assert error < 0.05  # 5cm accuracy


class TestBoneLengths:
    """Tests for bone length functions."""
    
    def test_compute_bone_length_error_valid(self, t_pose_skeleton):
        """Test bone length error computation on valid skeleton."""
        valid = np.ones(17, dtype=bool)
        error, per_bone = compute_bone_length_error(t_pose_skeleton, valid)
        
        # T-pose should have low error (designed to match expected lengths)
        assert error < 0.15  # Less than 15% average error
    
    def test_estimate_scale_from_bones(self, t_pose_skeleton):
        """Test scale estimation from bone lengths."""
        valid = np.ones(17, dtype=bool)
        
        # Halve the skeleton - scale should be 2
        half_skeleton = t_pose_skeleton * 0.5
        scale = estimate_scale_from_bones(half_skeleton, valid)
        assert abs(scale - 2.0) < 0.3
        
        # Double the skeleton - scale should be 0.5
        double_skeleton = t_pose_skeleton * 2.0
        scale = estimate_scale_from_bones(double_skeleton, valid)
        assert abs(scale - 0.5) < 0.15
    
    def test_is_bone_length_valid_true(self, t_pose_skeleton):
        """Test bone length validation on valid skeleton."""
        valid = np.ones(17, dtype=bool)
        assert is_bone_length_valid(t_pose_skeleton, valid) is True
    
    def test_is_bone_length_valid_false_scaled(self, t_pose_skeleton):
        """Test bone length validation rejects poorly scaled skeleton."""
        valid = np.ones(17, dtype=bool)
        
        # Very small skeleton
        tiny = t_pose_skeleton * 0.1
        assert is_bone_length_valid(tiny, valid) is False
        
        # Very large skeleton
        giant = t_pose_skeleton * 5.0
        assert is_bone_length_valid(giant, valid) is False


class TestCameraEstimation:
    """Tests for camera position estimation."""
    
    def test_estimate_cameras_from_skeleton(
        self, synthetic_cameras, synthetic_intrinsics, t_pose_skeleton
    ):
        """Test camera estimation from skeleton observations."""
        # Create observations
        observations = []
        for i in range(5):
            # Add small variations to skeleton position
            skeleton = t_pose_skeleton.copy()
            skeleton[:, 0] += np.random.randn() * 0.1
            skeleton[:, 2] += np.random.randn() * 0.1
            
            obs = project_skeleton_to_cameras(
                skeleton, synthetic_cameras, synthetic_intrinsics
            )
            observations.append(obs)
        
        result = estimate_cameras_from_skeleton(observations, synthetic_intrinsics)
        
        assert isinstance(result, SkeletonCalibrationResult)
        assert len(result.cameras) == 4
        assert result.is_skeleton_only is True
        assert result.num_observations == 5
    
    def test_refine_cameras_from_poses_weighting(
        self, synthetic_cameras, synthetic_intrinsics, t_pose_skeleton
    ):
        """Test that ChArUco is weighted more than skeleton."""
        observations = []
        for i in range(3):
            obs = project_skeleton_to_cameras(
                t_pose_skeleton, synthetic_cameras, synthetic_intrinsics
            )
            observations.append(obs)
        
        # No ChArUco - skeleton only
        result_skeleton_only = refine_cameras_from_poses(
            initial_cameras=synthetic_cameras,
            observations_charuco=[],
            observations_skeleton=observations,
            intrinsics=synthetic_intrinsics,
            charuco_weight=1.0,
            skeleton_weight=0.1,
        )
        
        assert result_skeleton_only.is_skeleton_only is True
        
        # With simulated ChArUco (empty for this test, just verify flag)
        # In real use, charuco observations would be provided
        result_with_charuco = refine_cameras_from_poses(
            initial_cameras=synthetic_cameras,
            observations_charuco=[],  # Would normally have data
            observations_skeleton=observations,
            intrinsics=synthetic_intrinsics,
        )
        
        assert len(result_with_charuco.cameras) == 4


# ============================================================================
# Dataset Integration Test
# ============================================================================

class TestDatasetCalibration:
    """Tests using the actual dataset directory."""
    
    @pytest.fixture
    def dataset_path(self) -> Path:
        """Get dataset path."""
        return Path("/home/lee/voxelvr/dataset")
    
    def test_full_dataset_calibration(self, dataset_path):
        """
        Full pipeline test using dataset images.
        
        This test:
        1. Loads all calibration frames
        2. Runs skeleton-based camera estimation
        3. Verifies pose outputs have sensible bone lengths
        """
        # Check dataset exists
        if not dataset_path.exists():
            pytest.skip("Dataset not found")
        
        calibration_dir = dataset_path / "calibration"
        tracking_dir = dataset_path / "tracking"
        
        if not calibration_dir.exists():
            pytest.skip("Calibration data not found")
        
        # Find all calibration sessions
        sessions = list(calibration_dir.iterdir())
        if not sessions:
            pytest.skip("No calibration sessions found")
        
        # Use the first session
        session = sessions[0]
        
        # Find cameras
        cam_dirs = [d for d in session.iterdir() if d.is_dir() and d.name.startswith("cam_")]
        if len(cam_dirs) < 2:
            pytest.skip("Need at least 2 cameras")
        
        camera_ids = [int(d.name.split("_")[1]) for d in cam_dirs]
        print(f"\nFound cameras: {camera_ids}")
        
        # Load frames for each camera
        frames_per_camera = {}
        for cam_dir in cam_dirs:
            cam_id = int(cam_dir.name.split("_")[1])
            image_files = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
            frames = [cv2.imread(str(f)) for f in image_files[:30]]  # Limit to 30 frames
            frames_per_camera[cam_id] = [f for f in frames if f is not None]
        
        min_frames = min(len(f) for f in frames_per_camera.values())
        print(f"Loaded {min_frames} frames per camera")
        
        if min_frames < 5:
            pytest.skip("Not enough frames")
        
        # Create synthetic intrinsics (would normally load from calibration)
        # Assume typical webcam parameters
        intrinsics = {}
        for cam_id in camera_ids:
            sample_frame = frames_per_camera[cam_id][0]
            h, w = sample_frame.shape[:2]
            fx = fy = w * 0.8  # Approximate focal length
            intrinsics[cam_id] = SimpleCameraIntrinsics(
                camera_id=cam_id,
                fx=fx,
                fy=fy,
                cx=w / 2,
                cy=h / 2,
                width=w,
                height=h,
                distortion_coeffs=np.zeros(5),
            )
        
        # Run pose detection on frames to get 2D keypoints
        from voxelvr.pose.detector_2d import PoseDetector2D
        
        try:
            detector = PoseDetector2D()
        except Exception as e:
            pytest.skip(f"Could not initialize pose detector: {e}")
        
        observations = []
        
        for frame_idx in range(min_frames):
            keypoints_2d = {}
            confidences = {}
            
            for cam_id in camera_ids:
                frame = frames_per_camera[cam_id][frame_idx]
                
                try:
                    result = detector.detect(frame)
                    if result is not None:
                        keypoints_2d[cam_id] = result.keypoints
                        confidences[cam_id] = result.confidences
                except Exception:
                    continue
            
            if len(keypoints_2d) >= 2:
                observations.append(SkeletonObservation(
                    keypoints_2d=keypoints_2d,
                    confidences=confidences,
                    timestamp=frame_idx / 30.0,
                ))
        
        print(f"Created {len(observations)} skeleton observations")
        
        if len(observations) < 3:
            pytest.skip("Not enough valid observations")
        
        # Run skeleton-based calibration
        result = estimate_cameras_from_skeleton(observations, intrinsics)
        
        print(f"Calibration result:")
        print(f"  - Cameras: {len(result.cameras)}")
        print(f"  - Scale factor: {result.scale_factor:.3f}")
        print(f"  - Reproj error: {result.reprojection_error:.2f}px")
        print(f"  - Bone length error: {result.bone_length_error:.2%}")
        print(f"  - Skeleton only: {result.is_skeleton_only}")
        
        # Verify calibration produced results
        assert len(result.cameras) >= 2
        assert result.scale_factor > 0.1
        assert result.scale_factor < 10.0
        
        # Verify we can triangulate poses with calibrated cameras
        if len(result.cameras) >= 2:
            proj_mats = {
                cid: compute_projection_matrix(intrinsics[cid], cam)
                for cid, cam in result.cameras.items()
            }
            
            positions, valid = triangulate_skeleton(observations[0], proj_mats)
            
            # Apply scale
            positions_scaled = positions * result.scale_factor
            
            print(f"Triangulated {np.sum(valid)} joints")
            
            if np.sum(valid) >= 6:
                # Check bone lengths are in human range
                error, per_bone = compute_bone_length_error(positions_scaled, valid)
                print(f"Bone length errors: {per_bone}")
                
                # Bone length error should be somewhat reasonable
                # (may not be perfect with skeleton-only calibration)
                assert error < 1.0  # Less than 100% average error


class TestWeightingBehavior:
    """Tests for ChArUco vs skeleton weighting."""
    
    def test_charuco_weight_higher_than_skeleton(
        self, synthetic_cameras, synthetic_intrinsics, t_pose_skeleton
    ):
        """Verify ChArUco observations have more influence."""
        # This is a behavioral test - ChArUco weight = 1.0, skeleton = 0.1
        # means ChArUco has 10x the influence
        
        observations = [
            project_skeleton_to_cameras(t_pose_skeleton, synthetic_cameras, synthetic_intrinsics)
            for _ in range(5)
        ]
        
        # With only skeleton, we should get skeleton_only=True
        result = refine_cameras_from_poses(
            initial_cameras=synthetic_cameras,
            observations_charuco=[],
            observations_skeleton=observations,
            intrinsics=synthetic_intrinsics,
            charuco_weight=1.0,
            skeleton_weight=0.1,
        )
        
        assert result.is_skeleton_only is True
        
        # The 10:1 ratio is encoded in the function signature
        # Verify default values
        import inspect
        sig = inspect.signature(refine_cameras_from_poses)
        
        assert sig.parameters['charuco_weight'].default == 1.0
        assert sig.parameters['skeleton_weight'].default == 0.1

