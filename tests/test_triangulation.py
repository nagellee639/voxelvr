"""
Multi-View Triangulation Tests

Tests the 3D reconstruction from multiple 2D views without cameras.
Uses synthetic data to verify geometric correctness.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from conftest import (
    generate_t_pose,
    generate_camera_intrinsics,
    generate_camera_extrinsics,
    generate_multi_camera_setup,
    project_3d_to_2d,
)


@dataclass
class MockKeypoints2D:
    """Mock Keypoints2D for testing without cv2."""
    positions: np.ndarray
    confidences: np.ndarray
    image_width: int = 1280
    image_height: int = 720
    camera_id: int = 0
    threshold: float = 0.3


class TestTriangulationBasics:
    """Basic triangulation functionality tests."""
    
    def test_triangulate_single_point(self):
        """Test triangulating a single 3D point from multiple views."""
        from voxelvr.pose.triangulation import triangulate_points
        
        # Ground truth point
        point_3d_gt = np.array([[0.0, 1.5, 0.0]])
        
        # Generate 3 camera views
        cameras = generate_multi_camera_setup(num_cameras=3)
        
        # Project to each camera and build projection matrices
        points_2d = {}
        projection_matrices = {}
        
        for cam in cameras:
            K = cam['intrinsics']
            T_cam_to_world = cam['extrinsics']
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            
            R = T_world_to_cam[:3, :3]
            t = T_world_to_cam[:3, 3]
            
            # Projection matrix P = K [R | t]
            Rt = np.hstack([R, t.reshape(3, 1)])
            P = K @ Rt
            projection_matrices[cam['id']] = P
            
            # Project point
            p2d, vis = project_3d_to_2d(point_3d_gt, K, T_cam_to_world)
            points_2d[cam['id']] = p2d
        
        # Triangulate
        points_3d, errors = triangulate_points(points_2d, projection_matrices)
        
        # Check result
        assert len(points_3d) == 1
        np.testing.assert_allclose(points_3d[0], point_3d_gt[0], atol=0.01)
    
    def test_triangulate_with_noise(self):
        """Test triangulation handles 2D noise gracefully."""
        from voxelvr.pose.triangulation import triangulate_points
        
        point_3d_gt = np.array([[0.0, 1.5, 0.0]])
        cameras = generate_multi_camera_setup(num_cameras=4)
        
        points_2d = {}
        projection_matrices = {}
        
        for cam in cameras:
            K = cam['intrinsics']
            T_cam_to_world = cam['extrinsics']
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            
            R = T_world_to_cam[:3, :3]
            t = T_world_to_cam[:3, 3]
            
            Rt = np.hstack([R, t.reshape(3, 1)])
            P = K @ Rt
            projection_matrices[cam['id']] = P
            
            # Project with noise
            p2d, _ = project_3d_to_2d(point_3d_gt, K, T_cam_to_world, add_noise=5.0)
            points_2d[cam['id']] = p2d
        
        points_3d, errors = triangulate_points(points_2d, projection_matrices)
        
        # With noise, we allow larger error
        assert len(points_3d) == 1
        error = np.linalg.norm(points_3d[0] - point_3d_gt[0])
        assert error < 0.1, f"Triangulation error too large: {error}"


class TestFullPoseTriangulation:
    """Test triangulation of full poses (17 keypoints)."""
    
    def test_triangulate_t_pose(self):
        """Test triangulating a complete T-pose."""
        from voxelvr.pose.triangulation import triangulate_pose
        
        # Ground truth
        pose_3d_gt = generate_t_pose()
        cameras = generate_multi_camera_setup(num_cameras=3)
        
        # Build projection matrices
        projection_matrices = {}
        keypoints_list = []
        
        for cam in cameras:
            K = cam['intrinsics']
            T_cam_to_world = cam['extrinsics']
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            
            R = T_world_to_cam[:3, :3]
            t = T_world_to_cam[:3, 3]
            
            Rt = np.hstack([R, t.reshape(3, 1)])
            P = K @ Rt
            projection_matrices[cam['id']] = P
            
            # Project all keypoints
            p2d, visibility = project_3d_to_2d(pose_3d_gt, K, T_cam_to_world, add_noise=2.0)
            
            # Create mock Keypoints2D object
            kp = MockKeypoints2D(
                positions=p2d,
                confidences=visibility.astype(np.float32) * 0.9,
                image_width=1280,
                image_height=720,
                camera_id=cam['id'],
            )
            keypoints_list.append(kp)
        
        # Triangulate
        positions_3d, confidences, valid_mask = triangulate_pose(
            keypoints_list,
            projection_matrices,
            confidence_threshold=0.3,
        )
        
        # Check results
        assert positions_3d.shape == (17, 3)
        assert np.sum(valid_mask) >= 15, f"Too few valid joints: {np.sum(valid_mask)}"
        
        # Check reconstruction accuracy for valid joints
        for i in range(17):
            if valid_mask[i]:
                error = np.linalg.norm(positions_3d[i] - pose_3d_gt[i])
                assert error < 0.1, f"Joint {i} error too large: {error}"
    
    def test_triangulate_with_occlusion(self):
        """Test triangulation when some joints are occluded in some views."""
        from voxelvr.pose.triangulation import triangulate_pose
        
        pose_3d_gt = generate_t_pose()
        cameras = generate_multi_camera_setup(num_cameras=4)
        
        projection_matrices = {}
        keypoints_list = []
        
        for cam_idx, cam in enumerate(cameras):
            K = cam['intrinsics']
            T_cam_to_world = cam['extrinsics']
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            
            R = T_world_to_cam[:3, :3]
            t = T_world_to_cam[:3, 3]
            
            Rt = np.hstack([R, t.reshape(3, 1)])
            P = K @ Rt
            projection_matrices[cam['id']] = P
            
            p2d, visibility = project_3d_to_2d(pose_3d_gt, K, T_cam_to_world)
            
            # Simulate occlusion: each camera misses different joints
            occlusion_mask = np.ones(17, dtype=bool)
            occluded_joints = [(cam_idx * 4 + j) % 17 for j in range(3)]
            for j in occluded_joints:
                occlusion_mask[j] = False
            
            confidences = (visibility & occlusion_mask).astype(np.float32) * 0.9
            
            kp = MockKeypoints2D(
                positions=p2d,
                confidences=confidences,
                image_width=1280,
                image_height=720,
                camera_id=cam['id'],
            )
            keypoints_list.append(kp)
        
        positions_3d, confidences, valid_mask = triangulate_pose(
            keypoints_list,
            projection_matrices,
            confidence_threshold=0.3,
        )
        
        # Should still reconstruct most joints
        assert np.sum(valid_mask) >= 12, f"Too few valid joints: {np.sum(valid_mask)}"


class TestTriangulationPipeline:
    """Test the complete triangulation pipeline."""
    
    def test_pipeline_initialization(self):
        """Test TriangulationPipeline can be initialized."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        
        cameras = generate_multi_camera_setup(num_cameras=3)
        
        projection_matrices = {}
        for cam in cameras:
            K = cam['intrinsics']
            T = np.linalg.inv(cam['extrinsics'])
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
        
        pipeline = TriangulationPipeline(projection_matrices)
        assert pipeline is not None
    
    def test_pipeline_process(self):
        """Test pipeline.process() returns valid output."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        
        pose_3d_gt = generate_t_pose()
        cameras = generate_multi_camera_setup(num_cameras=3)
        
        projection_matrices = {}
        keypoints_list = []
        
        for cam in cameras:
            K = cam['intrinsics']
            T_cam_to_world = cam['extrinsics']
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            
            R = T_world_to_cam[:3, :3]
            t = T_world_to_cam[:3, 3]
            
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
            
            p2d, visibility = project_3d_to_2d(pose_3d_gt, K, T_cam_to_world)
            
            kp = MockKeypoints2D(
                positions=p2d,
                confidences=visibility.astype(np.float32) * 0.9,
                image_width=1280,
                image_height=720,
                camera_id=cam['id'],
            )
            keypoints_list.append(kp)
        
        pipeline = TriangulationPipeline(projection_matrices)
        result = pipeline.process(keypoints_list)
        
        assert result is not None
        assert 'positions' in result
        assert 'confidences' in result
        assert 'valid' in result
        assert result['positions'].shape == (17, 3)


class TestRANSACTriangulation:
    """Test RANSAC-based robust triangulation."""
    
    def test_ransac_rejects_outliers(self):
        """Test that RANSAC rejects outlier detections."""
        from voxelvr.pose.triangulation import ransac_triangulate
        
        point_3d_gt = np.array([0.0, 1.5, 0.0])
        cameras = generate_multi_camera_setup(num_cameras=4)
        
        points_2d = {}
        projection_matrices = {}
        
        for cam in cameras:
            K = cam['intrinsics']
            T = np.linalg.inv(cam['extrinsics'])
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
            
            p2d, _ = project_3d_to_2d(point_3d_gt.reshape(1, 3), K, cam['extrinsics'])
            points_2d[cam['id']] = p2d
        
        # Add an outlier to one camera
        points_2d[0] = points_2d[0] + np.array([[200, 200]])  # Large error
        
        # RANSAC should still find the correct point
        best_point, error, inliers = ransac_triangulate(
            points_2d,
            projection_matrices,
            iterations=100,
            threshold=10.0,
        )
        
        # Should exclude the outlier camera
        assert 0 not in inliers, "Outlier camera should be excluded"
        assert len(inliers) >= 2, "Should have at least 2 inliers"
        
        # Result should be close to ground truth
        reconstruction_error = np.linalg.norm(best_point - point_3d_gt)
        assert reconstruction_error < 0.1, f"Error too large: {reconstruction_error}"
