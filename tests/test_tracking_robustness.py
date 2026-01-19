"""
Tests for Tracking Robustness

Verifies that the tracking pipeline can handle:
- Noisy 2D detections (jitter)
- Outlier cameras (bad calibration or detection)
- Missing data (occlusion)
"""

import pytest
import numpy as np
from voxelvr.pose.triangulation import triangulate_points, triangulate_pose, ransac_triangulate
from tests.conftest import (
    generate_t_pose,
    generate_multi_camera_setup,
    project_3d_to_2d,
)
from dataclasses import dataclass

@dataclass
class MockKeypoints2D:
    positions: np.ndarray
    confidences: np.ndarray
    camera_id: int
    image_width: int = 1280
    image_height: int = 720

class TestTriangulationRobustness:
    
    def test_noise_tolerance(self):
        """Test triangulation accuracy under Gaussian noise."""
        point_3d_gt = np.array([[0.0, 1.5, 1.0]])
        cameras = generate_multi_camera_setup(num_cameras=4)
        
        points_2d = {}
        projection_matrices = {}
        
        noise_std_dev = 5.0 # pixels
        
        for cam in cameras:
            K = cam['intrinsics']
            T_cam_to_world = cam['extrinsics']
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            R, t = T_world_to_cam[:3, :3], T_world_to_cam[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
            
            p2d, _ = project_3d_to_2d(point_3d_gt, K, T_cam_to_world, add_noise=0.0) # Project clean first
            
            # Add significant noise
            noise = np.random.normal(0, noise_std_dev, p2d.shape)
            points_2d[cam['id']] = p2d + noise
            
        # Standard DLT might fail or give high error with high noise
        # We want to check if RANSAC does better or if DLT is "good enough" for this level
        
        # Test DLT
        pt_dlt, _ = triangulate_points(points_2d, projection_matrices)
        err_dlt = np.linalg.norm(pt_dlt[0] - point_3d_gt[0])
        
        # Test RANSAC
        pt_ransac, _, _ = ransac_triangulate(points_2d, projection_matrices, threshold=10.0)
        err_ransac = np.linalg.norm(pt_ransac - point_3d_gt[0])
        
        print(f"DLT Error: {err_dlt:.4f} m, RANSAC Error: {err_ransac:.4f} m")
        
        # With 4 cameras and random noise, DLT averages it out, so it should be decent.
        # But RANSAC should also be robust.
        assert err_dlt < 0.2, f"DLT error too high with {noise_std_dev}px noise: {err_dlt}"
        assert err_ransac < 0.2, f"RANSAC error too high with {noise_std_dev}px noise: {err_ransac}"

    def test_outlier_rejection(self):
        """Test that RANSAC rejects a camera that is completely wrong."""
        point_3d_gt = np.array([[0.0, 1.5, 0.0]])
        cameras = generate_multi_camera_setup(num_cameras=5)
        
        points_2d = {}
        projection_matrices = {}
        
        for i, cam in enumerate(cameras):
            K = cam['intrinsics']
            T_cam_to_world = cam['extrinsics']
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            R, t = T_world_to_cam[:3, :3], T_world_to_cam[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
            
            if i == 0:
                # Camera 0 sees the point in a completely wrong place (e.g. reflection or bug)
                points_2d[cam['id']] = np.array([[100, 100]], dtype=np.float32) 
            else:
                p2d, _ = project_3d_to_2d(point_3d_gt, K, T_cam_to_world)
                points_2d[cam['id']] = p2d
        
        # RANSAC should find the point and exclude camera 0
        pt_ransac, _, inliers = ransac_triangulate(points_2d, projection_matrices, threshold=5.0)
        
        err_ransac = np.linalg.norm(pt_ransac - point_3d_gt[0])
        
        assert cameras[0]['id'] not in inliers, "Outlier camera should not be an inlier"
        assert len(inliers) >= 3, "Should have at least 4 valid cameras"
        assert err_ransac < 0.05, "RANSAC should recover accurate position"

    def test_full_pose_robustness(self):
        """Test full pose triangulation with random dropouts and noise."""
        pose_3d_gt = generate_t_pose()
        cameras = generate_multi_camera_setup(num_cameras=4)
        projection_matrices = {}
        keypoints_list = []
        
        for cam in cameras:
            K = cam['intrinsics']
            T_cam_to_world = cam['extrinsics']
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            R, t = T_world_to_cam[:3, :3], T_world_to_cam[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
            
            p2d, _ = project_3d_to_2d(pose_3d_gt, K, T_cam_to_world, add_noise=2.0)
            
            # Randomly drop some joints (set confidence to 0)
            conf = np.ones(17, dtype=np.float32)
            drop_indices = np.random.choice(17, size=2, replace=False)
            conf[drop_indices] = 0.0
            
            kp = MockKeypoints2D(p2d, conf, cam['id'])
            keypoints_list.append(kp)
            
        positions, confidences, valid = triangulate_pose(keypoints_list, projection_matrices)
        
        valid_count = np.sum(valid)
        assert valid_count > 10, f"Expected most joints to be valid, got {valid_count}"
        
        # Check accuracy of a valid joint
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[0]
            err = np.linalg.norm(positions[idx] - pose_3d_gt[idx])
            assert err < 0.1, f"Joint error too high: {err}"
