"""
End-to-End Integration Tests

Tests the complete tracking pipeline with synthetic data.
Requires full dependencies to be installed.
"""

import pytest
import numpy as np
from conftest import (
    generate_t_pose,
    generate_walking_pose,
    generate_pose_sequence,
    generate_multi_camera_setup,
    project_3d_to_2d,
)

# Check for optional dependencies
try:
    import pydantic_settings
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    HAS_PYDANTIC_SETTINGS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pythonosc
    HAS_OSC = True
except ImportError:
    HAS_OSC = False

requires_pydantic = pytest.mark.skipif(not HAS_PYDANTIC_SETTINGS, reason="pydantic_settings not installed")
requires_cv2 = pytest.mark.skipif(not HAS_CV2, reason="opencv-python not installed")
requires_osc = pytest.mark.skipif(not HAS_OSC, reason="python-osc not installed")


class TestFullPipelineIntegration:
    """Test the complete tracking pipeline end-to-end."""
    
    @requires_pydantic
    @requires_osc
    def test_pipeline_initialization(self):
        """Test all pipeline components can be initialized."""
        from voxelvr.config import VoxelVRConfig
        from voxelvr.pose import PoseDetector2D, PoseFilter
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.rotation import RotationFilter
        from voxelvr.transport import OSCSender
        
        # Config
        config = VoxelVRConfig()
        assert config is not None
        
        # Pose detector (don't load model yet)
        detector = PoseDetector2D(backend="cpu")
        assert detector is not None
        
        # Pose filter
        pose_filter = PoseFilter(num_joints=17)
        assert pose_filter is not None
        
        # Rotation filter
        rotation_filter = RotationFilter()
        assert rotation_filter is not None
        
        # OSC sender (don't connect)
        osc_sender = OSCSender(ip="127.0.0.1", port=9000)
        assert osc_sender is not None
    
    @requires_cv2
    @requires_osc
    def test_synthetic_tracking_loop(self):
        """Test full tracking loop with synthetic data."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.detector_2d import Keypoints2D
        from voxelvr.pose import PoseFilter
        from voxelvr.pose.rotation import estimate_all_rotations, RotationFilter
        from voxelvr.transport.osc_sender import pose_to_trackers_with_rotations
        from voxelvr.transport.coordinate import (
            create_default_transform, 
            transform_pose_to_vrchat
        )
        
        # Setup
        cameras = generate_multi_camera_setup(num_cameras=3)
        
        projection_matrices = {}
        for cam in cameras:
            K = cam['intrinsics']
            T = np.linalg.inv(cam['extrinsics'])
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
        
        triangulation = TriangulationPipeline(projection_matrices)
        pose_filter = PoseFilter(num_joints=17)
        rotation_filter = RotationFilter()
        coord_transform = create_default_transform()
        
        # Simulate 1 second of tracking at 30 FPS
        frames_processed = 0
        reconstruction_errors = []
        
        for t, pose_3d_gt in generate_pose_sequence(duration=1.0, fps=30.0):
            # Simulate 2D detections
            keypoints_list = []
            for cam in cameras:
                p2d, vis = project_3d_to_2d(
                    pose_3d_gt,
                    cam['intrinsics'],
                    cam['extrinsics'],
                    add_noise=2.0
                )
                kp = Keypoints2D(
                    positions=p2d,
                    confidences=vis.astype(np.float32) * 0.9,
                    image_width=1280,
                    image_height=720,
                    camera_id=cam['id'],
                    timestamp=t,
                )
                keypoints_list.append(kp)
            
            # Triangulate
            result = triangulation.process(keypoints_list)
            assert result is not None, f"Triangulation failed at t={t}"
            
            # Filter positions
            filtered_pos = pose_filter.filter(result['positions'], result['valid'])
            
            # Transform to VRChat coordinates
            transformed, _ = transform_pose_to_vrchat(filtered_pos, coord_transform)
            
            # Build trackers with rotations
            trackers = pose_to_trackers_with_rotations(
                transformed,
                result['confidences'],
                result['valid'],
            )
            
            # Verify output
            assert len(trackers) > 0, f"No trackers generated at t={t}"
            
            # Calculate reconstruction error
            valid_indices = np.where(result['valid'])[0]
            if len(valid_indices) > 0:
                error = np.mean([
                    np.linalg.norm(filtered_pos[i] - pose_3d_gt[i])
                    for i in valid_indices
                ])
                reconstruction_errors.append(error)
            
            frames_processed += 1
        
        # Verify results
        assert frames_processed == 30, f"Expected 30 frames, got {frames_processed}"
        
        mean_error = np.mean(reconstruction_errors)
        print(f"\nIntegration test results:")
        print(f"  Frames processed: {frames_processed}")
        print(f"  Mean reconstruction error: {mean_error:.4f} m")
        print(f"  Max reconstruction error: {np.max(reconstruction_errors):.4f} m")
        
        # Error should be < 5cm with 2-pixel noise
        assert mean_error < 0.05, f"Reconstruction error too high: {mean_error}"


class TestCalibrationIntegration:
    """Test calibration data flow."""
    
    @requires_pydantic
    def test_calibration_data_format(self):
        """Test calibration data can be saved and loaded."""
        from voxelvr.config import CameraIntrinsics, CameraExtrinsics
        import tempfile
        from pathlib import Path
        from datetime import datetime
        
        # Create synthetic calibration data with correct schema
        intrinsics = CameraIntrinsics(
            camera_id=0,
            camera_name="Test Camera",
            resolution=(1280, 720),
            camera_matrix=[[800, 0, 640], [0, 800, 360], [0, 0, 1]],
            distortion_coeffs=[0.1, -0.2, 0, 0, 0],
            reprojection_error=0.5,
            calibration_date=datetime.now().isoformat(),
        )
        
        extrinsics = CameraExtrinsics(
            camera_id=0,
            camera_name="Test Camera",
            transform_matrix=np.eye(4).tolist(),
            rotation_matrix=np.eye(3).tolist(),
            translation_vector=[2.0, 1.5, 3.0],
            reprojection_error=0.5,
            calibration_date=datetime.now().isoformat(),
        )
        
        # Save to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            intrinsics_path = Path(tmpdir) / "intrinsics.json"
            extrinsics_path = Path(tmpdir) / "extrinsics.json"
            
            intrinsics.save(intrinsics_path)
            extrinsics.save(extrinsics_path)
            
            # Load back
            loaded_intrinsics = CameraIntrinsics.load(intrinsics_path)
            
            assert loaded_intrinsics.camera_id == 0
            assert loaded_intrinsics.resolution == (1280, 720)


class TestOSCIntegration:
    """Test OSC output integration."""
    
    @requires_osc
    def test_osc_message_format(self):
        """Test OSC messages are correctly formatted."""
        from voxelvr.transport.osc_sender import (
            pose_to_trackers_with_rotations,
            VRChatTracker,
            VRCHAT_TRACKER_IDS,
        )
        
        pose = generate_t_pose()
        confidences = np.ones(17) * 0.9
        valid_mask = np.ones(17, dtype=bool)
        
        trackers = pose_to_trackers_with_rotations(
            pose, confidences, valid_mask
        )
        
        # Should have all 8 trackers
        for name in VRCHAT_TRACKER_IDS.keys():
            assert name in trackers, f"Missing tracker: {name}"
            
            tracker = trackers[name]
            assert isinstance(tracker, VRChatTracker)
            assert len(tracker.position) == 3
            assert len(tracker.rotation) == 3
            assert 0 <= tracker.confidence <= 1
    
    @requires_osc
    def test_osc_sender_creation(self):
        """Test OSC sender can be created without errors."""
        from voxelvr.transport import OSCSender
        
        sender = OSCSender(ip="127.0.0.1", port=9000, send_rate=60.0)
        
        assert sender.ip == "127.0.0.1"
        assert sender.port == 9000
        assert sender.send_rate == 60.0


class TestCoordinateTransformIntegration:
    """Test coordinate transformation integration."""
    
    @requires_osc
    def test_transform_preserves_structure(self):
        """Test coordinate transform preserves skeleton structure."""
        from voxelvr.transport.coordinate import (
            create_default_transform,
            transform_pose_to_vrchat,
        )
        
        pose = generate_t_pose()
        transform = create_default_transform()
        
        transformed, _ = transform_pose_to_vrchat(pose, transform)
        
        # Should have same shape
        assert transformed.shape == pose.shape
        
        # Bone lengths should be preserved
        def bone_length(p, i, j):
            return np.linalg.norm(p[i] - p[j])
        
        # Check a few bones
        bones = [(5, 7), (7, 9), (11, 13), (13, 15)]  # arm and leg
        
        for i, j in bones:
            original_length = bone_length(pose, i, j)
            transformed_length = bone_length(transformed, i, j)
            
            np.testing.assert_allclose(
                original_length, transformed_length, rtol=0.01,
                err_msg=f"Bone {i}-{j} length changed"
            )
    
    @requires_osc
    def test_transform_y_up(self):
        """Test transformed coordinates conversion for VRChat."""
        from voxelvr.transport.coordinate import (
            CoordinateTransform,
            transform_pose_to_vrchat,
        )
        
        # Create pose with Y-up convention (our synthetic data)
        pose = generate_t_pose()
        
        # Identity transform (no flip) since our input is already Y-up
        identity_transform = CoordinateTransform(rotation=np.eye(3))
        
        transformed, _ = transform_pose_to_vrchat(pose, identity_transform)
        
        # Head should be higher than feet (in Y) - same as input
        head_y = transformed[0, 1]  # nose
        feet_y = (transformed[15, 1] + transformed[16, 1]) / 2  # ankles
        
        assert head_y > feet_y, f"Head ({head_y}) should be above feet ({feet_y})"
        
        # Also verify the pose structure is preserved
        assert transformed.shape == pose.shape
        np.testing.assert_array_almost_equal(transformed, pose)
