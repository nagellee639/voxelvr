"""
Sanity Check Tests

Comprehensive real-world validation tests that output human-readable data.
These tests simulate realistic scenarios and verify the system produces
sensible outputs for human poses.

Run with: pytest tests/test_sanity.py -v -s
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time


# ============================================================================
# Test Data: Realistic Human Poses
# ============================================================================

def generate_realistic_standing_pose(height: float = 1.75) -> np.ndarray:
    """Generate a realistic standing human pose scaled to given height."""
    # Proportions based on average adult human (head is ~1/8 of height)
    scale = height / 1.75
    
    return np.array([
        [0.0, 1.70, 0.0],      # 0: nose
        [-0.03, 1.73, -0.02],  # 1: left_eye
        [0.03, 1.73, -0.02],   # 2: right_eye
        [-0.08, 1.70, 0.02],   # 3: left_ear
        [0.08, 1.70, 0.02],    # 4: right_ear
        [-0.18, 1.45, 0.0],    # 5: left_shoulder
        [0.18, 1.45, 0.0],     # 6: right_shoulder
        [-0.22, 1.15, 0.05],   # 7: left_elbow (slightly bent)
        [0.22, 1.15, 0.05],    # 8: right_elbow
        [-0.20, 0.90, 0.10],   # 9: left_wrist
        [0.20, 0.90, 0.10],    # 10: right_wrist
        [-0.12, 0.95, 0.0],    # 11: left_hip
        [0.12, 0.95, 0.0],     # 12: right_hip
        [-0.12, 0.50, 0.02],   # 13: left_knee
        [0.12, 0.50, 0.02],    # 14: right_knee
        [-0.12, 0.05, 0.0],    # 15: left_ankle
        [0.12, 0.05, 0.0],     # 16: right_ankle
    ], dtype=np.float32) * scale


def generate_walking_mid_stride() -> np.ndarray:
    """Generate a mid-stride walking pose."""
    base = generate_realistic_standing_pose()
    
    # Right leg forward, left leg back
    base[13] += [0.0, -0.05, 0.25]   # left knee back
    base[15] += [0.0, 0.0, 0.30]     # left ankle back
    base[14] += [0.0, 0.05, -0.20]   # right knee forward
    base[16] += [0.0, 0.02, -0.25]   # right ankle forward
    
    # Opposite arm swing
    base[7] += [0.0, 0.0, -0.15]     # left elbow forward
    base[9] += [0.0, 0.0, -0.20]     # left wrist forward
    base[8] += [0.0, 0.0, 0.10]      # right elbow back
    base[10] += [0.0, 0.0, 0.15]     # right wrist back
    
    return base


def generate_sitting_pose() -> np.ndarray:
    """Generate a sitting pose (like in a chair)."""
    base = generate_realistic_standing_pose()
    
    # Lower upper body
    base[0:11, 1] -= 0.40  # Everything above hips goes down
    
    # Bend at hips - knees come forward
    base[13] = [-0.12, 0.45, 0.40]   # left knee forward
    base[14] = [0.12, 0.45, 0.40]    # right knee forward
    base[15] = [-0.12, 0.05, 0.50]   # left ankle
    base[16] = [0.12, 0.05, 0.50]    # right ankle
    
    return base


def generate_arms_raised_pose() -> np.ndarray:
    """Generate a pose with arms raised above head."""
    base = generate_realistic_standing_pose()
    
    # Raise arms
    base[7] = [-0.20, 1.70, 0.0]     # left elbow up
    base[8] = [0.20, 1.70, 0.0]      # right elbow up
    base[9] = [-0.15, 1.95, -0.05]   # left wrist above head
    base[10] = [0.15, 1.95, -0.05]   # right wrist above head
    
    return base


# ============================================================================
# Camera Configuration Generators
# ============================================================================

def generate_camera_setup(
    num_cameras: int,
    radius: float = 2.5,
    heights: List[float] = None,
    fov_degrees: float = 60.0,
    resolution: Tuple[int, int] = (1280, 720),
) -> List[Dict]:
    """Generate realistic camera configurations."""
    if heights is None:
        heights = [1.3] * num_cameras  # Chest height
    
    cameras = []
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        height = heights[i] if i < len(heights) else heights[0]
        
        position = np.array([x, height, z])
        look_at = np.array([0.0, 1.0, 0.0])  # Look at hip level
        
        # Generate intrinsics
        w, h = resolution
        fx = w / (2 * np.tan(np.radians(fov_degrees) / 2))
        fy = fx
        K = np.array([
            [fx, 0, w/2],
            [0, fy, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Generate extrinsics (camera-to-world)
        z_axis = look_at - position
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(z_axis, world_up)) > 0.99:
            world_up = np.array([0.0, 0.0, 1.0])
        
        x_axis = np.cross(z_axis, world_up)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        R = np.column_stack([x_axis, y_axis, z_axis])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        cameras.append({
            'id': i,
            'position': position,
            'intrinsics': K,
            'extrinsics': T,
            'resolution': resolution,
            'fov': fov_degrees,
        })
    
    return cameras


def project_pose_to_camera(
    pose_3d: np.ndarray,
    camera: Dict,
    noise_pixels: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D pose to 2D camera coordinates."""
    K = camera['intrinsics']
    T_cam_to_world = camera['extrinsics']
    T_world_to_cam = np.linalg.inv(T_cam_to_world)
    
    R = T_world_to_cam[:3, :3]
    t = T_world_to_cam[:3, 3]
    
    # Transform to camera space
    points_cam = (R @ pose_3d.T).T + t
    
    # Check visibility (in front of camera and within FOV)
    visibility = points_cam[:, 2] > 0.1
    
    # Project to image
    w, h = camera['resolution']
    points_2d = np.zeros((len(pose_3d), 2))
    
    for i, (pc, vis) in enumerate(zip(points_cam, visibility)):
        if vis and pc[2] > 0:
            projected = K @ pc
            x, y = projected[:2] / projected[2]
            
            # Check if within image bounds
            if 0 <= x <= w and 0 <= y <= h:
                points_2d[i] = [x, y]
                if noise_pixels > 0:
                    points_2d[i] += np.random.normal(0, noise_pixels, 2)
            else:
                visibility[i] = False
    
    return points_2d, visibility


# ============================================================================
# Sanity Check Tests
# ============================================================================

class TestPoseAnatomySanity:
    """Verify reconstructed poses have valid human anatomy."""
    
    @pytest.fixture
    def triangulation_pipeline(self):
        """Setup triangulation with 4 cameras."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        
        cameras = generate_camera_setup(num_cameras=4)
        
        projection_matrices = {}
        for cam in cameras:
            K = cam['intrinsics']
            T = np.linalg.inv(cam['extrinsics'])
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
        
        return TriangulationPipeline(projection_matrices), cameras
    
    def test_bone_lengths_are_realistic(self, triangulation_pipeline):
        """Verify bone lengths match human proportions."""
        from dataclasses import dataclass
        
        pipeline, cameras = triangulation_pipeline
        
        # Expected bone lengths for average adult (in meters)
        EXPECTED_BONES = {
            'shoulder_width': (0.30, 0.50),      # 30-50cm
            'upper_arm': (0.25, 0.40),           # 25-40cm
            'forearm': (0.20, 0.35),             # 20-35cm
            'hip_width': (0.15, 0.35),           # 15-35cm
            'thigh': (0.35, 0.55),               # 35-55cm
            'shin': (0.35, 0.55),                # 35-55cm (knee to ankle)
            'torso': (0.40, 0.60),               # Hip to shoulder
        }
        
        pose = generate_realistic_standing_pose()
        
        # Create mock keypoints
        @dataclass
        class MockKp:
            positions: np.ndarray
            confidences: np.ndarray
            camera_id: int
            image_width: int = 1280
            image_height: int = 720
        
        keypoints_list = []
        for cam in cameras:
            p2d, vis = project_pose_to_camera(pose, cam, noise_pixels=2.0)
            keypoints_list.append(MockKp(
                positions=p2d,
                confidences=vis.astype(np.float32) * 0.9,
                camera_id=cam['id'],
            ))
        
        result = pipeline.process(keypoints_list)
        assert result is not None, "Triangulation failed"
        
        reconstructed = result['positions']
        
        # Calculate bone lengths
        def bone_length(i, j):
            return np.linalg.norm(reconstructed[i] - reconstructed[j])
        
        bones = {
            'shoulder_width': bone_length(5, 6),
            'upper_arm_L': bone_length(5, 7),
            'upper_arm_R': bone_length(6, 8),
            'forearm_L': bone_length(7, 9),
            'forearm_R': bone_length(8, 10),
            'hip_width': bone_length(11, 12),
            'thigh_L': bone_length(11, 13),
            'thigh_R': bone_length(12, 14),
            'shin_L': bone_length(13, 15),
            'shin_R': bone_length(14, 16),
            'torso_L': bone_length(5, 11),
            'torso_R': bone_length(6, 12),
        }
        
        print("\n" + "="*60)
        print("BONE LENGTH SANITY CHECK")
        print("="*60)
        print(f"{'Bone':<20} {'Length (cm)':>12} {'Expected':>15} {'Status':>10}")
        print("-"*60)
        
        all_valid = True
        for name, length in bones.items():
            # Find expected range
            for key, (min_val, max_val) in EXPECTED_BONES.items():
                if key in name.lower().replace('_l', '').replace('_r', ''):
                    expected = f"{min_val*100:.0f}-{max_val*100:.0f}"
                    status = "✓ OK" if min_val <= length <= max_val else "✗ FAIL"
                    if status == "✗ FAIL":
                        all_valid = False
                    print(f"{name:<20} {length*100:>12.1f} {expected:>15} {status:>10}")
                    break
        
        print("="*60)
        
        # Calculate overall body height
        head_y = np.mean([reconstructed[0, 1], reconstructed[1, 1], reconstructed[2, 1]])
        feet_y = np.mean([reconstructed[15, 1], reconstructed[16, 1]])
        body_height = head_y - feet_y
        
        print(f"\nTotal body height: {body_height*100:.1f} cm")
        print(f"Expected: ~175 cm (input was 1.75m)")
        
        assert 1.5 < body_height < 2.0, f"Body height unrealistic: {body_height}m"
        assert all_valid, "Some bone lengths are outside realistic range"
    
    def test_joint_positions_are_anatomically_correct(self, triangulation_pipeline):
        """Verify joints are in correct relative positions."""
        from dataclasses import dataclass
        
        pipeline, cameras = triangulation_pipeline
        pose = generate_realistic_standing_pose()
        
        @dataclass
        class MockKp:
            positions: np.ndarray
            confidences: np.ndarray
            camera_id: int
            image_width: int = 1280
            image_height: int = 720
        
        keypoints_list = []
        for cam in cameras:
            p2d, vis = project_pose_to_camera(pose, cam, noise_pixels=1.0)
            keypoints_list.append(MockKp(
                positions=p2d,
                confidences=vis.astype(np.float32) * 0.9,
                camera_id=cam['id'],
            ))
        
        result = pipeline.process(keypoints_list)
        p = result['positions']
        
        print("\n" + "="*60)
        print("ANATOMICAL POSITION CHECK")
        print("="*60)
        
        checks = []
        
        # Head above shoulders
        head_y = p[0, 1]
        shoulder_y = (p[5, 1] + p[6, 1]) / 2
        check = head_y > shoulder_y
        checks.append(check)
        print(f"Head above shoulders: {check} (head={head_y:.2f}, shoulders={shoulder_y:.2f})")
        
        # Shoulders above hips
        hip_y = (p[11, 1] + p[12, 1]) / 2
        check = shoulder_y > hip_y
        checks.append(check)
        print(f"Shoulders above hips: {check} (shoulders={shoulder_y:.2f}, hips={hip_y:.2f})")
        
        # Hips above knees
        knee_y = (p[13, 1] + p[14, 1]) / 2
        check = hip_y > knee_y
        checks.append(check)
        print(f"Hips above knees: {check} (hips={hip_y:.2f}, knees={knee_y:.2f})")
        
        # Knees above ankles
        ankle_y = (p[15, 1] + p[16, 1]) / 2
        check = knee_y > ankle_y
        checks.append(check)
        print(f"Knees above ankles: {check} (knees={knee_y:.2f}, ankles={ankle_y:.2f})")
        
        # Left/right symmetry (roughly)
        lr_diff = abs(p[5, 0]) - abs(p[6, 0])  # Shoulders should be symmetric
        check = abs(lr_diff) < 0.1
        checks.append(check)
        print(f"Shoulder symmetry: {check} (diff={lr_diff:.3f})")
        
        print("="*60)
        
        assert all(checks), "Anatomical positions incorrect"


class TestRealWorldScenarios:
    """Test system with realistic camera setups and conditions."""
    
    def test_different_camera_configurations(self):
        """Test various camera placement configurations."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        from dataclasses import dataclass
        
        @dataclass
        class MockKp:
            positions: np.ndarray
            confidences: np.ndarray
            camera_id: int
            image_width: int = 1280
            image_height: int = 720
        
        configurations = [
            {"name": "3 cameras, ring, 2.5m radius", "num": 3, "radius": 2.5, "heights": [1.3, 1.3, 1.3]},
            {"name": "4 cameras, ring, 3m radius", "num": 4, "radius": 3.0, "heights": [1.3, 1.3, 1.3, 1.3]},
            {"name": "4 cameras, mixed heights", "num": 4, "radius": 2.5, "heights": [1.0, 1.5, 1.0, 1.5]},
            {"name": "3 cameras, close, 1.5m radius", "num": 3, "radius": 1.5, "heights": [1.2, 1.2, 1.2]},
            {"name": "4 cameras, far, 4m radius", "num": 4, "radius": 4.0, "heights": [1.3, 1.3, 1.3, 1.3]},
        ]
        
        pose = generate_realistic_standing_pose()
        
        print("\n" + "="*70)
        print("CAMERA CONFIGURATION TEST")
        print("="*70)
        print(f"{'Configuration':<35} {'Valid Joints':>12} {'Mean Error (cm)':>15} {'Status':>8}")
        print("-"*70)
        
        results = []
        
        for config in configurations:
            cameras = generate_camera_setup(
                num_cameras=config["num"],
                radius=config["radius"],
                heights=config["heights"],
            )
            
            projection_matrices = {}
            for cam in cameras:
                K = cam['intrinsics']
                T = np.linalg.inv(cam['extrinsics'])
                R, t = T[:3, :3], T[:3, 3]
                P = K @ np.hstack([R, t.reshape(3, 1)])
                projection_matrices[cam['id']] = P
            
            pipeline = TriangulationPipeline(projection_matrices)
            
            keypoints_list = []
            for cam in cameras:
                p2d, vis = project_pose_to_camera(pose, cam, noise_pixels=2.0)
                keypoints_list.append(MockKp(
                    positions=p2d,
                    confidences=vis.astype(np.float32) * 0.9,
                    camera_id=cam['id'],
                ))
            
            result = pipeline.process(keypoints_list)
            
            if result is not None:
                valid_count = np.sum(result['valid'])
                errors = []
                for i in range(17):
                    if result['valid'][i]:
                        errors.append(np.linalg.norm(result['positions'][i] - pose[i]))
                mean_error = np.mean(errors) * 100 if errors else float('inf')
                status = "✓" if valid_count >= 12 and mean_error < 5 else "!"
            else:
                valid_count = 0
                mean_error = float('inf')
                status = "✗"
            
            print(f"{config['name']:<35} {valid_count:>12} {mean_error:>15.2f} {status:>8}")
            results.append((config['name'], valid_count, mean_error))
        
        print("="*70)
        
        # At least most configurations should work
        successful = sum(1 for _, valid, err in results if valid >= 12 and err < 10)
        assert successful >= 3, f"Too many configurations failed: {successful}/5"
    
    def test_noise_robustness(self):
        """Test system handles different noise levels."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        from dataclasses import dataclass
        
        @dataclass
        class MockKp:
            positions: np.ndarray
            confidences: np.ndarray
            camera_id: int
            image_width: int = 1280
            image_height: int = 720
        
        cameras = generate_camera_setup(num_cameras=4)
        
        projection_matrices = {}
        for cam in cameras:
            K = cam['intrinsics']
            T = np.linalg.inv(cam['extrinsics'])
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
        
        pipeline = TriangulationPipeline(projection_matrices)
        pose = generate_realistic_standing_pose()
        
        noise_levels = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
        
        print("\n" + "="*60)
        print("NOISE ROBUSTNESS TEST")
        print("="*60)
        print(f"{'Noise (px)':<15} {'Valid Joints':>12} {'Mean Error (cm)':>15} {'Status':>10}")
        print("-"*60)
        
        results = []
        
        for noise in noise_levels:
            errors = []
            
            # Run multiple trials
            for trial in range(5):
                keypoints_list = []
                for cam in cameras:
                    p2d, vis = project_pose_to_camera(pose, cam, noise_pixels=noise)
                    keypoints_list.append(MockKp(
                        positions=p2d,
                        confidences=vis.astype(np.float32) * 0.9,
                        camera_id=cam['id'],
                    ))
                
                result = pipeline.process(keypoints_list)
                if result is not None:
                    for i in range(17):
                        if result['valid'][i]:
                            errors.append(np.linalg.norm(result['positions'][i] - pose[i]))
            
            mean_error = np.mean(errors) * 100 if errors else float('inf')
            valid_count = len(errors) // 5  # Average per trial
            
            # Expected: low noise = low error, high noise = higher error
            if noise <= 2:
                expected = "< 3cm"
                status = "✓" if mean_error < 3 else "✗"
            elif noise <= 5:
                expected = "< 5cm"
                status = "✓" if mean_error < 5 else "!"
            elif noise <= 10:
                expected = "< 10cm"
                status = "✓" if mean_error < 10 else "!"
            else:
                expected = "variable"
                status = "~"
            
            print(f"{noise:<15.1f} {valid_count:>12} {mean_error:>15.2f} {status:>10}")
            results.append((noise, mean_error))
        
        print("="*60)
        
        # Error should increase with noise
        assert results[0][1] < results[-1][1], "Error should increase with noise"
    
    def test_different_poses(self):
        """Test system handles various human poses."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        from dataclasses import dataclass
        
        @dataclass
        class MockKp:
            positions: np.ndarray
            confidences: np.ndarray
            camera_id: int
            image_width: int = 1280
            image_height: int = 720
        
        cameras = generate_camera_setup(num_cameras=4)
        
        projection_matrices = {}
        for cam in cameras:
            K = cam['intrinsics']
            T = np.linalg.inv(cam['extrinsics'])
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
        
        pipeline = TriangulationPipeline(projection_matrices)
        
        poses = {
            "Standing": generate_realistic_standing_pose(),
            "Walking": generate_walking_mid_stride(),
            "Sitting": generate_sitting_pose(),
            "Arms Raised": generate_arms_raised_pose(),
        }
        
        print("\n" + "="*60)
        print("DIFFERENT POSES TEST")
        print("="*60)
        print(f"{'Pose':<15} {'Valid Joints':>12} {'Mean Error (cm)':>15} {'Max Error (cm)':>15}")
        print("-"*60)
        
        for pose_name, pose in poses.items():
            keypoints_list = []
            for cam in cameras:
                p2d, vis = project_pose_to_camera(pose, cam, noise_pixels=2.0)
                keypoints_list.append(MockKp(
                    positions=p2d,
                    confidences=vis.astype(np.float32) * 0.9,
                    camera_id=cam['id'],
                ))
            
            result = pipeline.process(keypoints_list)
            
            if result is not None:
                valid_count = np.sum(result['valid'])
                errors = []
                for i in range(17):
                    if result['valid'][i]:
                        errors.append(np.linalg.norm(result['positions'][i] - pose[i]))
                mean_error = np.mean(errors) * 100 if errors else float('inf')
                max_error = np.max(errors) * 100 if errors else float('inf')
            else:
                valid_count = 0
                mean_error = max_error = float('inf')
            
            print(f"{pose_name:<15} {valid_count:>12} {mean_error:>15.2f} {max_error:>15.2f}")
        
        print("="*60)


class TestOutputFormats:
    """Verify output data is in correct format for VRChat."""
    
    def test_tracker_output_format(self):
        """Verify tracker output has expected format and values."""
        from voxelvr.pose.rotation import estimate_all_rotations
        from voxelvr.transport.osc_sender import pose_to_trackers_with_rotations
        
        pose = generate_realistic_standing_pose()
        confidences = np.ones(17) * 0.9
        valid_mask = np.ones(17, dtype=bool)
        
        trackers = pose_to_trackers_with_rotations(pose, confidences, valid_mask)
        
        print("\n" + "="*70)
        print("TRACKER OUTPUT FORMAT CHECK")
        print("="*70)
        print(f"{'Tracker':<15} {'Position (x,y,z)':<25} {'Rotation (x,y,z)':<25} {'Conf':>6}")
        print("-"*70)
        
        expected_trackers = ['hip', 'chest', 'left_foot', 'right_foot', 
                            'left_knee', 'right_knee', 'left_elbow', 'right_elbow']
        
        for name in expected_trackers:
            if name in trackers:
                t = trackers[name]
                pos_str = f"({t.position[0]:.2f}, {t.position[1]:.2f}, {t.position[2]:.2f})"
                rot_str = f"({t.rotation[0]:.1f}, {t.rotation[1]:.1f}, {t.rotation[2]:.1f})"
                print(f"{name:<15} {pos_str:<25} {rot_str:<25} {t.confidence:>6.2f}")
                
                # Sanity checks
                assert len(t.position) == 3, f"Position should have 3 values"
                assert len(t.rotation) == 3, f"Rotation should have 3 values"
                assert 0 <= t.confidence <= 1, f"Confidence should be 0-1"
                
                # Rotation should be in degrees (-180 to 180 or 0 to 360)
                for r in t.rotation:
                    assert -360 <= r <= 360, f"Rotation out of range: {r}"
        
        print("="*70)
        
        # All 8 trackers should be present
        assert len(trackers) == 8, f"Expected 8 trackers, got {len(trackers)}"
    
    def test_position_units_are_meters(self):
        """Verify positions are in meters (VRChat expects meters)."""
        from voxelvr.transport.osc_sender import pose_to_trackers_with_rotations
        
        pose = generate_realistic_standing_pose(height=1.75)  # 1.75m tall person
        confidences = np.ones(17) * 0.9
        valid_mask = np.ones(17, dtype=bool)
        
        trackers = pose_to_trackers_with_rotations(pose, confidences, valid_mask)
        
        print("\n" + "="*60)
        print("POSITION UNITS CHECK (should be meters)")
        print("="*60)
        
        # Hip should be around 0.95m high
        hip = trackers['hip']
        print(f"Hip Y position: {hip.position[1]:.3f} m (expected ~0.95m)")
        assert 0.5 < hip.position[1] < 1.5, f"Hip position unrealistic: {hip.position[1]}m"
        
        # Chest should be around 1.45m high
        chest = trackers['chest']
        print(f"Chest Y position: {chest.position[1]:.3f} m (expected ~1.45m)")
        assert 1.0 < chest.position[1] < 2.0, f"Chest position unrealistic: {chest.position[1]}m"
        
        # Feet should be near 0
        left_foot = trackers['left_foot']
        print(f"Left foot Y position: {left_foot.position[1]:.3f} m (expected ~0.05m)")
        assert -0.2 < left_foot.position[1] < 0.5, f"Foot position unrealistic: {left_foot.position[1]}m"
        
        print("="*60)
        print("✓ All positions are in expected meter range")


class TestTemporalStability:
    """Test stability over time."""
    
    def test_jitter_free_with_filter(self):
        """Verify filters reduce jitter effectively."""
        from voxelvr.pose import PoseFilter
        
        pose_filter = PoseFilter(num_joints=17)
        
        # Simulate 2 seconds at 30 FPS with noisy input
        num_frames = 60
        base_pose = generate_realistic_standing_pose()
        
        raw_positions = []
        filtered_positions = []
        
        print("\n" + "="*60)
        print("TEMPORAL JITTER TEST")
        print("="*60)
        
        for i in range(num_frames):
            t = i / 30.0
            
            # Add random noise to simulate measurement error
            noisy_pose = base_pose + np.random.normal(0, 0.02, base_pose.shape)  # 2cm noise
            
            valid_mask = np.ones(17, dtype=bool)
            filtered_pose = pose_filter.filter(noisy_pose, valid_mask, timestamp=t)
            
            raw_positions.append(noisy_pose.copy())
            filtered_positions.append(filtered_pose.copy())
        
        raw_positions = np.array(raw_positions)
        filtered_positions = np.array(filtered_positions)
        
        # Calculate frame-to-frame jitter
        raw_jitter = []
        filtered_jitter = []
        
        for i in range(1, num_frames):
            raw_jitter.append(np.mean(np.linalg.norm(
                raw_positions[i] - raw_positions[i-1], axis=1
            )))
            filtered_jitter.append(np.mean(np.linalg.norm(
                filtered_positions[i] - filtered_positions[i-1], axis=1
            )))
        
        raw_jitter_mean = np.mean(raw_jitter) * 100
        raw_jitter_std = np.std(raw_jitter) * 100
        filtered_jitter_mean = np.mean(filtered_jitter[20:]) * 100  # Skip warmup
        filtered_jitter_std = np.std(filtered_jitter[20:]) * 100
        
        reduction = (1 - filtered_jitter_mean / raw_jitter_mean) * 100
        
        print(f"Raw input jitter:      {raw_jitter_mean:.2f} ± {raw_jitter_std:.2f} cm/frame")
        print(f"Filtered jitter:       {filtered_jitter_mean:.2f} ± {filtered_jitter_std:.2f} cm/frame")
        print(f"Jitter reduction:      {reduction:.1f}%")
        print("="*60)
        
        assert reduction > 50, f"Filter should reduce jitter by >50%, got {reduction:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
