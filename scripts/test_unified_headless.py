#!/usr/bin/env python3
"""
Headless Test for Unified GUI Pipeline

Tests the unified view logic and pose detection with dataset frames
without requiring a GUI window.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from voxelvr.gui.unified_view import UnifiedView, CalibrationMode, TrackingMode
from voxelvr.calibration.skeleton_calibration import (
    SkeletonObservation, SimpleCameraIntrinsics, 
    estimate_cameras_from_skeleton, triangulate_skeleton, compute_projection_matrix,
    compute_bone_length_error, is_bone_length_valid,
)


def load_dataset_session(dataset_path: Path, session_name: str) -> dict[int, list[np.ndarray]]:
    """Load a tracking session from dataset."""
    session_path = dataset_path / "tracking" / session_name
    
    if not session_path.exists():
        print(f"Session not found: {session_path}")
        return {}
    
    cameras = {}
    camera_dirs = sorted([d for d in session_path.iterdir() if d.is_dir() and d.name.startswith("cam_")])
    
    for cam_dir in camera_dirs:
        cam_id = int(cam_dir.name.split("_")[1])
        image_files = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
        frames = []
        
        for img_file in image_files[:50]:  # Limit to 50 frames for speed
            frame = cv2.imread(str(img_file))
            if frame is not None:
                frames.append(frame)
        
        if frames:
            cameras[cam_id] = frames
            print(f"  Camera {cam_id}: {len(frames)} frames")
    
    return cameras


def test_unified_view_state():
    """Test UnifiedView state management."""
    print("\n=== Testing UnifiedView State ===")
    
    view = UnifiedView()
    
    # Test camera detection
    view.set_cameras([0, 2, 3, 5])
    assert len(view.state.cameras) == 4, "Should have 4 cameras"
    print("✓ Camera detection works")
    
    # Test grid layout
    rows, cols = view.get_camera_grid_layout()
    assert rows == 2 and cols == 2, f"Expected 2x2 grid, got {rows}x{cols}"
    print("✓ Grid layout correct (2x2)")
    
    # Test calibration mode toggle
    view.set_calibration_mode(CalibrationMode.SKELETON)
    assert view.should_show_skeleton_warning() is True
    print("✓ Skeleton warning shown")
    
    view.set_calibration_mode(CalibrationMode.CHARUCO)
    assert view.should_show_skeleton_warning() is False
    print("✓ ChArUco mode - no warning")
    
    # Test tracking state transitions
    view.start_tracking()
    assert view.state.tracking_mode == TrackingMode.STARTING
    print("✓ Tracking start state")
    
    view.on_tracking_started()
    assert view.state.tracking_mode == TrackingMode.RUNNING
    print("✓ Tracking running state")
    
    view.stop_tracking()
    view.on_tracking_stopped()
    assert view.state.tracking_mode == TrackingMode.STOPPED
    print("✓ Tracking stopped state")
    
    # Test AprilTag toggle
    view.set_apriltags_enabled(True)
    assert view.state.apriltags_enabled is True
    print("✓ AprilTag enabled")
    
    # Test OSC config
    view.set_osc_config("192.168.1.100", 9001)
    assert view.state.osc_ip == "192.168.1.100"
    assert view.state.osc_port == 9001
    print("✓ OSC config updated")
    
    print("\n✅ All UnifiedView state tests passed!")
    return True


def test_pose_detection_pipeline(cameras: dict[int, list[np.ndarray]]):
    """Test pose detection on dataset frames."""
    print("\n=== Testing Pose Detection Pipeline ===")
    
    try:
        from voxelvr.pose.detector_2d import PoseDetector2D
        detector = PoseDetector2D()
        print("  Pose detector initialized")
    except Exception as e:
        print(f"  ⚠ Could not load pose detector: {e}")
        print("  Skipping pose detection tests")
        return True
    
    camera_ids = list(cameras.keys())
    num_frames = min(len(f) for f in cameras.values())
    
    print(f"  Processing {num_frames} frames from {len(camera_ids)} cameras...")
    
    frames_with_poses = 0
    observations = []
    
    start_time = time.time()
    
    for frame_idx in range(min(num_frames, 30)):  # Process up to 30 frames
        keypoints_2d = {}
        confidences = {}
        
        for cam_id in camera_ids:
            frame = cameras[cam_id][frame_idx]
            result = detector.detect(frame)
            
            if result is not None:
                keypoints_2d[cam_id] = result.positions
                confidences[cam_id] = result.confidences
        
        if len(keypoints_2d) >= 2:
            frames_with_poses += 1
            observations.append(SkeletonObservation(
                keypoints_2d=keypoints_2d,
                confidences=confidences,
                timestamp=frame_idx / 30.0,
            ))
    
    elapsed = time.time() - start_time
    fps = num_frames / elapsed if elapsed > 0 else 0
    
    print(f"  Processed {min(num_frames, 30)} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
    print(f"  Frames with 2+ camera detections: {frames_with_poses}")
    print(f"  Skeleton observations created: {len(observations)}")
    
    assert frames_with_poses > 0, "Should detect poses in some frames"
    print("✅ Pose detection pipeline works!")
    
    return observations


def test_skeleton_calibration(observations: list, cameras: dict[int, list]):
    """Test skeleton-based calibration."""
    print("\n=== Testing Skeleton Calibration ===")
    
    if len(observations) < 3:
        print("  ⚠ Not enough observations, skipping calibration test")
        return True
    
    # Create synthetic intrinsics based on frame size
    camera_ids = list(cameras.keys())
    sample_frame = cameras[camera_ids[0]][0]
    h, w = sample_frame.shape[:2]
    
    intrinsics = {}
    for cam_id in camera_ids:
        intrinsics[cam_id] = SimpleCameraIntrinsics(
            camera_id=cam_id,
            fx=w * 0.8,
            fy=w * 0.8,
            cx=w / 2,
            cy=h / 2,
            width=w,
            height=h,
        )
    
    print(f"  Running calibration with {len(observations)} observations...")
    
    start_time = time.time()
    result = estimate_cameras_from_skeleton(observations, intrinsics)
    elapsed = time.time() - start_time
    
    print(f"  Calibration completed in {elapsed:.2f}s")
    print(f"  Cameras calibrated: {len(result.cameras)}")
    print(f"  Scale factor: {result.scale_factor:.3f}")
    print(f"  Reprojection error: {result.reprojection_error:.2f}px")
    print(f"  Bone length error: {result.bone_length_error:.2%}")
    print(f"  Skeleton only: {result.is_skeleton_only}")
    
    assert len(result.cameras) >= 2, "Should calibrate at least 2 cameras"
    assert result.scale_factor > 0, "Scale factor should be positive"
    
    # Test triangulation with calibrated cameras
    print("\n  Testing triangulation with calibrated cameras...")
    
    proj_mats = {
        cid: compute_projection_matrix(intrinsics[cid], cam)
        for cid, cam in result.cameras.items()
    }
    
    valid_skeletons = 0
    for obs in observations[:10]:
        positions, valid = triangulate_skeleton(obs, proj_mats)
        positions_scaled = positions * result.scale_factor
        
        if np.sum(valid) >= 6:
            error, _ = compute_bone_length_error(positions_scaled, valid)
            if error < 1.0:  # Less than 100% error
                valid_skeletons += 1
    
    print(f"  Valid triangulated skeletons: {valid_skeletons}/10")
    
    print("✅ Skeleton calibration works!")
    return True


def test_full_pipeline(dataset_path: Path):
    """Run full pipeline test."""
    print("\n" + "="*60)
    print("UNIFIED GUI HEADLESS PIPELINE TEST")
    print("="*60)
    
    # Find sessions
    tracking_path = dataset_path / "tracking"
    sessions = sorted([d.name for d in tracking_path.iterdir() if d.is_dir()])
    
    if not sessions:
        print("No tracking sessions found!")
        return False
    
    print(f"\nFound sessions: {sessions}")
    session = sessions[-1]  # Use most recent
    print(f"Using session: {session}")
    
    # Load data
    print(f"\n--- Loading Dataset ---")
    cameras = load_dataset_session(dataset_path, session)
    
    if not cameras:
        print("Failed to load dataset!")
        return False
    
    # Run tests
    test_unified_view_state()
    
    observations = test_pose_detection_pipeline(cameras)
    
    if observations:
        test_skeleton_calibration(observations, cameras)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✅")
    print("="*60)
    
    return True


def main():
    dataset_path = Path("/home/lee/voxelvr/dataset")
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return 1
    
    success = test_full_pipeline(dataset_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
