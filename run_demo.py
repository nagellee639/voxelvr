#!/usr/bin/env python3
"""
VoxelVR Demo Visualizer

Standalone demo for testing the tracking pipeline without VRChat.
Shows a 3D skeleton visualization and multi-camera dashboard.

Usage:
    python run_demo.py
    
Controls:
    Q - Quit
    R - Start/stop recording
    F - Toggle filter (One-Euro)
    S - Save screenshot
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from voxelvr.config import VoxelVRConfig, CameraConfig, MultiCameraCalibration
from voxelvr.capture import Camera, CameraManager
from voxelvr.pose import PoseDetector2D, PoseFilter
from voxelvr.pose.triangulation import TriangulationPipeline, compute_projection_matrices
from voxelvr.demo.dashboard import TrackingDashboard, PerformanceMetrics
from voxelvr.demo.visualizer import SkeletonVisualizer, SimpleOpenCVVisualizer


def main():
    parser = argparse.ArgumentParser(description="VoxelVR Demo Visualizer")
    parser.add_argument(
        "--cameras", "-c",
        type=int,
        nargs="+",
        help="Camera IDs to use (auto-detect if not specified)"
    )
    parser.add_argument(
        "--calibration", "-C",
        type=Path,
        default=None,
        help="Path to calibration.json file"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable temporal filtering"
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Skip 3D visualization (dashboard only)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic test data (no cameras needed)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = VoxelVRConfig.load()
    
    # Check for synthetic mode
    if args.synthetic:
        return run_synthetic_demo(config, args)
    
    # Load calibration
    calibration_path = args.calibration or (config.calibration_dir / "calibration.json")
    
    if not calibration_path.exists():
        print(f"Calibration not found: {calibration_path}")
        print("Run 'python run_calibration.py' first, or use --synthetic for testing.")
        return 1
    
    print(f"Loading calibration: {calibration_path}")
    calibration = MultiCameraCalibration.load(calibration_path)
    
    # Get camera IDs from calibration
    camera_ids = args.cameras or list(calibration.cameras.keys())
    print(f"Using cameras: {camera_ids}")
    
    if len(camera_ids) < 2:
        print("Need at least 2 cameras for triangulation!")
        return 1
    
    # Create camera manager
    camera_configs = [
        CameraConfig(id=cam_id, resolution=(1280, 720), fps=30)
        for cam_id in camera_ids
    ]
    
    camera_manager = CameraManager(camera_configs)
    camera_manager.load_calibration(calibration)
    
    # Create pose detector
    print("Loading pose detection model...")
    detector = PoseDetector2D(confidence_threshold=0.3)
    if not detector.load_model():
        print("Failed to load pose model!")
        return 1
    
    # Build projection matrices
    intrinsics_list = []
    extrinsics_list = []
    
    from voxelvr.config import CameraIntrinsics, CameraExtrinsics
    
    for cam_id, data in calibration.cameras.items():
        if cam_id in camera_ids:
            intrinsics_list.append(CameraIntrinsics(**data['intrinsics']))
            extrinsics_list.append(CameraExtrinsics(**data['extrinsics']))
    
    projection_matrices = compute_projection_matrices(intrinsics_list, extrinsics_list)
    
    # Create triangulation pipeline
    triangulation = TriangulationPipeline(
        projection_matrices,
        confidence_threshold=0.3,
    )
    
    # Create pose filter
    pose_filter = PoseFilter(
        num_joints=17,
        min_cutoff=config.tracking.filter_min_cutoff,
        beta=config.tracking.filter_beta,
    ) if not args.no_filter else None
    
    # Create visualizers
    dashboard = TrackingDashboard(camera_ids)
    
    skeleton_viz = None
    if not args.no_3d:
        try:
            skeleton_viz = SkeletonVisualizer()
            skeleton_viz.initialize()
            skeleton_viz.show_non_blocking()
        except Exception as e:
            print(f"3D visualization not available: {e}")
            skeleton_viz = SimpleOpenCVVisualizer()
    
    # Start cameras
    print("Starting cameras...")
    if not camera_manager.start_all():
        print("Failed to start cameras!")
        return 1
    
    print("\nDemo running! Press Q to quit.")
    print("R: Record | F: Toggle filter | S: Screenshot")
    
    # Main loop
    try:
        frame_count = 0
        start_time = time.time()
        filter_enabled = not args.no_filter
        
        while True:
            loop_start = time.time()
            
            # Capture frames
            frames_raw = camera_manager.get_synchronized_frames()
            if not frames_raw:
                continue
            
            frames = {cam_id: f.image for cam_id, f in frames_raw.items()}
            capture_time = time.time() - loop_start
            
            # 2D pose detection
            detect_start = time.time()
            keypoints_2d = {}
            for cam_id, frame in frames.items():
                kp = detector.detect(frame, camera_id=cam_id)
                if kp:
                    keypoints_2d[cam_id] = kp
            detect_time = time.time() - detect_start
            
            # 3D triangulation
            triang_start = time.time()
            pose_3d = None
            
            if len(keypoints_2d) >= 2:
                kp_list = list(keypoints_2d.values())
                pose_3d = triangulation.process(kp_list)
            
            triang_time = time.time() - triang_start
            
            # Apply temporal filter
            if pose_3d and pose_filter and filter_enabled:
                pose_3d['positions'] = pose_filter.filter(
                    pose_3d['positions'],
                    pose_3d['valid'],
                )
            
            # Calculate metrics
            total_time = time.time() - loop_start
            
            metrics = PerformanceMetrics(
                capture_fps=1.0 / capture_time if capture_time > 0 else 0,
                detection_fps=1.0 / detect_time if detect_time > 0 else 0,
                triangulation_fps=1.0 / triang_time if triang_time > 0 else 0,
                total_fps=1.0 / total_time if total_time > 0 else 0,
                capture_latency_ms=capture_time * 1000,
                detection_latency_ms=detect_time * 1000,
                triangulation_latency_ms=triang_time * 1000,
                total_latency_ms=total_time * 1000,
                num_valid_joints=np.sum(pose_3d['valid']) if pose_3d else 0,
                avg_confidence=np.mean(pose_3d['confidences'][pose_3d['valid']]) if pose_3d and np.any(pose_3d['valid']) else 0,
            )
            
            # Update 3D skeleton
            if skeleton_viz and pose_3d and np.any(pose_3d['valid']):
                if isinstance(skeleton_viz, SkeletonVisualizer):
                    skeleton_viz.update(pose_3d['positions'], pose_3d['valid'])
                else:
                    skeleton_viz.show(pose_3d['positions'], pose_3d['valid'])
            
            # Update dashboard
            key = dashboard.show(frames, keypoints_2d, metrics)
            
            # Handle input
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                if dashboard.is_recording:
                    dashboard.stop_recording()
                else:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    dashboard.start_recording(
                        str(config.recordings_dir / f"demo_{timestamp}.mp4")
                    )
            elif key == ord('f') or key == ord('F'):
                filter_enabled = not filter_enabled
                print(f"Filter: {'ON' if filter_enabled else 'OFF'}")
                if pose_filter:
                    pose_filter.reset()
            elif key == ord('s') or key == ord('S'):
                import cv2
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(config.recordings_dir / f"screenshot_{timestamp}.png"),
                           dashboard.update(frames, keypoints_2d, metrics))
                print(f"Screenshot saved!")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("Shutting down...")
        camera_manager.stop_all()
        dashboard.close()
        if skeleton_viz and isinstance(skeleton_viz, SkeletonVisualizer):
            skeleton_viz.close()
    
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
    
    return 0


def run_synthetic_demo(config: VoxelVRConfig, args):
    """Run demo with synthetic skeleton data (no cameras needed)."""
    print("Running synthetic demo mode...")
    
    import cv2
    
    # Create a simple animated skeleton
    skeleton_viz = SimpleOpenCVVisualizer(window_size=(800, 600))
    
    t = 0
    print("Press Q to quit synthetic demo.")
    
    while True:
        t += 0.03
        
        # Generate synthetic skeleton (simple walking animation)
        positions = generate_synthetic_pose(t)
        valid_mask = np.ones(17, dtype=bool)
        
        # Draw
        img = skeleton_viz.update(positions, valid_mask)
        
        # Add synthetic label
        cv2.putText(img, "SYNTHETIC DATA", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        cv2.imshow("VoxelVR - Synthetic Demo", img)
        
        key = cv2.waitKey(33) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    cv2.destroyAllWindows()
    return 0


def generate_synthetic_pose(t: float) -> np.ndarray:
    """Generate a simple walking pose animation."""
    # Base T-pose
    base = np.array([
        [0, 1.7, 0],      # 0: nose
        [-0.05, 1.72, 0], # 1: left_eye
        [0.05, 1.72, 0],  # 2: right_eye
        [-0.1, 1.7, 0],   # 3: left_ear
        [0.1, 1.7, 0],    # 4: right_ear
        [-0.2, 1.5, 0],   # 5: left_shoulder
        [0.2, 1.5, 0],    # 6: right_shoulder
        [-0.4, 1.3, 0],   # 7: left_elbow
        [0.4, 1.3, 0],    # 8: right_elbow
        [-0.5, 1.1, 0],   # 9: left_wrist
        [0.5, 1.1, 0],    # 10: right_wrist
        [-0.15, 1.0, 0],  # 11: left_hip
        [0.15, 1.0, 0],   # 12: right_hip
        [-0.15, 0.5, 0],  # 13: left_knee
        [0.15, 0.5, 0],   # 14: right_knee
        [-0.15, 0.05, 0], # 15: left_ankle
        [0.15, 0.05, 0],  # 16: right_ankle
    ])
    
    # Add walking animation
    positions = base.copy()
    
    # Arm swing
    arm_swing = 0.3 * np.sin(t)
    positions[7, 2] = arm_swing
    positions[9, 2] = arm_swing * 0.8
    positions[8, 2] = -arm_swing
    positions[10, 2] = -arm_swing * 0.8
    
    # Leg swing
    leg_swing = 0.2 * np.sin(t)
    positions[13, 2] = leg_swing
    positions[15, 2] = leg_swing * 1.2
    positions[14, 2] = -leg_swing
    positions[16, 2] = -leg_swing * 1.2
    
    # Hip sway
    positions[11:13, 0] += 0.02 * np.sin(t * 2)
    
    # Body bounce
    positions[:, 1] += 0.02 * abs(np.sin(t * 2))
    
    return positions


if __name__ == "__main__":
    sys.exit(main())
