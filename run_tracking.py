#!/usr/bin/env python3
"""
VoxelVR Full Tracking Pipeline

Runs the complete tracking pipeline and sends data to VRChat via OSC.

Usage:
    python run_tracking.py
    
Requirements:
    - Calibrated cameras (run run_calibration.py first)
    - VRChat running with OSC enabled
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from voxelvr.config import (
    VoxelVRConfig, 
    CameraConfig, 
    MultiCameraCalibration,
    CameraIntrinsics,
    CameraExtrinsics,
)
from voxelvr.capture import CameraManager
from voxelvr.pose import PoseDetector2D, PoseFilter, ConfidenceFilter
from voxelvr.pose.triangulation import TriangulationPipeline, compute_projection_matrices
from voxelvr.transport import OSCSender, CoordinateTransform
from voxelvr.transport.osc_sender import pose_to_trackers_with_rotations
from voxelvr.transport.coordinate import create_default_transform, transform_pose_to_vrchat
from voxelvr.pose.rotation import RotationFilter


def main():
    parser = argparse.ArgumentParser(description="VoxelVR Full Tracking Pipeline")
    parser.add_argument(
        "--cameras", "-c",
        type=int,
        nargs="+",
        help="Camera IDs to use"
    )
    parser.add_argument(
        "--calibration", "-C",
        type=Path,
        default=None,
        help="Path to calibration.json"
    )
    parser.add_argument(
        "--osc-ip",
        type=str,
        default="127.0.0.1",
        help="VRChat OSC IP address"
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        default=9000,
        help="VRChat OSC port"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show camera preview window"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable temporal filtering"
    )
    parser.add_argument(
        "--no-confidence-filter",
        action="store_true",
        help="Disable confidence-based view filtering"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = VoxelVRConfig.load()
    
    # Load calibration
    calibration_path = args.calibration or (config.calibration_dir / "calibration.json")
    
    if not calibration_path.exists():
        print(f"Calibration not found: {calibration_path}")
        print("Run 'python run_calibration.py' first.")
        return 1
    
    print(f"Loading calibration: {calibration_path}")
    calibration = MultiCameraCalibration.load(calibration_path)
    
    # Get camera IDs
    camera_ids = args.cameras or list(calibration.cameras.keys())
    print(f"Using cameras: {camera_ids}")
    
    if len(camera_ids) < 2:
        print("Need at least 2 cameras!")
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
    detector = PoseDetector2D(confidence_threshold=config.tracking.confidence_threshold)
    if not detector.load_model():
        print("Failed to load pose model!")
        return 1
    
    # Build projection matrices
    intrinsics_list = []
    extrinsics_list = []
    
    for cam_id, data in calibration.cameras.items():
        if cam_id in camera_ids:
            intrinsics_list.append(CameraIntrinsics(**data['intrinsics']))
            extrinsics_list.append(CameraExtrinsics(**data['extrinsics']))
    
    projection_matrices = compute_projection_matrices(intrinsics_list, extrinsics_list)
    
    # Create triangulation pipeline
    triangulation = TriangulationPipeline(projection_matrices, confidence_threshold=config.tracking.confidence_threshold)
    
    # Create confidence filter
    confidence_filter = ConfidenceFilter(
        num_joints=17,
        confidence_threshold=config.tracking.confidence_threshold,
        grace_period_frames=config.tracking.confidence_grace_period_frames,
        reactivation_frames=config.tracking.confidence_reactivation_frames,
    ) if not args.no_confidence_filter else None
    
    # Create pose filter
    pose_filter = PoseFilter(
        num_joints=17,
        min_cutoff=config.tracking.filter_min_cutoff,
        beta=config.tracking.filter_beta,
        freeze_invalid_joints=config.tracking.freeze_unconfident_joints,
    ) if not args.no_filter else None
    
    # Create rotation filter (smoother alpha for rotations)
    rotation_filter = RotationFilter(alpha=0.2) if not args.no_filter else None
    
    # Create coordinate transform
    coord_transform = create_default_transform()
    
    # Create OSC sender
    osc_sender = OSCSender(
        ip=args.osc_ip,
        port=args.osc_port,
        send_rate=60.0,
    )
    
    if not osc_sender.connect():
        print("Failed to connect OSC sender!")
        return 1
    
    print(f"OSC connected: {args.osc_ip}:{args.osc_port}")
    
    # Preview window (optional)
    preview_window = None
    if args.preview:
        import cv2
        preview_window = "VoxelVR Preview"
    
    # Start cameras
    print("Starting cameras...")
    if not camera_manager.start_all():
        print("Failed to start cameras!")
        return 1
    
    print("\n" + "="*50)
    print("TRACKING ACTIVE - Sending to VRChat")
    print("="*50)
    print("Press Ctrl+C to stop")
    print("")
    
    # Main tracking loop
    frame_count = 0
    start_time = time.time()
    last_status_time = start_time
    
    try:
        while True:
            loop_start = time.time()
            
            # Capture frames
            frames_raw = camera_manager.get_synchronized_frames()
            if not frames_raw:
                continue
            
            frames = {cam_id: f.image for cam_id, f in frames_raw.items()}
            
            # 2D pose detection
            keypoints_2d = {}
            for cam_id, frame in frames.items():
                kp = detector.detect(frame, camera_id=cam_id)
                if kp:
                    keypoints_2d[cam_id] = kp
            
            # Apply confidence filtering
            filtered_keypoints = list(keypoints_2d.values())
            conf_diagnostics = None
            if confidence_filter:
                filtered_keypoints, conf_diagnostics = confidence_filter.update(filtered_keypoints)
            
            # 3D triangulation
            pose_3d = None
            if len(filtered_keypoints) >= 2:
                pose_3d = triangulation.process(filtered_keypoints)
            
            # Apply confidence-based joint freezing
            if pose_3d and confidence_filter:
                pose_3d['positions'] = confidence_filter.apply_freezing(
                    pose_3d['positions'],
                    pose_3d['valid'],
                )
                # Update valid mask since frozen joints are now "valid"
                pose_3d['valid'] = np.ones(17, dtype=bool)
            
            # Apply temporal filter
            if pose_3d and pose_filter:
                pose_3d['positions'] = pose_filter.filter(
                    pose_3d['positions'],
                    pose_3d['valid'],
                )
            
            # Transform to VRChat coordinates and send
            if pose_3d and np.any(pose_3d['valid']):
                # Transform coordinates
                transformed, _ = transform_pose_to_vrchat(
                    pose_3d['positions'],
                    coord_transform,
                )
                
                # Convert to VRChat trackers with rotation estimation
                trackers = pose_to_trackers_with_rotations(
                    transformed,
                    pose_3d['confidences'],
                    pose_3d['valid'],
                )
                
                # Send via OSC
                osc_sender.send_all_trackers(trackers)
            
            # Preview window
            if preview_window and args.preview:
                import cv2
                
                # Create mosaic of camera feeds
                preview_frames = []
                for cam_id in sorted(frames.keys()):
                    frame = frames[cam_id].copy()
                    
                    # Draw keypoints if available
                    if cam_id in keypoints_2d:
                        frame = detector.draw_keypoints(frame, keypoints_2d[cam_id])
                    
                    # Resize and add label
                    frame = cv2.resize(frame, (320, 180))
                    cv2.putText(frame, f"Cam {cam_id}", (5, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    preview_frames.append(frame)
                
                # Stack frames
                if len(preview_frames) <= 3:
                    mosaic = np.hstack(preview_frames)
                else:
                    row1 = np.hstack(preview_frames[:3])
                    row2 = np.hstack(preview_frames[3:] + [np.zeros_like(preview_frames[0])] * (3 - len(preview_frames[3:])))
                    mosaic = np.vstack([row1, row2])
                
                cv2.imshow(preview_window, mosaic)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Status update every second
            current_time = time.time()
            if current_time - last_status_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                valid_joints = np.sum(pose_3d['valid']) if pose_3d else 0
                trackers_sent = len(trackers) if pose_3d else 0
                
                status = f"\rFPS: {fps:.1f} | Joints: {valid_joints}/17 | Trackers: {trackers_sent}"
                
                # Add confidence filtering info
                if conf_diagnostics:
                    active_views = conf_diagnostics['active_views_per_joint']
                    min_views = int(np.min(active_views))
                    max_views = int(np.max(active_views))
                    avg_views = float(np.mean(active_views))
                    status += f" | Views: {min_views}-{max_views} (avg {avg_views:.1f})"
                
                status += f" | Frames: {frame_count}"
                print(status, end="")
                last_status_time = current_time
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        camera_manager.stop_all()
        osc_sender.disconnect()
        if preview_window:
            import cv2
            cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\nTracked {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
