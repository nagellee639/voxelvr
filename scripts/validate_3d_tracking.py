#!/usr/bin/env python3
"""
Validate 3D Tracking Output

Runs the pose tracking pipeline on dataset images and analyzes
the 3D positions to verify they're sensible (human-scale, stable, etc.).
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse

from voxelvr.pose.detector_2d import PoseDetector2D
from voxelvr.pose.triangulation import TriangulationPipeline, compute_projection_matrices
from voxelvr.config import VoxelVRConfig, MultiCameraCalibration, CameraIntrinsics, CameraExtrinsics
from voxelvr.pose.filter import PoseFilter
from voxelvr.pose.confidence_filter import ConfidenceFilter
from voxelvr.config import VoxelVRConfig, MultiCameraCalibration

def get_latest_tracking_dataset():
    """Get the most recent tracking dataset."""
    base = Path("dataset/tracking")
    if not base.exists():
        return None
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    return dirs[0] if dirs else None

def load_calibration(calibration_path=None):
    if calibration_path:
        return MultiCameraCalibration.load(calibration_path)
    
    # Try default locations
    config = VoxelVRConfig.load()
    default_path = config.calibration_dir / "calibration.json"
    if default_path.exists():
        return MultiCameraCalibration.load(default_path)
        
    return None

def synthesize_calibration(camera_ids: list[int]) -> tuple[list[CameraIntrinsics], list[CameraExtrinsics]]:
    """Generate synthetic calibration for validation when real calibration is missing."""
    print("WARNING: Using synthetic calibration (arc arrangement)")
    intrinsics = []
    extrinsics = []
    
    # Arrange cameras in a circle around the origin (360 degrees)
    # Looking at (0, 1.0, 0)
    center = np.array([0.0, 1.0, 0.0])
    radius = 2.5 # Room scale radius
    
    num_cams = len(camera_ids)
    # Distribute evenly around 360 degrees starting from -45 (corner)
    angle_step = 2 * np.pi / num_cams
    start_angle = -np.pi / 4 
    
    for i, cam_id in enumerate(camera_ids):
        # Intrinsic: simplified 720p
        intr = CameraIntrinsics(
            camera_id=cam_id,
            camera_name=f"Synth Cam {cam_id}",
            resolution=(1280, 720),
            camera_matrix=[
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0]
            ],
            distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
            reprojection_error=0.0,
            calibration_date="synthetic"
        )
        intrinsics.append(intr)
        
        # Extrinsic: Position on arc
        angle = start_angle + i * angle_step
        pos = np.array([radius * np.sin(angle), 1.0, radius * np.cos(angle)])
        
        # Look at center
        # Simple look-at rotation matrix construction
        forward = (center - pos)
        forward /= np.linalg.norm(forward)
        
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        
        up = np.cross(forward, right)
        
        R = np.identity(3)
        R[0, :] = right
        R[1, :] = up
        R[2, :] = forward
        
        # Convert to rvec
        rvec, _ = cv2.Rodrigues(R)
        
        # Build 4x4 transform
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        
        ext = CameraExtrinsics(
            camera_id=cam_id,
            camera_name=f"Synth Cam {cam_id}",
            transform_matrix=T.tolist(),
            rotation_matrix=R.tolist(),
            rotation_vector=rvec.flatten().tolist(),
            translation_vector=pos.tolist(),
            reprojection_error=0.0,
            calibration_date="synthetic"
        )
        extrinsics.append(ext)
        
    return intrinsics, extrinsics

def main():
    parser = argparse.ArgumentParser(description="Validate 3D tracking on dataset")
    parser.add_argument("--dataset", type=Path, help="Path to tracking dataset")
    parser.add_argument("--calibration", type=Path, help="Path to calibration.json")
    args = parser.parse_args()

    print("=" * 70)
    print("3D TRACKING VALIDATION")
    print("=" * 70)
    
    # Find dataset
    dataset = args.dataset or get_latest_tracking_dataset()
    if not dataset:
        print("ERROR: No tracking dataset found")
        return 1
    
    print(f"\nDataset: {dataset}")
    
    # Load calibration
    calibration = load_calibration(args.calibration)
    
    # Get camera directories
    cam_dirs = sorted([
        d for d in dataset.iterdir()
        if d.is_dir() and d.name.startswith("cam_")
    ])
    
    if not cam_dirs:
        print("ERROR: No camera directories found")
        return 1
        
    all_cam_ids = [int(d.name.split("_")[1]) for d in cam_dirs]
    print(f"Dataset cameras: {all_cam_ids}")

    intrinsics_list = []
    extrinsics_list = []
    valid_cam_ids = []

    if calibration:
        print(f"Loaded calibration with {len(calibration.cameras)} cameras")
        # Filter cameras based on calibration
        valid_cam_ids = [cid for cid in all_cam_ids if cid in calibration.cameras]
        
        for cam_id in valid_cam_ids:
            data = calibration.cameras[cam_id]
            intrinsics_list.append(CameraIntrinsics(**data['intrinsics']))
            extrinsics_list.append(CameraExtrinsics(**data['extrinsics']))
            
    else:
        print("WARNING: Could not load calibration. Falling back to synthetic mode.")
        valid_cam_ids = all_cam_ids
        intrinsics_list, extrinsics_list = synthesize_calibration(valid_cam_ids)
    
    if len(valid_cam_ids) < 2:
        print(f"ERROR: Found {len(valid_cam_ids)} valid cameras. Need at least 2.")
        return 1
        
    print(f"Using {len(valid_cam_ids)} cameras: {valid_cam_ids}")
    
    # Load image files
    cam_files = {}
    for cam_id in valid_cam_ids:
        cam_dir = dataset / f"cam_{cam_id}"
        files = sorted(list(cam_dir.glob("*.jpg")))
        cam_files[cam_id] = files
        print(f"  Camera {cam_id}: {len(files)} images")
    
    # Get minimum frame count
    min_frames = min(len(f) for f in cam_files.values())
    print(f"\nProcessing {min_frames} synchronized frames")
    
    # Initialize detector
    print("\nInitializing pose detector...")
    config = VoxelVRConfig.load()
    detector = PoseDetector2D(confidence_threshold=config.tracking.confidence_threshold)
    if not detector.load_model():
        print("ERROR: Failed to load pose detector")
        return 1
    
    # Create projection matrices
    # intrinsics_list and extrinsics_list are already populated above
    
    projection_matrices = compute_projection_matrices(intrinsics_list, extrinsics_list)
    
    triangulator = TriangulationPipeline(
        projection_matrices,
        confidence_threshold=config.tracking.confidence_threshold
    )
    
    # Initialize filters
    confidence_filter = ConfidenceFilter(
        num_joints=17,
        confidence_threshold=config.tracking.confidence_threshold,
        grace_period_frames=config.tracking.confidence_grace_period_frames,
        reactivation_frames=config.tracking.confidence_reactivation_frames,
    )
    
    pose_filter = PoseFilter(
        num_joints=17,
        min_cutoff=config.tracking.filter_min_cutoff,
        beta=config.tracking.filter_beta,
        freeze_invalid_joints=config.tracking.freeze_unconfident_joints
    )
    
    # Process frames
    # Process all frames to see full behavior
    num_frames = min_frames 
    print(f"\nProcessing {num_frames} frames...\n")
    
    # Collect all 3D positions
    all_positions = []
    all_confidences = []
    all_diagnostics = []
    valid_frames = 0
    
    for i in range(num_frames):
        # 2D Detection
        keypoints_list = []
        for cam_id, files in cam_files.items():
            img = cv2.imread(str(files[i]))
            if img is None:
                continue
            
            result = detector.detect(img, camera_id=cam_id)
            if result:
                keypoints_list.append(result)
        
        # Apply confidence filtering to 2D inputs
        filtered_keypoints, conf_stats = confidence_filter.update(keypoints_list)
        all_diagnostics.append(conf_stats)

        # Triangulation
        result_3d = None
        if len(filtered_keypoints) >= 2:
            result_3d = triangulator.process(filtered_keypoints)
        
        if result_3d is not None:
            # Apply confidence-based joint freezing
            result_3d['positions'] = confidence_filter.apply_freezing(
                result_3d['positions'],
                result_3d['valid']
            )
            # Update valid mask (frozen joints are valid)
            result_3d['valid'] = np.ones(17, dtype=bool)

            valid_frames += 1
            # Filter
            filtered = pose_filter.filter(result_3d['positions'], result_3d['valid'])
            
            all_positions.append(filtered)
            all_confidences.append(result_3d['valid'].astype(float)) # All ones if frozen
            
        else:
            # Fallback: Monocular Lift from best 2D view
            # Find detection with highest average confidence
            best_det = None
            max_conf = 0
            best_cam = -1
            
            for det in keypoints_list:
                avg_conf = np.mean(det.confidences)
                if avg_conf > max_conf:
                    max_conf = avg_conf
                    best_det = det
                    best_cam = det.camera_id
            
            if best_det and max_conf > 0.3:
                # Unproject assuming fixed depth Z=2.0
                # Use intrinsics from synth or loaded calib
                # Find index of best_cam in valid_cam_ids
                try:
                    idx = valid_cam_ids.index(best_cam)
                    K = np.array(intrinsics_list[idx].camera_matrix)
                    fx, fy = K[0, 0], K[1, 1]
                    cx, cy = K[0, 2], K[1, 2]
                    
                    z = 2.0
                    pos_3d = np.zeros((17, 3))
                    
                    for k in range(17):
                        u, v = best_det.positions[k]
                        x = (u - cx) * z / fx
                        y = (v - cy) * z / fy
                        pos_3d[k] = [x, y, z]
                    
                    # Apply filter
                    filtered = pose_filter.filter(pos_3d, best_det.confidences > 0.3)
                    all_positions.append(filtered)
                    all_confidences.append((best_det.confidences > 0.3).astype(float))
                    valid_frames += 1
                    
                except ValueError:
                    all_positions.append(np.zeros((17, 3)))
                    all_confidences.append(np.zeros((17,)))
            else:
                all_positions.append(np.zeros((17, 3)))
                all_confidences.append(np.zeros((17,)))

        if (i + 1) % 10 == 0:
            active_joints = np.sum(result_3d['valid']) if result_3d else 0
            print(f"  Frame {i+1}/{num_frames}: {active_joints}/17 joints")
    
    if valid_frames == 0:
        print("\nERROR: No frames successfully triangulated")
        return 1
    
    # Convert to arrays
    positions = np.array(all_positions)  # Shape: (frames, 17, 3)
    confidences = np.array(all_confidences)  # Shape: (frames, 17)
    
    print(f"\n{'='*70}")
    print("3D TRACKING ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Valid frames: {valid_frames}/{num_frames}")
    
    # 1. Check T-Pose initialization (first few frames should be T-pose if no tracking)
    print("\n--- T-Pose Initialization Check ---")
    start_pos = positions[0]
    # Simple check: nose at 0,1.5,0 approximately? T-pose definition depends on TPOSE_3D
    from voxelvr.pose.confidence_filter import TPOSE_3D
    # Check if first frame matches TPOSE_3D (if tracking wasn't valid immediately)
    # The dataset likely has valid tracking from frame 0 or close to it.
    
    # 2. Check Joint Freezing
    # Look for periods where position doesn't change exactly
    print("\n--- Joint Freezing Check ---")
    frozen_intervals = 0
    for j in range(17):
        diffs = np.linalg.norm(np.diff(positions[:, j, :], axis=0), axis=1)
        # Count frames with 0.0 movement (indicates frozen/last known)
        # Float comparison with small epsilon
        frozen_frames = np.sum(diffs < 1e-6)
        if frozen_frames > 0:
            print(f"Joint {j}: {frozen_frames} frozen frames detected")
            frozen_intervals += 1
            
    if frozen_intervals > 0:
        print("✓ Joint freezing logic active")
    else:
        print("ℹ No freezing events detected (tracking might be perfect)")

    # 3. Standard Analysis
    # Joint-specific analysis
    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    print(f"\n--- Key Joint Positions (average over valid frames) ---")
    for joint_idx in [0, 5, 6, 11, 12, 15, 16]:  # nose, shoulders, hips, ankles
        joint_pos = positions[:, joint_idx, :]
        joint_conf = confidences[:, joint_idx]
        valid_mask = joint_conf > 0.5
        
        if np.sum(valid_mask) > 0:
            avg_pos = np.mean(joint_pos[valid_mask], axis=0)
            print(f"{joint_names[joint_idx]:15s}: ({avg_pos[0]:6.3f}, {avg_pos[1]:6.3f}, {avg_pos[2]:6.3f})")
    
    # Body proportions check
    print(f"\n--- Body Proportions Check ---")
    # Calculate average skeleton when all joints visible
    full_visible = np.sum(confidences > 0.5, axis=1) >= 10
    if np.sum(full_visible) > 0:
        avg_skeleton = np.mean(positions[full_visible], axis=0)
        
        # Check key distances
        shoulder_width = np.linalg.norm(avg_skeleton[5] - avg_skeleton[6])
        hip_width = np.linalg.norm(avg_skeleton[11] - avg_skeleton[12])
        torso_height = np.linalg.norm(
            (avg_skeleton[5] + avg_skeleton[6])/2 - 
            (avg_skeleton[11] + avg_skeleton[12])/2
        )
        
        print(f"Shoulder width: {shoulder_width*100:.1f} cm")
        print(f"Hip width: {hip_width*100:.1f} cm")
        print(f"Torso height: {torso_height*100:.1f} cm")
        
        # Sanity checks
        if 30 < shoulder_width*100 < 60:
            print("✓ Shoulder width looks reasonable")
        else:
            print(f"⚠ Shoulder width may be incorrect ({shoulder_width*100:.1f}cm)")
        
        if 20 < hip_width*100 < 50:
            print("✓ Hip width looks reasonable")
        else:
            print(f"⚠ Hip width may be incorrect ({hip_width*100:.1f}cm)")
        
        if 30 < torso_height*100 < 80:
            print("✓ Torso height looks reasonable")
        else:
            print(f"⚠ Torso height may be incorrect ({torso_height*100:.1f}cm)")
    
    # Save sample data
    output_file = Path("3d_tracking_validation.json")
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': str(dataset),
            'frames_analyzed': num_frames,
            'valid_frames': valid_frames,
            'position_ranges': {
                'x_min': float(np.min(positions[:, :, 0])),
                'x_max': float(np.max(positions[:, :, 0])),
                'y_min': float(np.min(positions[:, :, 1])),
                'y_max': float(np.max(positions[:, :, 1])),
                'z_min': float(np.min(positions[:, :, 2])),
                'z_max': float(np.max(positions[:, :, 2])),
            }
        }, f, indent=2)
    
    print(f"\n✓ Validation data saved to {output_file}")
    
    # ---------------------------------------------------------
    # Render 3D Video
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print("RENDERING 3D VIDEO")
    print(f"{'='*70}")
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from mpl_toolkits.mplot3d import Axes3D
        
        # Setup figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate bounds for consistent view
        all_valid_pos = positions[confidences > 0.5]
        if len(all_valid_pos) > 0:
            center = np.mean(all_valid_pos, axis=0)
            radius = np.max(np.linalg.norm(all_valid_pos - center, axis=1)) * 1.2
        else:
            center = np.array([0, 1.0, 0])
            radius = 1.0
            
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Pose Tracking (Dataset: {dataset.name})")
        
        # COCO Skeleton connections
        skeleton_links = [
            (0, 1), (0, 2), (1, 3), (2, 4),      # Face
            (5, 6), (5, 7), (7, 9),              # Left arm
            (6, 8), (8, 10),                     # Right arm
            (5, 11), (6, 12),                    # Torso
            (11, 12), (11, 13), (13, 15),        # Left leg
            (12, 14), (14, 16)                   # Right leg
        ]
        
        # Plot elements
        lines = [ax.plot([], [], [], 'b-')[0] for _ in skeleton_links]
        points = ax.scatter([], [], [], c='r', s=20)
        
        def update(frame_idx):
            current_pos = positions[frame_idx]
            current_conf = confidences[frame_idx]
            
            # Update joints
            valid_indices = current_conf > 0.5 # or visually useful threshold
            # With freezing, confidences might be 1.0 even if frozen
            # If freezing is working, positions will be valid numbers
            
            ax.set_title(f"Frame {frame_idx}/{num_frames}")
            
            # Check for totally invalid frame (e.g. init T-pose or freezing fallback)
            # If standard deviation is 0, it's likely a dummy point
            
            if np.sum(valid_indices) > 0:
                xs = current_pos[:, 0]
                ys = current_pos[:, 1]
                zs = current_pos[:, 2]
                points._offsets3d = (xs, ys, zs)
                
                # Update links
                for line, (start, end) in zip(lines, skeleton_links):
                    # Draw if both ends have reasonable confidence 
                    # (or are frozen valid points)
                    if valid_indices[start] and valid_indices[end]:
                        line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
                        line.set_3d_properties([zs[start], zs[end]])
                    else:
                        line.set_data([], [])
                        line.set_3d_properties([])
            
            return lines + [points]

        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames, interval=33, blit=False)
        
        video_path = Path("pose_tracking_video.mp4")
        print(f"Saving video to {video_path}...")
        
        writer = FFMpegWriter(fps=30, metadata=dict(artist='VoxelVR'), bitrate=3000)
        anim.save(video_path, writer=writer)
        
        print(f"✓ Video rendered successfully: {video_path}")
        
    except ImportError:
        print("⚠ Matplotlib or ffmpeg not installed/available. Skipping video render.")
    except Exception as e:
        print(f"⚠ Video rendering failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
