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

from voxelvr.pose.detector_2d import PoseDetector2D
from voxelvr.pose.triangulation import TriangulationPipeline
from voxelvr.pose.filter import PoseFilter


def get_latest_tracking_dataset():
    """Get the most recent tracking dataset."""
    base = Path("dataset/tracking")
    if not base.exists():
        return None
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    return dirs[0] if dirs else None


def main():
    print("=" * 70)
    print("3D TRACKING VALIDATION")
    print("=" * 70)
    
    # Find dataset
    dataset = get_latest_tracking_dataset()
    if not dataset:
        print("ERROR: No tracking dataset found")
        return 1
    
    print(f"\nDataset: {dataset}")
    
    # Get camera directories
    cam_dirs = sorted([
        d for d in dataset.iterdir()
        if d.is_dir() and d.name.startswith("cam_")
    ])
    
    if not cam_dirs:
        print("ERROR: No camera directories found")
        return 1
    
    print(f"Cameras: {len(cam_dirs)}")
    
    # Load image files
    cam_files = {}
    for cam_dir in cam_dirs:
        cam_id = int(cam_dir.name.split("_")[1])
        files = sorted(list(cam_dir.glob("*.jpg")))
        cam_files[cam_id] = files
        print(f"  Camera {cam_id}: {len(files)} images")
    
    # Get minimum frame count
    min_frames = min(len(f) for f in cam_files.values())
    print(f"\nProcessing {min_frames} synchronized frames")
    
    # Initialize detector
    print("\nInitializing pose detector...")
    detector = PoseDetector2D(backend="auto")
    if not detector.load_model():
        print("ERROR: Failed to load pose detector")
        return 1
    
    # Create projection matrices (dummy for now)
    projection_matrices = {}
    for cam_id in cam_files.keys():
        P = np.array([
            [1000, 0, 640, 0],
            [0, 1000, 360, 0],
            [0, 0, 1, 0]
        ], dtype=np.float64)
        projection_matrices[cam_id] = P
    
    triangulator = TriangulationPipeline(projection_matrices)
    pose_filter = PoseFilter(num_joints=17)
    
    # Process frames (limit to 50 for analysis)
    num_frames = min(min_frames, 50)
    print(f"\nProcessing {num_frames} frames...\n")
    
    # Collect all 3D positions
    all_positions = []
    all_confidences = []
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
        
        # Triangulation
        result_3d = triangulator.process(keypoints_list)
        
        if result_3d is not None:
            valid_frames += 1
            # Filter
            filtered = pose_filter.filter(result_3d['positions'], result_3d['valid'])
            
            all_positions.append(filtered)
            all_confidences.append(result_3d['valid'].astype(float))
            
            if (i + 1) % 10 == 0:
                print(f"  Frame {i+1}/{num_frames}: {np.sum(result_3d['valid'])}/17 joints tracked")
    
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
    print(f"Position data shape: {positions.shape}")
    
    # Analyze positions
    print(f"\n--- Position Ranges (meters) ---")
    for axis, name in enumerate(['X', 'Y', 'Z']):
        axis_data = positions[:, :, axis]
        valid_data = axis_data[confidences > 0.5]
        if len(valid_data) > 0:
            print(f"{name}-axis: min={np.min(valid_data):.3f}m, "
                  f"max={np.max(valid_data):.3f}m, "
                  f"range={np.max(valid_data)-np.min(valid_data):.3f}m")
    
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
    
    # Temporal stability
    print(f"\n--- Temporal Stability (frame-to-frame movement) ---")
    for joint_idx in [0, 5, 6, 11, 12]:  # Key joints
        joint_pos = positions[:, joint_idx, :]
        joint_conf = confidences[:, joint_idx]
        
        # Calculate frame-to-frame differences
        diffs = np.diff(joint_pos, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        
        # Only consider frames where joint was tracked
        valid_mask = (joint_conf[:-1] > 0.5) & (joint_conf[1:] > 0.5)
        
        if np.sum(valid_mask) > 5:
            valid_diffs = distances[valid_mask]
            print(f"{joint_names[joint_idx]:15s}: avg={np.mean(valid_diffs)*100:.2f}cm, "
                  f"max={np.max(valid_diffs)*100:.2f}cm")
    
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
            print("⚠ Shoulder width may be incorrect")
        
        if 20 < hip_width*100 < 50:
            print("✓ Hip width looks reasonable")
        else:
            print("⚠ Hip width may be incorrect")
        
        if 30 < torso_height*100 < 80:
            print("✓ Torso height looks reasonable")
        else:
            print("⚠ Torso height may be incorrect")
    
    # Save sample data
    output_file = Path("3d_tracking_validation.json")
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': str(dataset),
            'frames_analyzed': num_frames,
            'valid_frames': valid_frames,
            'average_joints_per_frame': float(np.mean(np.sum(confidences > 0.5, axis=1))),
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
    print(f"\n{'='*70}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
