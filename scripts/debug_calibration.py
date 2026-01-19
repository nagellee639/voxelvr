#!/usr/bin/env python3
"""
Debug script for camera calibration and skeleton visualization issues.

Usage: python3 scripts/debug_calibration.py [calibration.json]
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_calibration(calibration_path: Path):
    """Analyze a calibration file for common issues."""
    print(f"\n=== Calibration Analysis: {calibration_path} ===\n")
    
    with open(calibration_path) as f:
        data = json.load(f)
    
    cameras = data.get('cameras', {})
    print(f"Number of cameras: {len(cameras)}")
    
    if not cameras:
        print("ERROR: No cameras in calibration!")
        return
    
    # Extract camera positions and orientations
    positions = {}
    up_vectors = {}
    forward_vectors = {}
    
    for cam_id, cam_data in cameras.items():
        if 'extrinsics' not in cam_data:
            print(f"  Camera {cam_id}: Missing extrinsics!")
            continue
            
        T = np.array(cam_data['extrinsics']['transform_matrix'])
        pos = T[:3, 3]
        R = T[:3, :3]
        
        positions[cam_id] = pos
        up_vectors[cam_id] = R @ np.array([0, -1, 0])  # Camera -Y is up
        forward_vectors[cam_id] = R @ np.array([0, 0, 1])  # Camera Z is forward
        
        print(f"  Camera {cam_id}:")
        print(f"    Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"    Up vector: [{up_vectors[cam_id][0]:.3f}, {up_vectors[cam_id][1]:.3f}, {up_vectors[cam_id][2]:.3f}]")
    
    # Compute pairwise distances
    print("\n=== Pairwise Distances ===")
    cam_ids = list(positions.keys())
    for i, id_a in enumerate(cam_ids):
        for id_b in cam_ids[i+1:]:
            dist = np.linalg.norm(positions[id_a] - positions[id_b])
            print(f"  {id_a} <-> {id_b}: {dist:.3f}m")
    
    # Check up vector consistency
    print("\n=== Up Vector Analysis ===")
    all_ups = list(up_vectors.values())
    if all_ups:
        avg_up = np.mean(all_ups, axis=0)
        avg_up_norm = avg_up / np.linalg.norm(avg_up)
        print(f"  Average up vector: [{avg_up_norm[0]:.3f}, {avg_up_norm[1]:.3f}, {avg_up_norm[2]:.3f}]")
        
        # Expected up is [0, 1, 0]
        expected_up = np.array([0, 1, 0])
        alignment = np.dot(avg_up_norm, expected_up)
        angle = np.degrees(np.arccos(np.clip(alignment, -1, 1)))
        print(f"  Alignment with Y-up: {alignment:.3f} ({angle:.1f}° off)")
        
        if angle > 30:
            print("  WARNING: Up vector is significantly off from Y-up!")
            print("  This suggests cameras may not be mounted upright or calibration has errors.")
    
    # Check for extreme positions
    print("\n=== Position Sanity Check ===")
    all_pos = np.array(list(positions.values()))
    center = np.mean(all_pos, axis=0)
    max_dist = np.max(np.linalg.norm(all_pos - center, axis=1))
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Max distance from center: {max_dist:.3f}m")
    
    if max_dist > 5:
        print("  WARNING: Cameras are very spread out (>5m from center)")
        print("  This is unusual for a webcam setup - check for calibration errors.")
    
    return data

def check_skeleton_viewer_cameras():
    """Check if skeleton viewer is receiving camera data."""
    print("\n=== Skeleton Viewer Camera Check ===")
    print("  To debug: Add print statements in skeleton_viewer.py:render_skeleton()")
    print("  At line ~247, add: print(f'Cameras to render: {list(self.camera_positions.keys())}')")

def suggest_debug_steps():
    """Print debug suggestions."""
    print("\n=== Debug Steps ===")
    print("""
1. **Camera Positions in 3D Viewer**:
   - Check if load_calibration is called after extrinsics complete
   - Add debug print in unified_app.py:_on_calibration_complete() to verify
   - Add print in skeleton_viewer.py:set_camera_positions() to see what's received

2. **Incorrect Camera Positions / Up Vector**:
   - The "Estimated World Up Vector" should be close to [0, 1, 0] or [0, -1, 0]
   - Your result [ 0.72, -0.69, -0.01] suggests ~45° tilt
   - Possible causes:
     a) Cameras are physically tilted significantly
     b) ChArUco board detection is inconsistent
     c) Pairwise calibration has high errors
   
3. **High Pairwise Calibration Errors**:
   - Pair (0, 6) had error 16.30-25.53 - this is borderline acceptable
   - Try to get ALL cameras to see the board at different positions
   - Move the ChArUco board SLOWLY to reduce motion blur
   
4. **Distorted Skeleton**:
   - If cameras are in wrong positions, triangulation will produce garbage
   - Verify with: python3 scripts/debug_calibration.py ~/.config/voxelvr/calibration/calibration.json

5. **Quick Test - Force Camera Display**:
   In skeleton_viewer.py:render_skeleton(), add at line 252:
   ```python
   print(f"DEBUG: Rendering {len(self.camera_positions)} cameras")
   ```
""")

if __name__ == "__main__":
    # Default calibration path
    default_path = Path.home() / ".config/voxelvr/calibration/calibration.json"
    
    if len(sys.argv) > 1:
        calib_path = Path(sys.argv[1])
    elif default_path.exists():
        calib_path = default_path
    else:
        print("No calibration file found. Run calibration first or specify path.")
        suggest_debug_steps()
        sys.exit(1)
    
    if calib_path.exists():
        analyze_calibration(calib_path)
    else:
        print(f"Calibration file not found: {calib_path}")
    
    check_skeleton_viewer_cameras()
    suggest_debug_steps()
