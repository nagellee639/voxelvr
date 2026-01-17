#!/usr/bin/env python3
"""
Calibrate from Dataset

Runs the full calibration pipeline using images from a dataset folder
instead of live camera capture.
"""

import argparse
import sys
import json
import cv2
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voxelvr.config import VoxelVRConfig, CalibrationConfig
from voxelvr.calibration import (
    calibrate_intrinsics,
    calibrate_extrinsics,
    detect_charuco
)

def load_frames(dataset_dir: Path) -> dict[int, list[str]]:
    """Load image paths for each camera."""
    cam_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("cam_")])
    frames = {}
    for d in cam_dirs:
        cam_id = int(d.name.split("_")[1])
        files = sorted(list(d.glob("*.jpg")))
        frames[cam_id] = files
        print(f"Camera {cam_id}: {len(files)} frames")
    return frames

def main():
    parser = argparse.ArgumentParser(description="Calibrate from Dataset")
    parser.add_argument("dataset", type=Path, help="Path to calibration dataset root")
    parser.add_argument("--output", "-o", type=Path, default="dataset_calibration.json", help="Output calibration file")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        return 1

    print(f"Loading dataset: {args.dataset}")
    cam_files = load_frames(args.dataset)
    
    if not cam_files:
        print("No camera directories found!")
        return 1

    config = VoxelVRConfig.load()
    calib_config = config.calibration
    
    # 1. Intrinsics
    intrinsics_list = []
    print("\n--- Intrinsic Calibration ---")
    
    for cam_id, files in cam_files.items():
        print(f"\nProcessing Camera {cam_id}...")
        
        # Load images
        images = []
        valid_files = []
        
        # Limit frames for intrinsics if too many? typical dataset has ~50-100
        # Use all for best accuracy
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                images.append(img)
                valid_files.append(f)
        
        if not images:
            print(f"No valid images for camera {cam_id}")
            continue

        # Calibrate
        intrinsics = calibrate_intrinsics(
            images,
            calib_config,
            camera_id=cam_id,
            camera_name=f"Camera {cam_id}"
        )
        
        if intrinsics:
            intrinsics_list.append(intrinsics)
            print(f"  RMS Error: {intrinsics.rms_error:.4f}")
        else:
            print(f"  Failed to calibrate camera {cam_id}")

    if len(intrinsics_list) < 2:
        print("Not enough calibrated cameras for extrinsics")
        return 1

    # 2. Extrinsics
    print("\n--- Extrinsic Calibration ---")
    
    # We need synchronized frames. Assuming dataset files are synchronized by index.
    min_len = min(len(cam_files[cid]) for cid in cam_files)
    print(f"Using {min_len} synchronized frames")
    
    captures = []
    
    intrinsic_cam_ids = [intr.camera_id for intr in intrinsics_list]
    
    for i in range(min_len):
        frame_set = {}
        for cam_id in intrinsic_cam_ids:
            # Re-read image (inefficient but safe) or cache?
            # Images are large, don't cache all. Read on demand.
            img_path = cam_files[cam_id][i]
            img = cv2.imread(str(img_path))
            if img is not None:
                frame_set[cam_id] = img
        
        if len(frame_set) == len(intrinsic_cam_ids):
            captures.append(frame_set)
            
        if (i+1) % 10 == 0:
            print(f"  Loaded {len(captures)} frame sets")

    print(f"Running extrinsic calibration on {len(captures)} sets...")
    
    calibration = calibrate_extrinsics(
        captures,
        intrinsics_list,
        calib_config
    )
    
    if calibration:
        calibration.save(args.output)
        print(f"\nSUCCESS: Calibration saved to {args.output}")
        
        # Print results
        print("\nCamera Extrinsics:")
        for cam_id, cam in calibration.cameras.items():
            pos = cam['extrinsics']['translation_vector']
            print(f"  Camera {cam_id}: {pos}")
            
        return 0
    else:
        print("\nFAILED: Extrinsic calibration failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
