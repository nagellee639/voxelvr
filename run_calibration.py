#!/usr/bin/env python3
"""
VoxelVR Calibration Wizard

Interactive tool for calibrating multiple cameras:
1. Intrinsics: Lens parameters for each camera
2. Extrinsics: Camera positions in room space

Usage:
    python run_calibration.py
    
Requirements:
    - 3+ USB webcams connected
    - Printed ChArUco calibration board
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from voxelvr.config import VoxelVRConfig, CameraConfig, CalibrationConfig
from voxelvr.calibration import (
    generate_charuco_pdf,
    capture_intrinsic_frames,
    calibrate_intrinsics,
    capture_extrinsic_frames,
    calibrate_extrinsics,
)
from voxelvr.calibration.intrinsics import get_camera_matrix, get_distortion_coeffs


def detect_cameras(max_cameras: int = 10) -> list[int]:
    """Detect available cameras."""
    import cv2
    
    available = []
    print("Detecting cameras...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  Camera {i}: {w}x{h}")
                available.append(i)
            cap.release()
    
    return available


def main():
    parser = argparse.ArgumentParser(description="VoxelVR Camera Calibration")
    parser.add_argument(
        "--cameras", "-c", 
        type=int, 
        nargs="+",
        help="Camera IDs to calibrate (auto-detect if not specified)"
    )
    parser.add_argument(
        "--generate-board", "-g",
        action="store_true",
        help="Generate printable ChArUco board and exit"
    )
    parser.add_argument(
        "--intrinsics-only", "-i",
        action="store_true", 
        help="Only calibrate intrinsics"
    )
    parser.add_argument(
        "--extrinsics-only", "-e",
        action="store_true",
        help="Only calibrate extrinsics (requires existing intrinsics)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for calibration data"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    config = VoxelVRConfig.load()
    config.ensure_dirs()
    
    output_dir = args.output_dir or config.calibration_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate board if requested
    if args.generate_board:
        board_path = output_dir / "charuco_board.png"
        generate_charuco_pdf(
            board_path,
            config.calibration.charuco_squares_x,
            config.calibration.charuco_squares_y,
            config.calibration.charuco_square_length,
            config.calibration.charuco_marker_length,
            config.calibration.charuco_dict,
        )
        print(f"\nBoard saved to: {board_path}")
        print("Print this at 100% scale and measure the square size to verify.")
        return 0
    
    # Detect or use specified cameras
    camera_ids = args.cameras or detect_cameras()
    
    if len(camera_ids) < 3:
        print(f"\nWarning: Only {len(camera_ids)} cameras detected.")
        print("VoxelVR works best with 3+ cameras for accurate triangulation.")
        if len(camera_ids) == 0:
            print("No cameras found! Please connect webcams and try again.")
            return 1
    
    print(f"\nUsing cameras: {camera_ids}")
    
    # Calibrate intrinsics
    intrinsics_list = []
    
    if not args.extrinsics_only:
        print("\n" + "="*50)
        print("STEP 1: INTRINSIC CALIBRATION")
        print("="*50)
        print("For each camera, hold the ChArUco board at various")
        print("angles and distances. Press SPACE to capture frames.")
        print("-"*50)
        
        for cam_id in camera_ids:
            print(f"\n>>> Calibrating Camera {cam_id}")
            
            # Capture frames
            frames = capture_intrinsic_frames(
                cam_id,
                config.calibration,
                output_dir / f"intrinsic_frames_cam{cam_id}",
            )
            
            if len(frames) < 5:
                print(f"Insufficient frames for camera {cam_id}, skipping...")
                continue
            
            # Calibrate
            intrinsics = calibrate_intrinsics(
                frames,
                config.calibration,
                camera_id=cam_id,
                camera_name=f"Camera {cam_id}",
            )
            
            if intrinsics:
                # Save intrinsics
                intrinsics_path = output_dir / f"intrinsics_cam{cam_id}.json"
                intrinsics.save(intrinsics_path)
                print(f"Saved: {intrinsics_path}")
                intrinsics_list.append(intrinsics)
    else:
        # Load existing intrinsics
        print("\nLoading existing intrinsics...")
        for cam_id in camera_ids:
            intrinsics_path = output_dir / f"intrinsics_cam{cam_id}.json"
            if intrinsics_path.exists():
                from voxelvr.config import CameraIntrinsics
                intrinsics = CameraIntrinsics.load(intrinsics_path)
                intrinsics_list.append(intrinsics)
                print(f"Loaded: {intrinsics_path}")
    
    if args.intrinsics_only:
        print("\nIntrinsics calibration complete!")
        return 0
    
    # Calibrate extrinsics
    if len(intrinsics_list) < 2:
        print("\nNeed at least 2 calibrated cameras for extrinsics!")
        return 1
    
    print("\n" + "="*50)
    print("STEP 2: EXTRINSIC CALIBRATION")
    print("="*50)
    print("Wave the ChArUco board in the center of your play area,")
    print("keeping it visible to ALL cameras simultaneously.")
    print("Press SPACE when all cameras show 'OK'.")
    print("-"*50)
    
    # Get camera IDs from intrinsics
    intrinsic_cam_ids = [intr.camera_id for intr in intrinsics_list]
    
    # Capture synchronized frames
    captures = capture_extrinsic_frames(
        intrinsic_cam_ids,
        intrinsics_list,
        config.calibration,
        output_dir / "extrinsic_frames",
    )
    
    if len(captures) < 5:
        print("Insufficient synchronized captures!")
        return 1
    
    # Calibrate extrinsics
    calibration = calibrate_extrinsics(
        captures,
        intrinsics_list,
        config.calibration,
    )
    
    if calibration:
        # Save complete calibration
        calibration_path = output_dir / "calibration.json"
        calibration.save(calibration_path)
        print(f"\nCalibration saved: {calibration_path}")
        
        # Print camera positions
        print("\nCamera Positions (meters from origin):")
        for cam_id, data in calibration.cameras.items():
            pos = data['extrinsics']['translation_vector']
            print(f"  Camera {cam_id}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        print("\nCalibration complete! You can now run:")
        print("  python run_demo.py")
    else:
        print("\nExtrinsic calibration failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
