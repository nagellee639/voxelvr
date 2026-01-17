"""
Camera Intrinsics Calibration

Determines the internal parameters of each camera:
- Focal length (fx, fy)
- Principal point (cx, cy)  
- Distortion coefficients (radial and tangential)
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import json

from .charuco import create_charuco_board, detect_charuco
from ..config import CameraIntrinsics, CalibrationConfig


def capture_intrinsic_frames(
    camera_id: int,
    config: CalibrationConfig,
    output_dir: Optional[Path] = None,
    show_preview: bool = True,
) -> List[np.ndarray]:
    """
    Interactively capture frames for intrinsic calibration.
    
    User should hold ChArUco board at various angles and distances.
    Press SPACE to capture a frame, ESC to finish early.
    
    Args:
        camera_id: OpenCV camera index
        config: Calibration configuration
        output_dir: Optional directory to save captured frames
        show_preview: Whether to show live preview
        
    Returns:
        List of captured frames with detected ChArUco boards
    """
    board, aruco_dict = create_charuco_board(
        config.charuco_squares_x,
        config.charuco_squares_y,
        config.charuco_square_length,
        config.charuco_marker_length,
        config.charuco_dict,
    )
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frames = []
    frame_count = 0
    
    print(f"\n=== Intrinsic Calibration for Camera {camera_id} ===")
    print(f"Capture {config.intrinsic_frames_required} frames with ChArUco board")
    print("Hold board at various angles and distances")
    print("SPACE: Capture frame | ESC: Finish early | Q: Quit\n")
    
    while len(frames) < config.intrinsic_frames_required:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            continue
        
        # Detect ChArUco
        result = detect_charuco(frame, board, aruco_dict)
        display = result['image_with_markers']
        
        # Add status text
        status = f"Frames: {len(frames)}/{config.intrinsic_frames_required}"
        if result['success']:
            status += f" | Corners: {len(result['corners'])} | READY"
            cv2.putText(display, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            status += " | No board detected"
            cv2.putText(display, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if show_preview:
            cv2.imshow(f"Intrinsic Calibration - Camera {camera_id}", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and result['success']:
            # Capture this frame
            frames.append(frame.copy())
            print(f"Captured frame {len(frames)}/{config.intrinsic_frames_required}")
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_dir / f"intrinsic_{len(frames):03d}.png"), frame)
            
        elif key == 27:  # ESC
            print("Finishing early...")
            break
        elif key == ord('q'):
            print("Quitting...")
            cap.release()
            cv2.destroyAllWindows()
            return []
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCaptured {len(frames)} frames for calibration")
    return frames


import time

def calibrate_intrinsics(
    frames: List[np.ndarray],
    config: CalibrationConfig,
    camera_id: int = 0,
    camera_name: str = "",
) -> Optional[CameraIntrinsics]:
    """
    Calibrate camera intrinsics from captured frames.
    
    Args:
        frames: List of frames with ChArUco board
        config: Calibration configuration
        camera_id: Camera identifier
        camera_name: Human-readable camera name
        
    Returns:
        CameraIntrinsics object or None if calibration failed
    """
    if len(frames) < 5:
        print("Error: Need at least 5 frames for calibration")
        return None
    
    board, aruco_dict = create_charuco_board(
        config.charuco_squares_x,
        config.charuco_squares_y,
        config.charuco_square_length,
        config.charuco_marker_length,
        config.charuco_dict,
    )
    
    # Collect corners from all frames
    all_corners = []
    all_ids = []
    image_size = None
    
    if len(frames) > 0:
        image_size = (frames[0].shape[1], frames[0].shape[0])

    # Try using optimized C++ implementation
    use_cpp = False
    try:
        if False: # Force Python implementation for dataset validation
            from .calibration_cpp_v2 import batch_detect_charuco
            use_cpp = True
            print("Using optimized C++ calibration backend")
    except ImportError:
        print("C++ extension not found, using Python implementation")

    total_frames = len(frames)
    print(f"Processing {total_frames} frames...")
    
    start_time = time.time()
    
    def progress_callback(current, total):
        elapsed = time.time() - start_time
        if current > 0 and elapsed > 0.1:
            rate = current / elapsed
            remaining_sec = (total - current) / rate
            mins, secs = divmod(int(remaining_sec), 60)
            time_str = f"{mins}m{secs:02d}s"
        else:
            time_str = "??m??s"
            
        percent = (current / total) * 100
        bar_len = 30
        filled = int(percent / 100 * bar_len)
        bar = '=' * filled + '-' * (bar_len - filled)
        
        print(f"\rProgress: [{bar}] {percent:5.1f}% | Est. Remaining: {time_str}   ", end="", flush=True)

    if use_cpp:
        # Process all frames at once with C++ and callback
        try:
            results = batch_detect_charuco(
                [np.ascontiguousarray(f, dtype=np.uint8) for f in frames],
                int(config.charuco_squares_x),
                int(config.charuco_squares_y),
                float(config.charuco_square_length),
                float(config.charuco_marker_length),
                str(config.charuco_dict),
                progress_callback
            )
            
            for res in results:
                if res.get('success') and res.get('corners') is not None and len(res['corners']) >= 4:
                    all_corners.append(res['corners'])
                    all_ids.append(res['ids'])
                    
        except Exception as e:
            print(f"\nBatch processing error: {e}")
            print("Falling back to Python implementation...")
            use_cpp = False
    
    if not use_cpp:
        # Python fallback (per frame)
        for i, frame in enumerate(frames):
            result = detect_charuco(frame, board, aruco_dict)
            if result['success'] and len(result['corners']) >= 4:
                all_corners.append(result['corners'])
                all_ids.append(result['ids'])
            
            # Update progress manually for python
            progress_callback(i + 1, total_frames)

    print("\nProcessing complete.")
    
    if len(all_corners) < 5:
        print(f"Error: Only {len(all_corners)} valid frames, need at least 5")
        return None
    
    print(f"Calibrating with {len(all_corners)} frames...")
    
    # Prepare data for calibration
    # Since cv2.aruco.calibrateCameraCharuco might be missing, we do it manually
    obj_points = []
    img_points = []
    
    board_corners = board.getChessboardCorners()
    
    for i in range(len(all_corners)):
        current_ids = all_ids[i].flatten()
        current_corners = all_corners[i]
        
        # Select object points corresponding to detected IDs
        # board_corners is (Total, 3), we treat it as list/array
        current_obj_points = board_corners[current_ids]
        
        obj_points.append(current_obj_points)
        img_points.append(current_corners)
    
    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )
    
    if not ret:
        print("Calibration failed")
        return None
    
    print(f"Calibration successful! Reprojection error: {ret:.4f} pixels")
    
    # Create intrinsics object
    intrinsics = CameraIntrinsics(
        camera_id=camera_id,
        camera_name=camera_name or f"Camera {camera_id}",
        resolution=image_size,
        camera_matrix=camera_matrix.tolist(),
        distortion_coeffs=dist_coeffs.flatten().tolist(),
        reprojection_error=ret,
        calibration_date=datetime.now().isoformat(),
    )
    
    return intrinsics


def load_intrinsics(path: Path) -> Optional[CameraIntrinsics]:
    """Load intrinsics from a JSON file."""
    if not path.exists():
        return None
    return CameraIntrinsics.load(path)


def get_camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert intrinsics to numpy camera matrix."""
    return np.array(intrinsics.camera_matrix, dtype=np.float64)


def get_distortion_coeffs(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert intrinsics to numpy distortion coefficients."""
    return np.array(intrinsics.distortion_coeffs, dtype=np.float64)
