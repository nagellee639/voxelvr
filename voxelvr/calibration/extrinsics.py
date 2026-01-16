"""
Multi-Camera Extrinsics Calibration

Determines the position and orientation of each camera in world space
(relative to a common origin point, typically floor center of play area).
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import json

from .charuco import create_charuco_board, detect_charuco, estimate_pose
from .intrinsics import get_camera_matrix, get_distortion_coeffs
from ..config import CameraExtrinsics, CameraIntrinsics, CalibrationConfig, MultiCameraCalibration


def rotation_vector_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R


def create_transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 transformation matrix from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T



def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _detect_worker(frame: np.ndarray, board_params: tuple, dict_name: str, cam_matrix: Optional[np.ndarray], dist_coeffs: Optional[np.ndarray]) -> dict:
    """Worker function for parallel ChArUco detection."""
    # Re-create board locally to ensure pickle compatibility/safety
    squares_x, squares_y, square_len, marker_len = board_params
    board, aruco_dict = create_charuco_board(
        squares_x, squares_y, square_len, marker_len, dict_name
    )
    return detect_charuco(frame, board, aruco_dict, cam_matrix, dist_coeffs)


def capture_extrinsic_frames(
    camera_ids: List[int],
    intrinsics_list: List[CameraIntrinsics],
    config: CalibrationConfig,
    output_dir: Optional[Path] = None,
    show_preview: bool = True,
) -> List[Dict[int, np.ndarray]]:
    """
    Capture synchronized frames from all cameras for extrinsic calibration.
    
    User waves ChArUco board in the center of the tracking volume,
    ensuring it's visible to all cameras simultaneously.
    
    Args:
        camera_ids: List of camera indices
        intrinsics_list: Pre-calibrated intrinsics for each camera
        config: Calibration configuration
        output_dir: Optional directory to save frames
        show_preview: Whether to show live preview
        
    Returns:
        List of dicts mapping camera_id -> frame for synchronized captures
    """
    if len(camera_ids) != len(intrinsics_list):
        raise ValueError("Must provide intrinsics for each camera")
    
    board, aruco_dict = create_charuco_board(
        config.charuco_squares_x,
        config.charuco_squares_y,
        config.charuco_square_length,
        config.charuco_marker_length,
        config.charuco_dict,
    )
    
    # Open all cameras
    caps = {}
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {cam_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        caps[cam_id] = cap
    
    # Build intrinsics lookup
    intrinsics_map = {intr.camera_id: intr for intr in intrinsics_list}
    
    captures = []
    
    print(f"\n=== Extrinsic Calibration ({len(camera_ids)} cameras) ===")
    print(f"Capture {config.extrinsic_frames_required} frames with board visible to ALL cameras")
    print("Wave board slowly around the center of play area")
    print("SPACE: Capture when all cameras see board | ESC: Finish | Q: Quit\n")
    
    while len(captures) < config.extrinsic_frames_required:
        # Read from all cameras
        frames = {}
        results = {}
        all_detected = True
        
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                all_detected = False
                continue
            
            frames[cam_id] = frame
            
            # Get calibration for this camera
            intr = intrinsics_map.get(cam_id)
            cam_matrix = get_camera_matrix(intr) if intr else None
            dist_coeffs = get_distortion_coeffs(intr) if intr else None
            
            result = detect_charuco(frame, board, aruco_dict, cam_matrix, dist_coeffs)
            results[cam_id] = result
            
            if not result['success']:
                all_detected = False
        
        # Create preview mosaic
        if show_preview:
            preview_frames = []
            for cam_id in camera_ids:
                if cam_id in results:
                    img = results[cam_id]['image_with_markers']
                    # Add camera label
                    status = "OK" if results[cam_id]['success'] else "NO BOARD"
                    color = (0, 255, 0) if results[cam_id]['success'] else (0, 0, 255)
                    cv2.putText(img, f"Cam {cam_id}: {status}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    preview_frames.append(cv2.resize(img, (640, 360)))
            
            if preview_frames:
                # Arrange in grid
                n = len(preview_frames)
                cols = min(n, 3)
                rows = (n + cols - 1) // cols
                
                # Pad to fill grid
                while len(preview_frames) < rows * cols:
                    preview_frames.append(np.zeros_like(preview_frames[0]))
                
                rows_imgs = []
                for r in range(rows):
                    row_imgs = preview_frames[r*cols:(r+1)*cols]
                    rows_imgs.append(np.hstack(row_imgs))
                mosaic = np.vstack(rows_imgs)
                
                # Add status bar
                status_text = f"Captures: {len(captures)}/{config.extrinsic_frames_required}"
                if all_detected:
                    status_text += " | ALL CAMERAS READY - Press SPACE"
                cv2.putText(mosaic, status_text, (10, mosaic.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow("Extrinsic Calibration", mosaic)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and all_detected:
            captures.append(frames.copy())
            print(f"Captured frame set {len(captures)}/{config.extrinsic_frames_required}")
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                for cam_id, frame in frames.items():
                    cv2.imwrite(
                        str(output_dir / f"extrinsic_{len(captures):03d}_cam{cam_id}.png"),
                        frame
                    )
        elif key == 27:  # ESC
            print("Finishing early...")
            break
        elif key == ord('q'):
            print("Quitting...")
            for cap in caps.values():
                cap.release()
            cv2.destroyAllWindows()
            return []
    
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCaptured {len(captures)} synchronized frame sets")
    return captures


def calibrate_extrinsics(
    captures: List[Dict[int, np.ndarray]],
    intrinsics_list: List[CameraIntrinsics],
    config: CalibrationConfig,
    reference_camera_id: Optional[int] = None,
) -> Optional[MultiCameraCalibration]:
    """
    Calibrate extrinsics for all cameras from synchronized captures.
    
    Uses the ChArUco board pose to establish a common world coordinate system.
    The first valid detection defines the world origin (board center on floor).
    
    Args:
        captures: List of synchronized frame dicts
        intrinsics_list: Intrinsics for each camera
        config: Calibration configuration
        reference_camera_id: Camera to use as reference (or auto-select)
        
    Returns:
        MultiCameraCalibration object or None if failed
    """
    if len(captures) < 5:
        print("Error: Need at least 5 synchronized captures")
        return None
    
    board, aruco_dict = create_charuco_board(
        config.charuco_squares_x,
        config.charuco_squares_y,
        config.charuco_square_length,
        config.charuco_marker_length,
        config.charuco_dict,
    )
    
    intrinsics_map = {intr.camera_id: intr for intr in intrinsics_list}
    
    # Collect all board poses per camera
    camera_poses = {cam_id: [] for cam_id in intrinsics_map.keys()}
    
    # Try using optimized C++ implementation
    try:
        from .calibration_cpp import batch_detect_charuco
        print("Using optimized C++ calibration backend")
        
        # Prepare batch for C++
        # We need flat list of images and method to map back
        # captures is List[Dict[cam_id, frame]]
        
        # Flatten: (capture_idx, cam_id, frame, intr)
        flat_items = []
        for i, capture in enumerate(captures):
            for cam_id, frame in capture.items():
                if cam_id in intrinsics_map:
                    flat_items.append((cam_id, frame))
        
        if flat_items:
            images = [item[1] for item in flat_items]
            
            # Run batch detection
            results = batch_detect_charuco(
                images,
                config.charuco_squares_x,
                config.charuco_squares_y,
                config.charuco_square_length,
                config.charuco_marker_length,
                config.charuco_dict
            )
            
            # Process results
            for i, result in enumerate(results):
                if result['success']:
                    cam_id = flat_items[i][0]
                    intr = intrinsics_map[cam_id]
                    cam_matrix = get_camera_matrix(intr)
                    dist_coeffs = get_distortion_coeffs(intr)
                    
                    # Convert corners to float32 if not already
                    corners = result['corners']
                    ids = result['ids']
                    
                    success, rvec, tvec = estimate_pose(
                        corners, ids, board, cam_matrix, dist_coeffs
                    )
                    if success:
                        camera_poses[cam_id].append((rvec, tvec))
                        
    except ImportError:
        print("C++ extension not found, falling back to Python implementation")
        # Python implementation (Parallel or Sequential)
        # Previous Parallel implementation is good, but let's just stick to simple for robustness if C++ fails
        # or reuse the parallel implementation if verified.
        # Given I overwrote the parallel code earlier, I should probably put back a reliable python loop
        # or leave the parallel code if I didn't overwrite it fully?
        # I did overwrite it in step 113.
        # I will re-implement a simple loop here as fallback to avoid complexity
        
        for capture in captures:
            for cam_id, frame in capture.items():
                intr = intrinsics_map.get(cam_id)
                if intr is None:
                    continue
                
                cam_matrix = get_camera_matrix(intr)
                dist_coeffs = get_distortion_coeffs(intr)
                
                result = detect_charuco(frame, board, aruco_dict, cam_matrix, dist_coeffs)
                
                if result['success']:
                    success, rvec, tvec = estimate_pose(
                        result['corners'], result['ids'], board, cam_matrix, dist_coeffs
                    )
                    if success:
                        camera_poses[cam_id].append((rvec, tvec))
    
    # Check we have enough data
    for cam_id, poses in camera_poses.items():
        if len(poses) < 5:
            print(f"Warning: Camera {cam_id} only has {len(poses)} valid poses")
    
    # Select reference camera (one with most detections)
    if reference_camera_id is None:
        reference_camera_id = max(camera_poses.keys(), key=lambda k: len(camera_poses[k]))
    
    print(f"Using camera {reference_camera_id} as reference")
    
    # For each camera, compute average transform to world
    # World origin = average board position across all captures
    
    calibration = MultiCameraCalibration(
        origin_description="ChArUco board center (average position during calibration)"
    )
    
    for cam_id, poses in camera_poses.items():
        if len(poses) < 3:
            print(f"Skipping camera {cam_id}: insufficient data")
            continue
        
        # Average the poses (simple averaging of rotation vectors and translations)
        # For better results, could use quaternion averaging
        avg_rvec = np.mean([p[0] for p in poses], axis=0)
        avg_tvec = np.mean([p[1] for p in poses], axis=0)
        
        R = rotation_vector_to_matrix(avg_rvec)
        t = avg_tvec.flatten()
        
        # Transform from board to camera -> invert to get camera in world
        T_board_to_cam = create_transform_matrix(R, t)
        T_cam_to_world = invert_transform(T_board_to_cam)
        
        # Calculate reprojection error
        errors = []
        for rvec, tvec in poses:
            R_i = rotation_vector_to_matrix(rvec)
            angle_diff = np.arccos(np.clip((np.trace(R_i @ R.T) - 1) / 2, -1, 1))
            trans_diff = np.linalg.norm(tvec.flatten() - t)
            errors.append(angle_diff * 180 / np.pi + trans_diff * 100)  # Combined metric
        
        avg_error = np.mean(errors)
        
        extrinsics = CameraExtrinsics(
            camera_id=cam_id,
            camera_name=intrinsics_map[cam_id].camera_name,
            transform_matrix=T_cam_to_world.tolist(),
            rotation_matrix=T_cam_to_world[:3, :3].tolist(),
            translation_vector=T_cam_to_world[:3, 3].tolist(),
            reprojection_error=avg_error,
            calibration_date=datetime.now().isoformat(),
        )
        
        calibration.add_camera(intrinsics_map[cam_id], extrinsics)
        print(f"Camera {cam_id}: position = {T_cam_to_world[:3, 3]}, error = {avg_error:.2f}")
    
    return calibration


def get_projection_matrix(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    """
    Compute 3x4 projection matrix for a camera.
    
    P = K @ [R | t] where R,t transform from world to camera
    """
    K = np.array(intrinsics.camera_matrix)
    
    # Extrinsics stores camera-to-world, we need world-to-camera
    T_cam_to_world = np.array(extrinsics.transform_matrix)
    T_world_to_cam = invert_transform(T_cam_to_world)
    
    R = T_world_to_cam[:3, :3]
    t = T_world_to_cam[:3, 3]
    
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt
    
    return P
