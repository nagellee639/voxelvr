"""
Pairwise Extrinsics Calibration with Transform Chaining

This module provides camera calibration when not all cameras can see
the calibration board simultaneously. It works by:
1. Calibrating camera pairs that can see the board together
2. Building a graph of camera relationships
3. Chaining transforms to establish positions for all cameras

The approach uses a reference camera as the origin and propagates
transforms through the graph using shortest paths.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import heapq

from .charuco import create_charuco_board, detect_charuco, estimate_pose
from .intrinsics import get_camera_matrix, get_distortion_coeffs
from .extrinsics import (
    rotation_vector_to_matrix, 
    create_transform_matrix, 
    invert_transform
)
from ..config import CameraIntrinsics, CameraExtrinsics, CalibrationConfig, MultiCameraCalibration


@dataclass
class PairwiseCalibrationResult:
    """Result of calibrating a camera pair."""
    camera_a: int
    camera_b: int
    transform_a_to_b: np.ndarray  # 4x4 transform from camera A to camera B
    reprojection_error: float
    num_frames_used: int


def calibrate_camera_pair(
    captures: List[Dict[int, np.ndarray]],
    intrinsics_a: CameraIntrinsics,
    intrinsics_b: CameraIntrinsics,
    config: CalibrationConfig,
) -> Optional[PairwiseCalibrationResult]:
    """
    Calibrate the relative pose between two cameras.
    
    Args:
        captures: List of frame dicts where both cameras have frames
        intrinsics_a: Intrinsics for camera A
        intrinsics_b: Intrinsics for camera B
        config: Calibration configuration
        
    Returns:
        PairwiseCalibrationResult or None if calibration failed
    """
    cam_a = intrinsics_a.camera_id
    cam_b = intrinsics_b.camera_id
    
    board, aruco_dict = create_charuco_board(
        config.charuco_squares_x,
        config.charuco_squares_y,
        config.charuco_square_length,
        config.charuco_marker_length,
        config.charuco_dict,
    )
    
    cam_matrix_a = get_camera_matrix(intrinsics_a)
    dist_coeffs_a = get_distortion_coeffs(intrinsics_a)
    cam_matrix_b = get_camera_matrix(intrinsics_b)
    dist_coeffs_b = get_distortion_coeffs(intrinsics_b)
    
    # Collect relative transforms from each frame
    transforms = []
    
    for capture in captures:
        if cam_a not in capture or cam_b not in capture:
            continue
            
        frame_a = capture[cam_a]
        frame_b = capture[cam_b]
        
        # Detect board in both cameras
        result_a = detect_charuco(frame_a, board, aruco_dict, cam_matrix_a, dist_coeffs_a)
        result_b = detect_charuco(frame_b, board, aruco_dict, cam_matrix_b, dist_coeffs_b)
        
        if not result_a['success'] or not result_b['success']:
            continue
            
        # Estimate board pose in each camera
        success_a, rvec_a, tvec_a = estimate_pose(
            result_a['corners'], result_a['ids'], board, cam_matrix_a, dist_coeffs_a
        )
        success_b, rvec_b, tvec_b = estimate_pose(
            result_b['corners'], result_b['ids'], board, cam_matrix_b, dist_coeffs_b
        )
        
        if not success_a or not success_b:
            continue
        
        # Compute transformations
        # T_board_to_a: transform from board frame to camera A frame
        R_a = rotation_vector_to_matrix(rvec_a)
        T_board_to_a = create_transform_matrix(R_a, tvec_a.flatten())
        
        # T_board_to_b: transform from board frame to camera B frame
        R_b = rotation_vector_to_matrix(rvec_b)
        T_board_to_b = create_transform_matrix(R_b, tvec_b.flatten())
        
        # T_a_to_b = T_board_to_b @ T_a_to_board = T_board_to_b @ inv(T_board_to_a)
        T_a_to_board = invert_transform(T_board_to_a)
        T_a_to_b = T_board_to_b @ T_a_to_board
        
        transforms.append(T_a_to_b)
    
    if len(transforms) < 3:
        print(f"Pair ({cam_a}, {cam_b}): Only {len(transforms)} valid frames, need at least 3")
        return None
    
    # Average the transforms
    # For translation, simple averaging works well
    # For rotation, we use the Rodrigues representation and average
    avg_translation = np.mean([T[:3, 3] for T in transforms], axis=0)
    
    # For rotation averaging, convert to rotation vectors and average
    rvecs = [cv2.Rodrigues(T[:3, :3])[0].flatten() for T in transforms]
    avg_rvec = np.mean(rvecs, axis=0)
    avg_rotation = rotation_vector_to_matrix(avg_rvec)
    
    T_a_to_b = create_transform_matrix(avg_rotation, avg_translation)
    
    # Compute reprojection error (variance of transforms)
    errors = []
    for T in transforms:
        trans_diff = np.linalg.norm(T[:3, 3] - avg_translation)
        R_diff = T[:3, :3] @ avg_rotation.T
        angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        errors.append(trans_diff * 100 + angle_diff * 180 / np.pi)
    
    avg_error = np.mean(errors)
    
    return PairwiseCalibrationResult(
        camera_a=cam_a,
        camera_b=cam_b,
        transform_a_to_b=T_a_to_b,
        reprojection_error=avg_error,
        num_frames_used=len(transforms),
    )


def chain_pairwise_transforms(
    pairwise_results: List[PairwiseCalibrationResult],
    reference_camera: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Chain pairwise transforms to compute camera poses relative to a reference.
    
    Uses Dijkstra's algorithm to find shortest paths through the camera graph,
    where edge weights are based on reprojection error (lower is better).
    
    Args:
        pairwise_results: List of pairwise calibration results
        reference_camera: Camera to use as origin (auto-selected if None)
        
    Returns:
        Dict mapping camera_id to 4x4 transform (camera_to_world)
    """
    if not pairwise_results:
        return {}
    
    # Build adjacency graph
    # graph[a][b] = (T_a_to_b, error)
    graph: Dict[int, Dict[int, Tuple[np.ndarray, float]]] = defaultdict(dict)
    all_cameras: Set[int] = set()
    
    for result in pairwise_results:
        a, b = result.camera_a, result.camera_b
        all_cameras.add(a)
        all_cameras.add(b)
        
        # Store both directions
        graph[a][b] = (result.transform_a_to_b, result.reprojection_error)
        graph[b][a] = (invert_transform(result.transform_a_to_b), result.reprojection_error)
    
    # Select reference camera (one with most connections)
    if reference_camera is None:
        reference_camera = max(all_cameras, key=lambda c: len(graph[c]))
    
    print(f"Using camera {reference_camera} as reference (origin)")
    
    # Dijkstra's algorithm to find shortest paths
    # distance[cam] = (total_error, transform_ref_to_cam)
    distances: Dict[int, Tuple[float, np.ndarray]] = {}
    distances[reference_camera] = (0.0, np.eye(4))
    
    # Priority queue: (total_error, camera_id)
    pq = [(0.0, reference_camera)]
    visited = set()
    
    while pq:
        current_error, current_cam = heapq.heappop(pq)
        
        if current_cam in visited:
            continue
        visited.add(current_cam)
        
        current_transform = distances[current_cam][1]
        
        for neighbor, (T_curr_to_neighbor, edge_error) in graph[current_cam].items():
            if neighbor in visited:
                continue
            
            new_error = current_error + edge_error
            new_transform = T_curr_to_neighbor @ current_transform
            
            if neighbor not in distances or new_error < distances[neighbor][0]:
                distances[neighbor] = (new_error, new_transform)
                heapq.heappush(pq, (new_error, neighbor))
    
    # Check for disconnected cameras
    disconnected = all_cameras - visited
    if disconnected:
        print(f"Warning: Cameras {disconnected} are not connected to reference camera {reference_camera}")
    
    # Convert to camera_to_world transforms
    # distances[cam] gives ref_to_cam, we need cam_to_world where world = ref
    camera_transforms = {}
    for cam, (error, T_ref_to_cam) in distances.items():
        # cam_to_world = inv(ref_to_cam) since ref = world
        camera_transforms[cam] = invert_transform(T_ref_to_cam)
    
    return camera_transforms


def calibrate_pairwise_extrinsics(
    pairwise_captures: Dict[Tuple[int, int], List[Dict[int, np.ndarray]]],
    intrinsics_map: Dict[int, CameraIntrinsics],
    config: CalibrationConfig,
    reference_camera: Optional[int] = None,
) -> Optional[MultiCameraCalibration]:
    """
    Full pairwise extrinsics calibration pipeline.
    
    Args:
        pairwise_captures: Dict mapping (cam_a, cam_b) to list of frame dicts
        intrinsics_map: Dict mapping camera_id to CameraIntrinsics
        config: Calibration configuration
        reference_camera: Camera to use as origin
        
    Returns:
        MultiCameraCalibration or None if calibration failed
    """
    from datetime import datetime
    
    # Calibrate each pair
    pairwise_results = []
    
    for (cam_a, cam_b), captures in pairwise_captures.items():
        if cam_a not in intrinsics_map or cam_b not in intrinsics_map:
            print(f"Skipping pair ({cam_a}, {cam_b}): missing intrinsics")
            continue
            
        result = calibrate_camera_pair(
            captures,
            intrinsics_map[cam_a],
            intrinsics_map[cam_b],
            config,
        )
        
        if result:
            pairwise_results.append(result)
            print(f"Pair ({cam_a}, {cam_b}): error={result.reprojection_error:.2f}, frames={result.num_frames_used}")
        else:
            print(f"Pair ({cam_a}, {cam_b}): calibration failed")
    
    if not pairwise_results:
        print("No valid pairwise calibrations")
        return None
    
    # Chain transforms
    camera_transforms = chain_pairwise_transforms(pairwise_results, reference_camera)
    
    if not camera_transforms:
        print("Failed to chain transforms")
        return None
    
    # Build MultiCameraCalibration
    calibration = MultiCameraCalibration(
        origin_description=f"Camera {reference_camera} (reference camera position)"
    )
    
    for cam_id, T_cam_to_world in camera_transforms.items():
        if cam_id not in intrinsics_map:
            continue
            
        intrinsics = intrinsics_map[cam_id]
        
        extrinsics = CameraExtrinsics(
            camera_id=cam_id,
            camera_name=intrinsics.camera_name,
            transform_matrix=T_cam_to_world.tolist(),
            rotation_matrix=T_cam_to_world[:3, :3].tolist(),
            translation_vector=T_cam_to_world[:3, 3].tolist(),
            reprojection_error=0.0,  # TODO: compute aggregate error
            calibration_date=datetime.now().isoformat(),
        )
        
        calibration.add_camera(intrinsics, extrinsics)
        print(f"Camera {cam_id}: position = {T_cam_to_world[:3, 3]}")
    
    return calibration


def get_required_pairs(camera_ids: List[int]) -> List[Tuple[int, int]]:
    """
    Get list of camera pairs needed for full connectivity.
    
    For N cameras, we need at least N-1 pairs to form a spanning tree.
    This returns all possible pairs to give flexibility.
    
    Args:
        camera_ids: List of camera IDs
        
    Returns:
        List of (camera_a, camera_b) tuples
    """
    pairs = []
    for i in range(len(camera_ids)):
        for j in range(i + 1, len(camera_ids)):
            pairs.append((camera_ids[i], camera_ids[j]))
    return pairs


def check_connectivity(
    pairwise_results: List[PairwiseCalibrationResult],
    camera_ids: List[int],
) -> Tuple[bool, Set[int]]:
    """
    Check if all cameras are connected through pairwise calibrations.
    
    Args:
        pairwise_results: List of successful pairwise calibrations
        camera_ids: All camera IDs that should be connected
        
    Returns:
        Tuple of (is_connected, set of disconnected cameras)
    """
    if not pairwise_results:
        return False, set(camera_ids)
    
    # Build adjacency list
    adj: Dict[int, Set[int]] = defaultdict(set)
    for result in pairwise_results:
        adj[result.camera_a].add(result.camera_b)
        adj[result.camera_b].add(result.camera_a)
    
    # BFS from first camera with edges
    start_cam = list(adj.keys())[0]
    visited = set()
    queue = [start_cam]
    
    while queue:
        cam = queue.pop(0)
        if cam in visited:
            continue
        visited.add(cam)
        for neighbor in adj[cam]:
            if neighbor not in visited:
                queue.append(neighbor)
    
    all_cameras = set(camera_ids)
    disconnected = all_cameras - visited
    
    return len(disconnected) == 0, disconnected
