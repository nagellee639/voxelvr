"""
World Frame Alignment

Provides functionality to align the calibrated camera cluster to a meaningful
world coordinate system (e.g., Y-up, floor at Y=0) for VRChat compatibility.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..config import MultiCameraCalibration, CameraExtrinsics
from .extrinsics import create_transform_matrix, invert_transform

def align_calibration(calibration: MultiCameraCalibration) -> MultiCameraCalibration:
    """
    Align the calibration to a logical world frame.
    
    1. Estimates 'Up' vector by checking if cameras have consistent orientation.
    2. If cameras are tilted at different angles, skip up-alignment and just center.
    3. Centers the origin (X,Z) on the camera centroid.
    
    Returns:
        New MultiCameraCalibration object with adjusted transforms.
    """
    cameras = calibration.cameras
    if not cameras:
        return calibration
        
    # 1. Collect camera data
    up_vectors = []
    positions = []
    
    for cam_id, data in cameras.items():
        if 'extrinsics' not in data:
            continue
            
        T_cam_to_world_curr = np.array(data['extrinsics']['transform_matrix'])
        positions.append(T_cam_to_world_curr[:3, 3])
        
        # Camera Up (-Y) in World Frame
        R = T_cam_to_world_curr[:3, :3]
        cam_up = R @ np.array([0, -1, 0])
        up_vectors.append(cam_up)
        
    if not up_vectors or len(positions) < 2:
        return calibration
    
    # 2. Check if camera up vectors are consistent
    # If cameras are tilted at very different angles, their up vectors won't align
    up_vectors_arr = np.array(up_vectors)
    
    # Pairwise dot products
    consistent = True
    for i in range(len(up_vectors)):
        for j in range(i+1, len(up_vectors)):
            dot = np.abs(np.dot(up_vectors[i], up_vectors[j]))
            if dot < 0.7:  # Less than ~45 degrees agreement
                consistent = False
                break
        if not consistent:
            break
    
    if consistent:
        # Cameras are mostly upright - use average up vector
        avg_up = np.mean(up_vectors_arr, axis=0)
        avg_up = avg_up / np.linalg.norm(avg_up)
        print(f"Cameras are consistently oriented. Up Vector: {avg_up}")
    else:
        # Cameras are tilted differently - fall back to gravity assumption
        # Use the average camera up but warn user
        avg_up = np.mean(up_vectors_arr, axis=0)
        avg_up_norm = np.linalg.norm(avg_up)
        
        if avg_up_norm < 0.3:
            # Up vectors cancel out - cameras facing random directions
            # Don't try to align, just center
            print("WARNING: Cameras tilted in conflicting directions - skipping up-alignment")
            avg_up = np.array([0, 1, 0])  # Default to Y-up
            # Skip rotation, just translate
            R_align = np.eye(3)
        else:
            avg_up = avg_up / avg_up_norm
            print(f"WARNING: Cameras tilted at different angles. Estimated Up: {avg_up}")
            print("  Tip: Use Post-Calibration button to align after tracking starts!")
    
    print(f"Estimated World Up Vector: {avg_up}")
    
    # 3. Compute Rotation to align Estimated Up with World Y [0, 1, 0]
    target_up = np.array([0, 1, 0])
    
    # Rotation that moves avg_up to target_up
    axis = np.cross(avg_up, target_up)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(avg_up, target_up)
    
    if sin_angle < 1e-6:
        # Already aligned or opposite
        if cos_angle > 0:
            R_align = np.eye(3)
        else:
            # 180 deg flip
            R_align = np.diag([1, -1, 1]) 
    else:
        axis = axis / sin_angle
        angle = np.arctan2(sin_angle, cos_angle)
        
        # Rodrigues formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
    # 4. Determine Origin Translation
    # Project positions to new frame
    positions_aligned = (R_align @ np.array(positions).T).T
    
    # Center X, Z
    centroid = np.mean(positions_aligned, axis=0)
    center_x = centroid[0]
    center_z = centroid[2]
    
    # Estimate Floor Y
    # Use centroid Y - 1.5m as floor (assumes cameras at roughly eye height)
    floor_y = centroid[1] - 1.5
    
    offset = np.array([center_x, floor_y, center_z])
    
    # T_w_to_wp: Transform from old world to new world
    T_w_to_wp = np.eye(4)
    T_w_to_wp[:3, :3] = R_align
    T_w_to_wp[:3, 3] = -offset
    
    # Apply to all cameras
    new_calibration = MultiCameraCalibration(
        origin_description="Auto-aligned to gravity and camera centroid"
    )
    
    for cam_id, data in cameras.items():
        if 'extrinsics' not in data:
            continue
            
        T_c_to_w = np.array(data['extrinsics']['transform_matrix'])
        T_c_to_wp = T_w_to_wp @ T_c_to_w
        
        # Extract new components
        R_new = T_c_to_wp[:3, :3]
        t_new = T_c_to_wp[:3, 3]
        
        # Reconstruct objects
        from ..config import CameraIntrinsics
        intrin_obj = CameraIntrinsics(**data['intrinsics'])
        
        extrin_obj = CameraExtrinsics(
            camera_id=cam_id,
            camera_name=intrin_obj.camera_name,
            transform_matrix=T_c_to_wp.tolist(),
            rotation_matrix=R_new.tolist(),
            translation_vector=t_new.tolist(),
            reprojection_error=data['extrinsics'].get('reprojection_error', 0.0),
            calibration_date=data['extrinsics']['calibration_date']
        )
        
        new_calibration.add_camera(intrin_obj, extrin_obj)
        
    return new_calibration

