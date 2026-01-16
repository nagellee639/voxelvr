#!/usr/bin/env python3
"""
Test Image Generator

Generates synthetic images for testing VoxelVR without physical cameras.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voxelvr.calibration.charuco import generate_charuco_img

def generate_warped_board(
    width: int = 1280,
    height: int = 720,
    yaw: float = 0,
    pitch: float = 0,
    roll: float = 0,
    scale: float = 0.5,
    distance: float = 1.0,
) -> np.ndarray:
    """
    Generate a Charuco board image with perspective warp.
    Using simple 2D perspective transform for now.
    """
    # 1. Generate base board
    # Increase DPI/size for quality before downscaling
    board_img = generate_charuco_img(
        squares_x=5, squares_y=5, square_length=0.04, marker_length=0.03, dpi=200
    )
    h_board, w_board = board_img.shape[:2]
    
    # 2. Define source points (corners of the board image)
    src_pts = np.float32([
        [0, 0],
        [w_board, 0],
        [w_board, h_board],
        [0, h_board]
    ])
    
    # 3. Define destination points (simulating 3D projection)
    # Center of target image
    cx, cy = width / 2, height / 2
    
    # Random-ish warping based on angles (simulated)
    # For simplicity, we just move corners around to create perspective
    
    # Base half-size scaled
    hw = (w_board * scale) / 2
    hh = (h_board * scale) / 2
    
    # Apply some perspective distortion
    # "Pitch" -> top narrower than bottom
    pitch_factor = pitch * 100 # pixels indent
    
    dst_pts = np.float32([
        [cx - hw + pitch_factor, cy - hh + pitch_factor],  # Top Left
        [cx + hw - pitch_factor, cy - hh + pitch_factor],  # Top Right
        [cx + hw, cy + hh],              # Bottom Right
        [cx - hw, cy + hh]               # Bottom Left
    ])
    
    # Add offset (simulating position)
    offset_x = np.random.uniform(-200, 200)
    offset_y = np.random.uniform(-100, 100)
    dst_pts[:, 0] += offset_x
    dst_pts[:, 1] += offset_y
    
    # 4. Warp
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Start with gray background
    bg = np.ones((height, width, 3), dtype=np.uint8) * 100
    
    warped = cv2.warpPerspective(board_img, M, (width, height), borderValue=(100, 100, 100))
    
    return warped

def main():
    output_dir = Path("test_data/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating calibration images in {output_dir}...")
    
    # Generate 10 images with different "angles"
    for i in range(10):
        # Random pitch/distort
        pitch = np.random.uniform(0.1, 0.4)
        scale = np.random.uniform(0.3, 0.6)
        
        img = generate_warped_board(pitch=pitch, scale=scale)
        
        filename = output_dir / f"calib_{i:02d}.jpg"
        cv2.imwrite(str(filename), img)
        print(f"  Saved {filename}")
        
    print("Done.")

if __name__ == "__main__":
    main()
