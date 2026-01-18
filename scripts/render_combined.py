#!/usr/bin/env python3
"""
Render Combined 2x2 Tracking Overlay

Reads frames from multiple dataset cameras, runs pose detection,
and renders a 2x2 grid video.
"""

import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from voxelvr.pose.detector_2d import PoseDetector2D
from voxelvr.config import VoxelVRConfig

def main():
    parser = argparse.ArgumentParser(description="Render combined 2x2 tracking overlay")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to tracking dataset root")
    parser.add_argument("--output", type=str, default="combined_tracking.mp4", help="Output video filename")
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found")
        return 1
        
    # Find all camera directories
    cam_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("cam_")])
    if not cam_dirs:
        print("No camera directories found")
        return 1
    
    print(f"Found {len(cam_dirs)} cameras: {[d.name for d in cam_dirs]}")
    
    # Collect files
    cam_files = {}
    for d in cam_dirs:
        cam_id = int(d.name.split("_")[1])
        files = sorted(list(d.glob("*.jpg")))
        cam_files[cam_id] = files
        
    # Use first 4 cameras
    selected_ids = list(cam_files.keys())[:4]
    
    # Sync length
    min_len = min(len(cam_files[cid]) for cid in selected_ids)
    print(f"Processing {min_len} frames across {len(selected_ids)} cameras")
    
    # Initialize Detector
    config = VoxelVRConfig.load()
    detector = PoseDetector2D(confidence_threshold=config.tracking.confidence_threshold)
    if not detector.load_model():
        print("Failed to load pose model")
        return 1
        
    # Get frame size from first image
    first_img = cv2.imread(str(cam_files[selected_ids[0]][0]))
    h, w, _ = first_img.shape
    
    # Target grid size: 2x2
    # Output size = same as input (scaled down) or 2x input?
    # Let's keep resolution reasonable: 1920x1080 output
    # Grid: 960x540 per cell
    
    target_w = 960
    target_h = int(target_w * h / w)
    
    out_w = target_w * 2
    out_h = target_h * 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, (out_w, out_h))
    
    print(f"Rendering to {args.output} ({out_w}x{out_h})...")
    
    for i in tqdm(range(min_len)):
        grid_frames = []
        
        for cid in selected_ids:
            # Read
            fpath = cam_files[cid][i]
            frame = cv2.imread(str(fpath))
            
            if frame is None:
                # Black frame
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Detect
            result = detector.detect(frame)
            if result:
                vis = detector.draw_keypoints(frame, result)
            else:
                vis = frame
                
            # Resize
            vis_small = cv2.resize(vis, (target_w, target_h))
            
            # Add label
            cv2.putText(vis_small, f"Cam {cid}", (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            grid_frames.append(vis_small)
            
        # Fill remaining slots if < 4
        while len(grid_frames) < 4:
            grid_frames.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
            
        # Stitch
        # Top row: 0, 1
        # Bottom row: 2, 3
        top = np.hstack((grid_frames[0], grid_frames[1]))
        bot = np.hstack((grid_frames[2], grid_frames[3]))
        full = np.vstack((top, bot))
        
        out.write(full)
        
    out.release()
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
