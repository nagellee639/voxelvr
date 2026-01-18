#!/usr/bin/env python3
"""
Render 2D Tracking Overlay

Reads frames from a dataset camera, runs pose detection,
and renders a video with the keypoints overlaid.
"""

import cv2
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

from voxelvr.pose.detector_2d import PoseDetector2D
from voxelvr.config import VoxelVRConfig

def main():
    parser = argparse.ArgumentParser(description="Render 2D tracking overlay")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to tracking dataset root")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID to render")
    parser.add_argument("--output", type=str, default="overlay_video.mp4", help="Output video filename")
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found")
        return 1
        
    cam_dir = dataset_path / f"cam_{args.camera}"
    if not cam_dir.exists():
        print(f"Error: Camera directory {cam_dir} not found")
        # Try finding any camera
        avail = [d.name for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("cam_")]
        print(f"Available cameras: {avail}")
        return 1
        
    files = sorted(list(cam_dir.glob("*.jpg")))
    if not files:
        print(f"No images found in {cam_dir}")
        return 1
        
    print(f"Processing {len(files)} frames from Camera {args.camera}")
    
    # Initialize Detector
    config = VoxelVRConfig.load()
    detector = PoseDetector2D(confidence_threshold=config.tracking.confidence_threshold)
    if not detector.load_model():
        print("Failed to load pose model")
        return 1
        
    # Read first frame to get size
    first_frame = cv2.imread(str(files[0]))
    if first_frame is None:
        print("Failed to read first frame")
        return 1
        
    height, width = first_frame.shape[:2]
    
    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30.0, (width, height))
    
    print(f"Rendering to {args.output}...")
    
    for file_path in tqdm(files):
        frame = cv2.imread(str(file_path))
        if frame is None:
            continue
            
        # Detect
        result = detector.detect(frame)
        
        # Draw
        if result:
            vis = detector.draw_keypoints(frame, result)
        else:
            vis = frame
            
        out.write(vis)
        
    out.release()
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
