#!/usr/bin/env python3
"""
Sample Dataset Downloader and Generator

Downloads small sample datasets for testing the tracking pipeline
without real cameras.

Datasets:
1. Synthetic - Generated programmatically (no download needed)
2. CMU Panoptic - Subset of CMU Panoptic Dataset (requires download)
3. Shelf/Campus - Classic multi-view pose datasets (requires download)

Usage:
    python scripts/sample_data.py --generate       # Generate synthetic data
    python scripts/sample_data.py --list           # List available datasets
    python scripts/sample_data.py --download cmu   # Download CMU sample
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import zipfile
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import (
    generate_t_pose,
    generate_walking_pose,
    generate_multi_camera_setup,
    project_3d_to_2d,
    generate_camera_intrinsics,
)


class SyntheticDataset:
    """
    Generate synthetic multi-view pose data for testing.
    
    No download required - generates programmatically.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        num_cameras: int = 3,
        duration: float = 10.0,
        fps: float = 30.0,
        pose_type: str = "walking",
        noise_pixels: float = 2.0,
    ) -> Dict:
        """
        Generate synthetic dataset.
        
        Args:
            num_cameras: Number of virtual cameras
            duration: Duration in seconds
            fps: Frames per second
            pose_type: "walking", "t_pose", "random"
            noise_pixels: Gaussian noise in pixel coordinates
            
        Returns:
            Dictionary with dataset metadata
        """
        print(f"Generating synthetic dataset...")
        print(f"  Cameras: {num_cameras}")
        print(f"  Duration: {duration}s @ {fps} FPS")
        print(f"  Pose type: {pose_type}")
        
        # Setup cameras
        cameras = generate_multi_camera_setup(num_cameras)
        
        # Save camera parameters
        camera_dir = self.output_dir / "cameras"
        camera_dir.mkdir(exist_ok=True)
        
        for cam in cameras:
            cam_data = {
                'id': cam['id'],
                'intrinsics': cam['intrinsics'].tolist(),
                'extrinsics': cam['extrinsics'].tolist(),
                'position': cam['position'].tolist(),
                'width': 1280,
                'height': 720,
            }
            
            with open(camera_dir / f"camera_{cam['id']}.json", 'w') as f:
                json.dump(cam_data, f, indent=2)
        
        # Generate frames
        num_frames = int(duration * fps)
        
        poses_3d = []
        keypoints_2d = {cam['id']: [] for cam in cameras}
        
        for frame_idx in range(num_frames):
            t = frame_idx / fps
            
            # Generate 3D pose
            if pose_type == "walking":
                pose = generate_walking_pose(t)
            elif pose_type == "t_pose":
                pose = generate_t_pose()
            else:
                pose = generate_walking_pose(t) + np.random.normal(0, 0.05, (17, 3))
            
            poses_3d.append(pose.tolist())
            
            # Project to each camera
            for cam in cameras:
                p2d, visibility = project_3d_to_2d(
                    pose,
                    cam['intrinsics'],
                    cam['extrinsics'],
                    add_noise=noise_pixels,
                )
                
                keypoints_2d[cam['id']].append({
                    'frame': frame_idx,
                    'timestamp': t,
                    'keypoints': p2d.tolist(),
                    'visibility': visibility.tolist(),
                    'confidences': (visibility.astype(float) * 0.9).tolist(),
                })
        
        # Save ground truth poses
        gt_path = self.output_dir / "ground_truth_3d.json"
        with open(gt_path, 'w') as f:
            json.dump({
                'num_frames': num_frames,
                'fps': fps,
                'poses': poses_3d,
            }, f)
        
        # Save 2D keypoints for each camera
        for cam_id, kp_list in keypoints_2d.items():
            kp_path = self.output_dir / f"keypoints_2d_cam{cam_id}.json"
            with open(kp_path, 'w') as f:
                json.dump({
                    'camera_id': cam_id,
                    'num_frames': num_frames,
                    'keypoints': kp_list,
                }, f)
        
        # Generate metadata
        metadata = {
            'name': 'synthetic',
            'type': pose_type,
            'num_cameras': num_cameras,
            'num_frames': num_frames,
            'fps': fps,
            'duration': duration,
            'noise_pixels': noise_pixels,
            'has_ground_truth': True,
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved to: {self.output_dir}")
        print(f"  Frames: {num_frames}")
        print(f"  Ground truth: {gt_path}")
        
        return metadata


class DatasetLoader:
    """Load and iterate through a sample dataset."""
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        
        # Load metadata
        with open(self.dataset_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        # Load cameras
        self.cameras = {}
        camera_dir = self.dataset_dir / "cameras"
        for cam_file in camera_dir.glob("camera_*.json"):
            with open(cam_file) as f:
                cam_data = json.load(f)
                self.cameras[cam_data['id']] = cam_data
        
        # Load ground truth if available
        gt_path = self.dataset_dir / "ground_truth_3d.json"
        if gt_path.exists():
            with open(gt_path) as f:
                self.ground_truth = json.load(f)
        else:
            self.ground_truth = None
        
        # Load 2D keypoints
        self.keypoints_2d = {}
        for cam_id in self.cameras.keys():
            kp_path = self.dataset_dir / f"keypoints_2d_cam{cam_id}.json"
            if kp_path.exists():
                with open(kp_path) as f:
                    self.keypoints_2d[cam_id] = json.load(f)
    
    def __len__(self) -> int:
        return self.metadata['num_frames']
    
    def __iter__(self):
        for frame_idx in range(len(self)):
            yield self.get_frame(frame_idx)
    
    def get_frame(self, frame_idx: int) -> Dict:
        """
        Get all data for a single frame.
        
        Returns:
            Dictionary with 2D keypoints per camera and ground truth 3D pose
        """
        frame_data = {
            'frame_idx': frame_idx,
            'keypoints_2d': {},
            'ground_truth_3d': None,
        }
        
        # 2D keypoints
        for cam_id, kp_data in self.keypoints_2d.items():
            frame_data['keypoints_2d'][cam_id] = kp_data['keypoints'][frame_idx]
        
        # Ground truth
        if self.ground_truth is not None:
            frame_data['ground_truth_3d'] = np.array(self.ground_truth['poses'][frame_idx])
        
        return frame_data
    
    def get_projection_matrices(self) -> Dict[int, np.ndarray]:
        """Get projection matrices for all cameras."""
        projection_matrices = {}
        
        for cam_id, cam_data in self.cameras.items():
            K = np.array(cam_data['intrinsics'])
            T = np.linalg.inv(np.array(cam_data['extrinsics']))
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam_id] = P
        
        return projection_matrices


def download_cmu_sample(output_dir: Path) -> bool:
    """
    Download a small sample from CMU Panoptic dataset.
    
    Note: Full dataset is very large. This downloads only calibration
    and a few frames for testing.
    """
    print("CMU Panoptic sample download is not implemented.")
    print("For full dataset, visit: http://domedb.perception.cs.cmu.edu/")
    print("\nAlternatively, use synthetic data for testing:")
    print("  python scripts/sample_data.py --generate")
    return False


def main():
    parser = argparse.ArgumentParser(description="Sample Dataset Manager")
    parser.add_argument('--generate', action='store_true',
                       help='Generate synthetic dataset')
    parser.add_argument('--output', '-o', type=Path,
                       default=Path("data/synthetic"),
                       help='Output directory')
    parser.add_argument('--cameras', '-c', type=int, default=3,
                       help='Number of cameras (for synthetic)')
    parser.add_argument('--duration', '-d', type=float, default=5.0,
                       help='Duration in seconds')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Frames per second')
    parser.add_argument('--noise', type=float, default=2.0,
                       help='2D noise in pixels')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    parser.add_argument('--download', type=str, choices=['cmu', 'shelf'],
                       help='Download sample dataset')
    parser.add_argument('--test', action='store_true',
                       help='Test loading a dataset')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        print("  - synthetic: Programmatically generated (no download)")
        print("  - cmu: CMU Panoptic subset (requires download)")
        print("  - shelf: Shelf dataset (requires download)")
        return 0
    
    if args.generate:
        generator = SyntheticDataset(args.output)
        generator.generate(
            num_cameras=args.cameras,
            duration=args.duration,
            fps=args.fps,
            noise_pixels=args.noise,
        )
        return 0
    
    if args.download:
        if args.download == 'cmu':
            success = download_cmu_sample(args.output)
        else:
            print(f"Download not implemented for: {args.download}")
            success = False
        return 0 if success else 1
    
    if args.test:
        if not args.output.exists():
            print(f"Dataset not found at {args.output}")
            print("Run with --generate first")
            return 1
        
        loader = DatasetLoader(args.output)
        print(f"Dataset: {loader.metadata.get('name', 'unknown')}")
        print(f"Frames: {len(loader)}")
        print(f"Cameras: {list(loader.cameras.keys())}")
        
        # Test loading first frame
        frame = loader.get_frame(0)
        print(f"Frame 0: {len(frame['keypoints_2d'])} camera views")
        
        if frame['ground_truth_3d'] is not None:
            print(f"Ground truth shape: {frame['ground_truth_3d'].shape}")
        
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
