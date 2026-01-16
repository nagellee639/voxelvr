"""
Dataset-Based Accuracy Tests

Tests reconstruction accuracy using sample datasets.
Compares reconstructed 3D poses against ground truth.
"""

import pytest
import numpy as np
from pathlib import Path
import json


class TestSyntheticDatasetAccuracy:
    """Test accuracy on synthetic dataset with ground truth."""
    
    @pytest.fixture
    def synthetic_dataset(self, tmp_path):
        """Generate a small synthetic dataset for testing."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from sample_data import SyntheticDataset
        
        output_dir = tmp_path / "test_dataset"
        generator = SyntheticDataset(output_dir)
        generator.generate(
            num_cameras=3,
            duration=1.0,
            fps=30.0,
            pose_type="walking",
            noise_pixels=2.0,
        )
        
        return output_dir
    
    def test_reconstruction_accuracy(self, synthetic_dataset):
        """Test 3D reconstruction accuracy against ground truth."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from sample_data import DatasetLoader
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.detector_2d import Keypoints2D
        
        # Load dataset
        loader = DatasetLoader(synthetic_dataset)
        projection_matrices = loader.get_projection_matrices()
        
        pipeline = TriangulationPipeline(projection_matrices)
        
        errors = []
        valid_counts = []
        
        for frame_data in loader:
            # Build keypoints
            keypoints_list = []
            for cam_id, kp_data in frame_data['keypoints_2d'].items():
                kp = Keypoints2D(
                    positions=np.array(kp_data['keypoints']),
                    confidences=np.array(kp_data['confidences']),
                    image_width=1280,
                    image_height=720,
                    camera_id=cam_id,
                )
                keypoints_list.append(kp)
            
            # Triangulate
            result = pipeline.process(keypoints_list)
            
            if result is None:
                continue
            
            # Compare to ground truth
            gt = frame_data['ground_truth_3d']
            
            for i in range(17):
                if result['valid'][i]:
                    error = np.linalg.norm(result['positions'][i] - gt[i])
                    errors.append(error)
            
            valid_counts.append(np.sum(result['valid']))
        
        errors = np.array(errors)
        
        print(f"\nReconstruction Accuracy:")
        print(f"  Mean error: {np.mean(errors)*100:.2f} cm")
        print(f"  Median error: {np.median(errors)*100:.2f} cm")
        print(f"  95th percentile: {np.percentile(errors, 95)*100:.2f} cm")
        print(f"  Max error: {np.max(errors)*100:.2f} cm")
        print(f"  Avg valid joints: {np.mean(valid_counts):.1f}/17")
        
        # Accuracy requirements
        assert np.mean(errors) < 0.05, f"Mean error too high: {np.mean(errors)*100:.2f}cm"
        assert np.percentile(errors, 95) < 0.10, "95th percentile too high"
    
    def test_temporal_consistency(self, synthetic_dataset):
        """Test that reconstruction is temporally consistent."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from sample_data import DatasetLoader
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.detector_2d import Keypoints2D
        from voxelvr.pose import PoseFilter
        
        loader = DatasetLoader(synthetic_dataset)
        projection_matrices = loader.get_projection_matrices()
        
        pipeline = TriangulationPipeline(projection_matrices)
        pose_filter = PoseFilter(num_joints=17)
        
        prev_pose = None
        jitter_raw = []
        jitter_filtered = []
        
        for frame_data in loader:
            keypoints_list = []
            for cam_id, kp_data in frame_data['keypoints_2d'].items():
                kp = Keypoints2D(
                    positions=np.array(kp_data['keypoints']),
                    confidences=np.array(kp_data['confidences']),
                    image_width=1280,
                    image_height=720,
                    camera_id=cam_id,
                )
                keypoints_list.append(kp)
            
            result = pipeline.process(keypoints_list)
            if result is None:
                continue
            
            # Raw pose
            raw_pose = result['positions'].copy()
            
            # Filtered pose
            filtered_pose = pose_filter.filter(raw_pose, result['valid'])
            
            if prev_pose is not None:
                # Calculate frame-to-frame jitter
                raw_jitter = np.mean(np.linalg.norm(raw_pose - prev_pose, axis=1))
                filtered_jitter = np.mean(np.linalg.norm(filtered_pose - prev_pose, axis=1))
                
                jitter_raw.append(raw_jitter)
                jitter_filtered.append(filtered_jitter)
            
            prev_pose = filtered_pose.copy()
        
        raw_jitter_std = np.std(jitter_raw)
        filtered_jitter_std = np.std(jitter_filtered)
        
        print(f"\nTemporal Consistency:")
        print(f"  Raw jitter std: {raw_jitter_std*100:.3f} cm")
        print(f"  Filtered jitter std: {filtered_jitter_std*100:.3f} cm")
        print(f"  Jitter reduction: {(1-filtered_jitter_std/raw_jitter_std)*100:.1f}%")
        
        # Filter should reduce jitter
        assert filtered_jitter_std < raw_jitter_std, "Filter should reduce jitter"


class TestNoiseLevels:
    """Test accuracy at different noise levels."""
    
    @pytest.mark.slow
    def test_accuracy_vs_noise(self, tmp_path):
        """Test how accuracy degrades with increasing noise."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from sample_data import SyntheticDataset, DatasetLoader
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.detector_2d import Keypoints2D
        
        noise_levels = [0.5, 1.0, 2.0, 5.0, 10.0]
        results = []
        
        for noise in noise_levels:
            # Generate dataset with this noise level
            output_dir = tmp_path / f"noise_{noise}"
            generator = SyntheticDataset(output_dir)
            generator.generate(
                num_cameras=3,
                duration=1.0,
                fps=30.0,
                noise_pixels=noise,
            )
            
            # Test accuracy
            loader = DatasetLoader(output_dir)
            projection_matrices = loader.get_projection_matrices()
            pipeline = TriangulationPipeline(projection_matrices)
            
            errors = []
            for frame_data in loader:
                keypoints_list = []
                for cam_id, kp_data in frame_data['keypoints_2d'].items():
                    kp = Keypoints2D(
                        positions=np.array(kp_data['keypoints']),
                        confidences=np.array(kp_data['confidences']),
                        image_width=1280,
                        image_height=720,
                        camera_id=cam_id,
                    )
                    keypoints_list.append(kp)
                
                result = pipeline.process(keypoints_list)
                if result is None:
                    continue
                
                gt = frame_data['ground_truth_3d']
                for i in range(17):
                    if result['valid'][i]:
                        errors.append(np.linalg.norm(result['positions'][i] - gt[i]))
            
            mean_error = np.mean(errors) * 100  # cm
            results.append((noise, mean_error))
        
        print("\nAccuracy vs Noise Level:")
        print("-" * 30)
        for noise, error in results:
            print(f"  {noise:5.1f} px noise: {error:5.2f} cm error")
        
        # Error should increase with noise
        for i in range(1, len(results)):
            assert results[i][1] >= results[i-1][1] * 0.8, \
                "Error should generally increase with noise"


class TestCameraCount:
    """Test accuracy with different numbers of cameras."""
    
    @pytest.mark.slow
    def test_accuracy_vs_cameras(self, tmp_path):
        """Test how accuracy improves with more cameras."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from sample_data import SyntheticDataset, DatasetLoader
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.detector_2d import Keypoints2D
        
        camera_counts = [2, 3, 4, 5, 6]
        results = []
        
        for num_cams in camera_counts:
            output_dir = tmp_path / f"cams_{num_cams}"
            generator = SyntheticDataset(output_dir)
            generator.generate(
                num_cameras=num_cams,
                duration=1.0,
                fps=30.0,
                noise_pixels=2.0,
            )
            
            loader = DatasetLoader(output_dir)
            projection_matrices = loader.get_projection_matrices()
            pipeline = TriangulationPipeline(projection_matrices)
            
            errors = []
            for frame_data in loader:
                keypoints_list = []
                for cam_id, kp_data in frame_data['keypoints_2d'].items():
                    kp = Keypoints2D(
                        positions=np.array(kp_data['keypoints']),
                        confidences=np.array(kp_data['confidences']),
                        image_width=1280,
                        image_height=720,
                        camera_id=cam_id,
                    )
                    keypoints_list.append(kp)
                
                result = pipeline.process(keypoints_list)
                if result is None:
                    continue
                
                gt = frame_data['ground_truth_3d']
                for i in range(17):
                    if result['valid'][i]:
                        errors.append(np.linalg.norm(result['positions'][i] - gt[i]))
            
            mean_error = np.mean(errors) * 100
            results.append((num_cams, mean_error))
        
        print("\nAccuracy vs Camera Count:")
        print("-" * 30)
        for num_cams, error in results:
            print(f"  {num_cams} cameras: {error:5.2f} cm error")
        
        # More cameras should generally improve accuracy
        # (allowing for some noise in measurements)
        assert results[-1][1] <= results[0][1] * 1.2, \
            "More cameras should not significantly hurt accuracy"
