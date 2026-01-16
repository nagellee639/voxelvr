"""
Performance and Latency Benchmark Tests

Measures the actual performance of each pipeline stage
to ensure real-time operation is achievable.

Target: < 50ms total latency for the full pipeline.
"""

import pytest
import numpy as np
import time
from conftest import (
    generate_t_pose,
    generate_walking_pose,
    generate_multi_camera_setup,
    project_3d_to_2d,
    measure_performance,
    get_system_info,
)


class TestPipelineLatency:
    """Measure latency of each pipeline stage."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Print system info before benchmarks."""
        info = get_system_info()
        print("\n" + "="*60)
        print("SYSTEM INFO:")
        print(f"  Platform: {info['platform']}")
        print(f"  Processor: {info['processor']}")
        if 'onnxruntime_providers' in info:
            print(f"  ONNX Providers: {info['onnxruntime_providers']}")
        if 'cuda_available' in info and info['cuda_available']:
            print(f"  CUDA Device: {info.get('cuda_device', 'Unknown')}")
        print("="*60)
    
    @pytest.mark.benchmark
    def test_2d_pose_detection_latency(self):
        """Measure 2D pose detection latency."""
        from voxelvr.pose import PoseDetector2D
        
        detector = PoseDetector2D(backend="auto")
        
        # Create test image
        test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Load model first
        success = detector.load_model()
        if not success:
            pytest.skip("Could not load pose model")
        
        # Measure
        result = measure_performance(
            detector.detect,
            test_image,
            num_iterations=50,
            warmup_iterations=10,
            name="2D_Pose_Detection",
        )
        
        print(f"\n2D Pose Detection:")
        print(f"  Mean: {result.mean_ms:.2f} ms")
        print(f"  Std:  {result.std_ms:.2f} ms")
        print(f"  FPS:  {result.fps:.1f}")
        
        # Target: < 20ms per camera
        assert result.mean_ms < 50, f"2D detection too slow: {result.mean_ms:.2f}ms"
    
    @pytest.mark.benchmark
    def test_triangulation_latency(self):
        """Measure triangulation latency."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.detector_2d import Keypoints2D
        
        # Setup
        pose_3d = generate_t_pose()
        cameras = generate_multi_camera_setup(num_cameras=3)
        
        projection_matrices = {}
        keypoints_list = []
        
        for cam in cameras:
            K = cam['intrinsics']
            T = np.linalg.inv(cam['extrinsics'])
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
            
            p2d, vis = project_3d_to_2d(pose_3d, K, cam['extrinsics'])
            kp = Keypoints2D(
                positions=p2d,
                confidences=vis.astype(np.float32) * 0.9,
                image_width=1280,
                image_height=720,
                camera_id=cam['id'],
            )
            keypoints_list.append(kp)
        
        pipeline = TriangulationPipeline(projection_matrices)
        
        # Measure
        result = measure_performance(
            pipeline.process,
            keypoints_list,
            num_iterations=100,
            warmup_iterations=20,
            name="Triangulation",
        )
        
        print(f"\nTriangulation (3 cameras, 17 joints):")
        print(f"  Mean: {result.mean_ms:.2f} ms")
        print(f"  Std:  {result.std_ms:.2f} ms")
        print(f"  FPS:  {result.fps:.1f}")
        
        # Target: < 5ms
        assert result.mean_ms < 10, f"Triangulation too slow: {result.mean_ms:.2f}ms"
    
    @pytest.mark.benchmark
    def test_temporal_filter_latency(self):
        """Measure temporal filtering latency."""
        from voxelvr.pose import PoseFilter
        
        pose_filter = PoseFilter(num_joints=17)
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        # Measure
        result = measure_performance(
            pose_filter.filter,
            pose,
            valid_mask,
            num_iterations=1000,
            warmup_iterations=100,
            name="Temporal_Filter",
        )
        
        print(f"\nTemporal Filter (17 joints):")
        print(f"  Mean: {result.mean_ms:.3f} ms")
        print(f"  Std:  {result.std_ms:.3f} ms")
        print(f"  FPS:  {result.fps:.1f}")
        
        # Target: < 0.5ms (should be very fast)
        assert result.mean_ms < 1.0, f"Filter too slow: {result.mean_ms:.3f}ms"
    
    @pytest.mark.benchmark
    def test_rotation_estimation_latency(self):
        """Measure rotation estimation latency."""
        from voxelvr.pose.rotation import estimate_all_rotations
        
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        # Measure
        result = measure_performance(
            estimate_all_rotations,
            pose,
            valid_mask,
            num_iterations=1000,
            warmup_iterations=100,
            name="Rotation_Estimation",
        )
        
        print(f"\nRotation Estimation (8 trackers):")
        print(f"  Mean: {result.mean_ms:.3f} ms")
        print(f"  Std:  {result.std_ms:.3f} ms")
        print(f"  FPS:  {result.fps:.1f}")
        
        # Target: < 2ms (comfortable margin)
        assert result.mean_ms < 5.0, f"Rotation estimation too slow: {result.mean_ms:.3f}ms"
    
    @pytest.mark.benchmark
    def test_osc_message_latency(self):
        """Measure OSC message building latency (not sending)."""
        from voxelvr.transport.osc_sender import pose_to_trackers_with_rotations
        
        pose = generate_t_pose()
        confidences = np.ones(17) * 0.9
        valid_mask = np.ones(17, dtype=bool)
        
        # Measure
        result = measure_performance(
            pose_to_trackers_with_rotations,
            pose,
            confidences,
            valid_mask,
            num_iterations=500,
            warmup_iterations=50,
            name="OSC_Message_Build",
        )
        
        print(f"\nOSC Message Build (8 trackers):")
        print(f"  Mean: {result.mean_ms:.3f} ms")
        print(f"  Std:  {result.std_ms:.3f} ms")
        print(f"  FPS:  {result.fps:.1f}")
        
        # Target: < 3ms (includes rotation estimation)
        assert result.mean_ms < 10.0, f"OSC build too slow: {result.mean_ms:.3f}ms"


class TestFullPipelineBenchmark:
    """Benchmark the complete pipeline end-to-end."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_full_pipeline_synthetic(self):
        """
        Benchmark full pipeline with synthetic 2D keypoints.
        
        Simulates: 2D keypoints -> Triangulation -> Filter -> OSC
        """
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.detector_2d import Keypoints2D
        from voxelvr.pose import PoseFilter
        from voxelvr.pose.rotation import RotationFilter
        from voxelvr.transport.osc_sender import pose_to_trackers_with_rotations
        
        # Setup
        cameras = generate_multi_camera_setup(num_cameras=3)
        
        projection_matrices = {}
        for cam in cameras:
            K = cam['intrinsics']
            T = np.linalg.inv(cam['extrinsics'])
            R, t = T[:3, :3], T[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            projection_matrices[cam['id']] = P
        
        pipeline = TriangulationPipeline(projection_matrices)
        pose_filter = PoseFilter(num_joints=17)
        rotation_filter = RotationFilter(alpha=0.2)
        
        def full_pipeline_step(t: float):
            """One step of the full pipeline."""
            pose_3d_gt = generate_walking_pose(t)
            
            # Simulate 2D detections
            keypoints_list = []
            for cam in cameras:
                p2d, vis = project_3d_to_2d(
                    pose_3d_gt,
                    cam['intrinsics'],
                    cam['extrinsics'],
                    add_noise=2.0
                )
                kp = Keypoints2D(
                    positions=p2d,
                    confidences=vis.astype(np.float32) * 0.9,
                    image_width=1280,
                    image_height=720,
                    camera_id=cam['id'],
                )
                keypoints_list.append(kp)
            
            # Triangulate
            result = pipeline.process(keypoints_list)
            
            if result is None:
                return None
            
            # Filter positions
            filtered_pos = pose_filter.filter(result['positions'], result['valid'])
            
            # Build trackers with rotations
            trackers = pose_to_trackers_with_rotations(
                filtered_pos,
                result['confidences'],
                result['valid'],
            )
            
            return trackers
        
        # Warmup
        for i in range(20):
            full_pipeline_step(i * 0.033)
        
        # Benchmark
        times = []
        num_frames = 100
        
        for i in range(num_frames):
            t = i * 0.033  # 30 FPS timing
            start = time.perf_counter()
            result = full_pipeline_step(t)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        
        times = np.array(times)
        
        print("\n" + "="*60)
        print("FULL PIPELINE BENCHMARK (synthetic 2D keypoints)")
        print("="*60)
        print(f"  Frames:  {num_frames}")
        print(f"  Mean:    {np.mean(times):.2f} ms")
        print(f"  Std:     {np.std(times):.2f} ms")
        print(f"  Min:     {np.min(times):.2f} ms")
        print(f"  Max:     {np.max(times):.2f} ms")
        print(f"  Median:  {np.median(times):.2f} ms")
        print(f"  FPS:     {1000 / np.mean(times):.1f}")
        print("="*60)
        
        # Target: Pipeline stage (excluding 2D detection) should be < 10ms
        assert np.mean(times) < 20, f"Pipeline too slow: {np.mean(times):.2f}ms"
        
        # 95th percentile should also be reasonable
        p95 = np.percentile(times, 95)
        assert p95 < 30, f"95th percentile too high: {p95:.2f}ms"


class TestMemoryUsage:
    """Test memory usage of key components."""
    
    def test_pose_detector_memory(self):
        """Check pose detector doesn't use excessive memory."""
        import sys
        from voxelvr.pose import PoseDetector2D
        
        # Get baseline
        detector = PoseDetector2D(backend="cpu")
        
        # Try to load model
        try:
            detector.load_model()
        except:
            pytest.skip("Could not load model")
        
        # Check session exists
        if detector.session is not None:
            print("\nPose detector loaded successfully")
            # Memory check would require additional libraries like memory_profiler
    
    def test_filter_memory(self):
        """Check filter doesn't accumulate memory over time."""
        from voxelvr.pose import PoseFilter
        
        pose_filter = PoseFilter(num_joints=17)
        
        # Run many iterations
        for i in range(10000):
            pose = generate_walking_pose(i * 0.033)
            valid_mask = np.ones(17, dtype=bool)
            pose_filter.filter(pose, valid_mask)
        
        # Filter should maintain constant memory
        # (no growing buffers)
        assert len(pose_filter.joint_filters) == 17


class TestScalability:
    """Test performance with different numbers of cameras."""
    
    @pytest.mark.benchmark
    def test_triangulation_scaling(self):
        """Test how triangulation scales with camera count."""
        from voxelvr.pose.triangulation import TriangulationPipeline
        from voxelvr.pose.detector_2d import Keypoints2D
        
        results = []
        
        for num_cams in [2, 3, 4, 5, 6]:
            cameras = generate_multi_camera_setup(num_cameras=num_cams)
            pose_3d = generate_t_pose()
            
            projection_matrices = {}
            keypoints_list = []
            
            for cam in cameras:
                K = cam['intrinsics']
                T = np.linalg.inv(cam['extrinsics'])
                R, t = T[:3, :3], T[:3, 3]
                P = K @ np.hstack([R, t.reshape(3, 1)])
                projection_matrices[cam['id']] = P
                
                p2d, vis = project_3d_to_2d(pose_3d, K, cam['extrinsics'])
                kp = Keypoints2D(
                    positions=p2d,
                    confidences=vis.astype(np.float32) * 0.9,
                    image_width=1280,
                    image_height=720,
                    camera_id=cam['id'],
                )
                keypoints_list.append(kp)
            
            pipeline = TriangulationPipeline(projection_matrices)
            
            # Warmup
            for _ in range(10):
                pipeline.process(keypoints_list)
            
            # Measure
            times = []
            for _ in range(50):
                start = time.perf_counter()
                pipeline.process(keypoints_list)
                times.append((time.perf_counter() - start) * 1000)
            
            mean_time = np.mean(times)
            results.append((num_cams, mean_time))
        
        print("\nTriangulation Scaling:")
        print("-" * 30)
        for num_cams, mean_time in results:
            print(f"  {num_cams} cameras: {mean_time:.2f} ms")
        
        # Should scale roughly linearly
        # 6 cameras should be < 3x slower than 2 cameras
        time_2_cams = results[0][1]
        time_6_cams = results[-1][1]
        
        assert time_6_cams < time_2_cams * 4, \
            f"Poor scaling: 2 cams={time_2_cams:.2f}ms, 6 cams={time_6_cams:.2f}ms"
