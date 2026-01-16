"""
Python Overhead Analysis

Measures Python-specific overhead to identify candidates for C++ reimplementation.
Compares pure Python/NumPy operations against theoretical minimum times.
"""

import pytest
import numpy as np
import time
from conftest import generate_t_pose, measure_performance


class TestPythonOverhead:
    """Measure Python-specific overhead in key operations."""
    
    def test_function_call_overhead(self):
        """Measure overhead of Python function calls."""
        
        def empty_function():
            pass
        
        def function_with_args(a, b, c):
            return a
        
        # Measure empty function call
        iterations = 100000
        
        start = time.perf_counter()
        for _ in range(iterations):
            empty_function()
        empty_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Measure function with args
        start = time.perf_counter()
        for _ in range(iterations):
            function_with_args(1, 2, 3)
        args_time = (time.perf_counter() - start) * 1000 / iterations
        
        print("\nPython Function Call Overhead:")
        print(f"  Empty function: {empty_time * 1000:.3f} µs/call")
        print(f"  With 3 args:    {args_time * 1000:.3f} µs/call")
        
        # Function calls should be < 1µs ideally
        assert empty_time < 0.01, f"Function call overhead too high: {empty_time:.6f}ms"
    
    def test_numpy_operation_overhead(self):
        """Measure NumPy operation overhead vs pure computation."""
        
        # Small arrays (where Python overhead dominates)
        small_array = np.random.rand(17, 3).astype(np.float32)
        
        # Large arrays (where computation dominates)
        large_array = np.random.rand(10000, 3).astype(np.float32)
        
        iterations = 10000
        
        # Small array operations
        start = time.perf_counter()
        for _ in range(iterations):
            result = small_array + small_array
        small_add_time = (time.perf_counter() - start) * 1000 / iterations
        
        start = time.perf_counter()
        for _ in range(iterations):
            result = np.dot(small_array.T, small_array)
        small_dot_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Large array operations
        large_iterations = 100
        
        start = time.perf_counter()
        for _ in range(large_iterations):
            result = large_array + large_array
        large_add_time = (time.perf_counter() - start) * 1000 / large_iterations
        
        start = time.perf_counter()
        for _ in range(large_iterations):
            result = np.dot(large_array.T, large_array)
        large_dot_time = (time.perf_counter() - start) * 1000 / large_iterations
        
        print("\nNumPy Operation Overhead:")
        print("  Small array (17x3):")
        print(f"    Add: {small_add_time * 1000:.2f} µs")
        print(f"    Dot: {small_dot_time * 1000:.2f} µs")
        print("  Large array (10000x3):")
        print(f"    Add: {large_add_time * 1000:.2f} µs")
        print(f"    Dot: {large_dot_time * 1000:.2f} µs")
        print(f"  Speedup ratio (large/small):")
        print(f"    Add: {large_add_time / small_add_time:.1f}x (data is {10000/17:.0f}x larger)")
        print(f"    Dot: {large_dot_time / small_dot_time:.1f}x")
        
        # The speedup should be LESS than the data size increase
        # This indicates Python overhead is significant for small arrays
        add_overhead_ratio = (large_add_time / small_add_time) / (10000 / 17)
        print(f"\n  Python overhead indicator (lower = more overhead):")
        print(f"    Add efficiency: {add_overhead_ratio:.2%}")
    
    def test_memory_allocation_overhead(self):
        """Measure memory allocation overhead."""
        
        # Creating new arrays
        iterations = 10000
        
        start = time.perf_counter()
        for _ in range(iterations):
            arr = np.zeros((17, 3), dtype=np.float32)
        zeros_time = (time.perf_counter() - start) * 1000 / iterations
        
        start = time.perf_counter()
        for _ in range(iterations):
            arr = np.empty((17, 3), dtype=np.float32)
        empty_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Pre-allocated (in-place operation)
        pre_alloc = np.zeros((17, 3), dtype=np.float32)
        source = np.ones((17, 3), dtype=np.float32)
        
        start = time.perf_counter()
        for _ in range(iterations):
            np.copyto(pre_alloc, source)
        copy_time = (time.perf_counter() - start) * 1000 / iterations
        
        print("\nMemory Allocation Overhead:")
        print(f"  np.zeros((17,3)):  {zeros_time * 1000:.2f} µs")
        print(f"  np.empty((17,3)):  {empty_time * 1000:.2f} µs")
        print(f"  np.copyto (reuse): {copy_time * 1000:.2f} µs")
        print(f"  Savings from reuse: {(zeros_time - copy_time) / zeros_time * 100:.1f}%")
    
    def test_object_creation_overhead(self):
        """Measure dataclass/object creation overhead."""
        from dataclasses import dataclass
        from typing import Tuple
        
        @dataclass
        class TrackerData:
            position: Tuple[float, float, float]
            rotation: Tuple[float, float, float]
            confidence: float
        
        iterations = 10000
        
        # Dataclass creation
        start = time.perf_counter()
        for _ in range(iterations):
            t = TrackerData(
                position=(1.0, 2.0, 3.0),
                rotation=(0.0, 0.0, 0.0),
                confidence=0.9,
            )
        dataclass_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Dict creation
        start = time.perf_counter()
        for _ in range(iterations):
            t = {
                'position': (1.0, 2.0, 3.0),
                'rotation': (0.0, 0.0, 0.0),
                'confidence': 0.9,
            }
        dict_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Tuple creation
        start = time.perf_counter()
        for _ in range(iterations):
            t = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 0.9)
        tuple_time = (time.perf_counter() - start) * 1000 / iterations
        
        print("\nObject Creation Overhead:")
        print(f"  Dataclass: {dataclass_time * 1000:.2f} µs")
        print(f"  Dict:      {dict_time * 1000:.2f} µs")
        print(f"  Tuple:     {tuple_time * 1000:.2f} µs")


class TestComponentOverhead:
    """Measure overhead in actual VoxelVR components."""
    
    def test_rotation_estimation_breakdown(self):
        """Break down rotation estimation into components."""
        from voxelvr.pose.rotation import (
            estimate_hip_rotation,
            estimate_foot_rotation,
            estimate_knee_rotation,
            estimate_elbow_rotation,
            estimate_all_rotations,
            normalize,
            safe_cross,
            build_rotation_matrix,
        )
        
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        iterations = 5000
        
        # Measure individual functions
        measurements = {}
        
        # normalize
        v = np.array([1.0, 2.0, 3.0])
        start = time.perf_counter()
        for _ in range(iterations):
            normalize(v)
        measurements['normalize'] = (time.perf_counter() - start) * 1000 / iterations
        
        # safe_cross
        a, b = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        start = time.perf_counter()
        for _ in range(iterations):
            safe_cross(a, b)
        measurements['safe_cross'] = (time.perf_counter() - start) * 1000 / iterations
        
        # build_rotation_matrix
        fwd, up = np.array([0., 0., 1.]), np.array([0., 1., 0.])
        start = time.perf_counter()
        for _ in range(iterations):
            build_rotation_matrix(fwd, up)
        measurements['build_rotation_matrix'] = (time.perf_counter() - start) * 1000 / iterations
        
        # estimate_hip_rotation
        start = time.perf_counter()
        for _ in range(iterations):
            estimate_hip_rotation(pose, valid_mask)
        measurements['estimate_hip'] = (time.perf_counter() - start) * 1000 / iterations
        
        # estimate_foot_rotation
        start = time.perf_counter()
        for _ in range(iterations):
            estimate_foot_rotation(pose[15], pose[13], is_left=True)
        measurements['estimate_foot'] = (time.perf_counter() - start) * 1000 / iterations
        
        # Full estimate_all_rotations
        start = time.perf_counter()
        for _ in range(iterations):
            estimate_all_rotations(pose, valid_mask)
        measurements['estimate_all'] = (time.perf_counter() - start) * 1000 / iterations
        
        print("\nRotation Estimation Breakdown:")
        print("-" * 40)
        total_components = 0
        for name, time_ms in sorted(measurements.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:25s}: {time_ms * 1000:7.2f} µs")
            if name != 'estimate_all':
                total_components += time_ms
        
        print(f"\n  Sum of components: {total_components * 1000:.2f} µs")
        print(f"  Full function:     {measurements['estimate_all'] * 1000:.2f} µs")
        print(f"  Overhead:          {(measurements['estimate_all'] - total_components) * 1000:.2f} µs")
    
    def test_filter_overhead_analysis(self):
        """Analyze One-Euro filter overhead."""
        from voxelvr.pose.filter import OneEuroFilter, PoseFilter
        
        # Single value filter
        single_filter = OneEuroFilter()
        iterations = 10000
        
        start = time.perf_counter()
        for i in range(iterations):
            single_filter.filter(float(i) * 0.1)
        single_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Full pose filter (17 joints x 3 dimensions = 51 filters)
        pose_filter = PoseFilter(num_joints=17)
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        start = time.perf_counter()
        for i in range(iterations):
            pose_filter.filter(pose + i * 0.001, valid_mask)
        pose_time = (time.perf_counter() - start) * 1000 / iterations
        
        expected_time = single_time * 17 * 3  # 17 joints x 3 dimensions
        
        print("\nOne-Euro Filter Overhead:")
        print(f"  Single filter:     {single_time * 1000:.2f} µs")
        print(f"  Expected (17x3):   {expected_time * 1000:.2f} µs")
        print(f"  Actual PoseFilter: {pose_time * 1000:.2f} µs")
        print(f"  Loop overhead:     {(pose_time - expected_time) * 1000:.2f} µs ({(pose_time/expected_time - 1)*100:.1f}%)")
    
    def test_triangulation_overhead(self):
        """Analyze triangulation overhead vs pure linalg."""
        from voxelvr.pose.triangulation import triangulate_points
        
        # Pure SVD timing (this is the core computation)
        A = np.random.rand(6, 4)  # Typical size for 3-camera triangulation
        
        iterations = 5000
        
        start = time.perf_counter()
        for _ in range(iterations):
            np.linalg.svd(A)
        svd_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Full triangulation for 17 points
        cameras = 3
        projection_matrices = {
            i: np.random.rand(3, 4) for i in range(cameras)
        }
        points_2d = {
            i: np.random.rand(17, 2) * 1000 for i in range(cameras)
        }
        
        start = time.perf_counter()
        for _ in range(iterations // 10):  # Fewer iterations as this is slower
            triangulate_points(points_2d, projection_matrices)
        triang_time = (time.perf_counter() - start) * 1000 / (iterations // 10)
        
        expected_svd_time = svd_time * 17  # 17 SVD operations
        
        print("\nTriangulation Overhead:")
        print(f"  Pure SVD (one point):  {svd_time * 1000:.2f} µs")
        print(f"  Expected (17 points):  {expected_svd_time * 1000:.2f} µs")
        print(f"  Actual triangulation:  {triang_time * 1000:.2f} µs")
        print(f"  Python loop overhead:  {(triang_time - expected_svd_time) * 1000:.2f} µs ({(triang_time/expected_svd_time - 1)*100:.1f}%)")


class TestCppCandidates:
    """Identify functions that would benefit from C++ reimplementation."""
    
    def test_identify_bottlenecks(self):
        """Run comprehensive analysis and identify C++ candidates."""
        from voxelvr.pose.rotation import estimate_all_rotations
        from voxelvr.pose.filter import PoseFilter
        from voxelvr.pose.triangulation import triangulate_points
        
        pose = generate_t_pose()
        valid_mask = np.ones(17, dtype=bool)
        
        # Measure each component
        measurements = {}
        
        # Rotation estimation
        result = measure_performance(
            estimate_all_rotations, pose, valid_mask,
            num_iterations=2000, warmup_iterations=200,
            name="rotation_estimation"
        )
        measurements['Rotation Estimation'] = result
        
        # Pose filter
        pose_filter = PoseFilter(num_joints=17)
        result = measure_performance(
            pose_filter.filter, pose, valid_mask,
            num_iterations=2000, warmup_iterations=200,
            name="pose_filter"
        )
        measurements['Pose Filter'] = result
        
        # Triangulation (synthetic data)
        cameras = 3
        projection_matrices = {i: np.random.rand(3, 4) for i in range(cameras)}
        points_2d = {i: np.random.rand(17, 2) * 1000 for i in range(cameras)}
        
        result = measure_performance(
            triangulate_points, points_2d, projection_matrices,
            num_iterations=1000, warmup_iterations=100,
            name="triangulation"
        )
        measurements['Triangulation'] = result
        
        print("\n" + "="*70)
        print("C++ REIMPLEMENTATION CANDIDATES")
        print("="*70)
        print(f"{'Component':<25} {'Mean (µs)':<12} {'Std (µs)':<10} {'Priority'}")
        print("-"*70)
        
        # Sort by time (slowest first)
        for name, result in sorted(measurements.items(), key=lambda x: -x[1].mean_ms):
            priority = "HIGH" if result.mean_ms > 0.1 else "MEDIUM" if result.mean_ms > 0.05 else "LOW"
            print(f"{name:<25} {result.mean_ms * 1000:>10.2f} {result.std_ms * 1000:>10.2f} {priority}")
        
        print("-"*70)
        print("\nRECOMMENDATIONS:")
        print("  - Functions with HIGH priority would benefit most from C++")
        print("  - Consider pybind11 or Cython for easy Python integration")
        print("  - NumPy operations are already C (BLAS/LAPACK) - low priority")
        print("  - Python loop overhead is main target for optimization")
    
    def test_pure_python_vs_numpy_vectorized(self):
        """Compare Python loops vs NumPy vectorized operations."""
        
        # Setup test data
        n_points = 17
        positions = np.random.rand(n_points, 3).astype(np.float32)
        
        iterations = 5000
        
        # Python loop version
        def python_normalize_all(positions):
            result = np.zeros_like(positions)
            for i in range(len(positions)):
                length = np.sqrt(np.sum(positions[i] ** 2))
                if length > 1e-6:
                    result[i] = positions[i] / length
            return result
        
        # NumPy vectorized version
        def numpy_normalize_all(positions):
            lengths = np.linalg.norm(positions, axis=1, keepdims=True)
            lengths = np.maximum(lengths, 1e-6)  # Avoid division by zero
            return positions / lengths
        
        # Measure Python loop
        start = time.perf_counter()
        for _ in range(iterations):
            python_normalize_all(positions)
        python_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Measure NumPy vectorized
        start = time.perf_counter()
        for _ in range(iterations):
            numpy_normalize_all(positions)
        numpy_time = (time.perf_counter() - start) * 1000 / iterations
        
        print("\nPython Loop vs NumPy Vectorized:")
        print(f"  Python loop: {python_time * 1000:.2f} µs")
        print(f"  NumPy vec:   {numpy_time * 1000:.2f} µs")
        print(f"  Speedup:     {python_time / numpy_time:.1f}x")
        print("\n  → Use NumPy vectorization before considering C++")
        
        # Verify results match
        py_result = python_normalize_all(positions)
        np_result = numpy_normalize_all(positions)
        np.testing.assert_allclose(py_result, np_result, rtol=1e-5)


class TestThreadingOverhead:
    """Test threading/GIL overhead for multi-camera processing."""
    
    def test_sequential_vs_parallel_detection(self):
        """Compare sequential vs parallel 2D detection simulation."""
        import concurrent.futures
        
        # Simulate 2D detection work (without actual model)
        def simulate_detection(image: np.ndarray) -> np.ndarray:
            # Simulate typical preprocessing (resize to 192x192)
            # Use proper slicing to get target size
            h, w = image.shape[:2]
            h_step, w_step = h // 192, w // 192
            resized = image[::h_step, ::w_step, :][:192, :192]
            resized = resized.astype(np.float32)
            
            # Simulate some computation
            result = np.sum(resized, axis=(0, 1))
            return result
        
        num_cameras = 4
        images = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(num_cameras)]
        
        iterations = 50
        
        # Sequential
        start = time.perf_counter()
        for _ in range(iterations):
            results = [simulate_detection(img) for img in images]
        seq_time = (time.perf_counter() - start) * 1000 / iterations
        
        # Thread pool
        start = time.perf_counter()
        for _ in range(iterations):
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_cameras) as executor:
                results = list(executor.map(simulate_detection, images))
        thread_time = (time.perf_counter() - start) * 1000 / iterations
        
        print("\nSequential vs Threaded Processing (simulated):")
        print(f"  Sequential ({num_cameras} cameras): {seq_time:.2f} ms")
        print(f"  ThreadPool ({num_cameras} cameras): {thread_time:.2f} ms")
        print(f"  Speedup: {seq_time / thread_time:.2f}x")
        print("\n  Note: Real speedup depends on GIL release in ONNX Runtime")
