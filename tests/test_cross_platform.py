"""
Cross-Platform Compatibility Tests

Verifies that VoxelVR works correctly on:
- Windows/Linux
- AMD/NVIDIA/Intel GPUs
- CPU fallback
"""

import pytest
import platform
import sys
import numpy as np


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility."""
    
    def test_platform_detection(self):
        """Verify platform detection works."""
        current_platform = platform.system()
        assert current_platform in ['Windows', 'Linux', 'Darwin'], \
            f"Unsupported platform: {current_platform}"
    
    def test_numpy_available(self):
        """Verify NumPy is available and working."""
        arr = np.array([1, 2, 3])
        assert len(arr) == 3
        assert arr.sum() == 6
    
    def test_opencv_available(self):
        """Verify OpenCV is available and working."""
        import cv2
        
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = (255, 255, 255)
        
        # Test basic operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (100, 100)
        
        resized = cv2.resize(img, (50, 50))
        assert resized.shape == (50, 50, 3)


class TestONNXRuntimeBackends:
    """Test ONNX Runtime backend availability."""
    
    def test_onnxruntime_import(self):
        """Verify ONNX Runtime can be imported."""
        try:
            import onnxruntime as ort
            assert hasattr(ort, 'InferenceSession')
        except ImportError:
            pytest.skip("ONNX Runtime not installed")
    
    def test_available_providers(self):
        """Check which execution providers are available."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("ONNX Runtime not installed")
        
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        # Must have at least CPU provider
        assert 'CPUExecutionProvider' in providers, \
            "CPU provider should always be available"
    
    def test_gpu_provider_available(self):
        """Check for GPU provider (informational, doesn't fail)."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("ONNX Runtime not installed")
        
        providers = ort.get_available_providers()
        
        gpu_providers = [
            'CUDAExecutionProvider',      # NVIDIA
            'DmlExecutionProvider',        # DirectML (Windows)
            'ROCMExecutionProvider',       # AMD ROCm (Linux)
        ]
        
        available_gpu = [p for p in gpu_providers if p in providers]
        
        if available_gpu:
            print(f"GPU providers available: {available_gpu}")
        else:
            print("No GPU providers available - will use CPU")
            # Don't fail, just inform
    
    def test_directml_on_windows(self):
        """Test DirectML availability on Windows."""
        if platform.system() != 'Windows':
            pytest.skip("DirectML is Windows-only")
        
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            # DirectML should be available on Windows with proper install
            if 'DmlExecutionProvider' not in providers:
                pytest.skip("DirectML not installed - install onnxruntime-directml")
            
            assert 'DmlExecutionProvider' in providers
        except ImportError:
            pytest.skip("ONNX Runtime not installed")
    
    def test_cuda_on_nvidia(self):
        """Test CUDA availability with NVIDIA GPU."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' not in providers:
                pytest.skip("CUDA not available - install onnxruntime-gpu with CUDA")
            
            assert 'CUDAExecutionProvider' in providers
        except ImportError:
            pytest.skip("ONNX Runtime not installed")
    
    def test_rocm_on_amd(self):
        """Test ROCm availability with AMD GPU on Linux."""
        if platform.system() != 'Linux':
            pytest.skip("ROCm is Linux-only")
        
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'ROCMExecutionProvider' not in providers:
                pytest.skip("ROCm not available - install onnxruntime with ROCm support")
            
            assert 'ROCMExecutionProvider' in providers
        except ImportError:
            pytest.skip("ONNX Runtime not installed")


class TestPoseDetectorBackends:
    """Test pose detector with different backends."""
    
    def test_detector_auto_backend(self):
        """Test detector with automatic backend selection."""
        from voxelvr.pose import PoseDetector2D
        
        detector = PoseDetector2D(backend="auto")
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Try to detect (may download model on first run)
        try:
            result = detector.detect(dummy_image)
            # Result may be None on random image, but shouldn't crash
            print(f"Detection result: {result}")
        except Exception as e:
            pytest.fail(f"Detection failed: {e}")
    
    def test_detector_cpu_fallback(self):
        """Test detector falls back to CPU correctly."""
        from voxelvr.pose import PoseDetector2D
        
        detector = PoseDetector2D(backend="cpu")
        
        # Load model with CPU backend
        success = detector.load_model()
        
        if success:
            # Verify CPU provider is being used
            providers = detector.session.get_providers()
            assert 'CPUExecutionProvider' in providers
    
    @pytest.mark.gpu
    def test_detector_gpu_backend(self):
        """Test detector with GPU backend."""
        from voxelvr.pose import PoseDetector2D
        
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except ImportError:
            pytest.skip("ONNX Runtime not installed")
        
        gpu_available = any(p in providers for p in [
            'CUDAExecutionProvider',
            'DmlExecutionProvider',
            'ROCMExecutionProvider'
        ])
        
        if not gpu_available:
            pytest.skip("No GPU provider available")
        
        detector = PoseDetector2D(backend="auto")
        success = detector.load_model()
        
        if success:
            session_providers = detector.session.get_providers()
            print(f"Session providers: {session_providers}")
            # Should have a non-CPU provider first
            assert session_providers[0] != 'CPUExecutionProvider'


class TestOSCCompatibility:
    """Test OSC library works cross-platform."""
    
    def test_osc_import(self):
        """Verify python-osc can be imported."""
        from pythonosc import udp_client
        from pythonosc.osc_message_builder import OscMessageBuilder
        
        # Create a client (doesn't actually connect)
        client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
        assert client is not None
    
    def test_osc_message_build(self):
        """Test building OSC messages."""
        from pythonosc.osc_message_builder import OscMessageBuilder
        
        builder = OscMessageBuilder(address="/test/address")
        builder.add_arg(1.0)
        builder.add_arg(2.0)
        builder.add_arg(3.0)
        
        message = builder.build()
        assert message is not None


class TestPathHandling:
    """Test path handling across platforms."""
    
    def test_config_dir_creation(self):
        """Test config directory is created correctly."""
        from pathlib import Path
        from voxelvr.config import VoxelVRConfig
        
        config = VoxelVRConfig()
        config.ensure_dirs()
        
        assert config.config_dir.exists()
        assert config.calibration_dir.exists()
    
    def test_home_directory(self):
        """Test home directory detection."""
        from pathlib import Path
        
        home = Path.home()
        assert home.exists()
        assert home.is_dir()
    
    def test_voxelvr_cache_dir(self):
        """Test VoxelVR cache directory."""
        from pathlib import Path
        
        cache_dir = Path.home() / ".voxelvr"
        # Don't require it to exist, just verify path is valid
        assert str(cache_dir).startswith(str(Path.home()))
