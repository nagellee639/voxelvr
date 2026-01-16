
import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from voxelvr.capture.manager import CameraManager, CameraConfig
from voxelvr.capture.camera import CameraFrame, Camera

class TestCameraRobustness:
    
    @pytest.fixture
    def mock_camera_cls(self):
        with patch('voxelvr.capture.manager.Camera') as MockCamera:
            yield MockCamera

    def test_sync_failure_handling(self, mock_camera_cls):
        """Test behavior when cameras are out of sync."""
        mgr = CameraManager()
        
        # Create two mock camera instances
        cam1 = MagicMock(spec=Camera)
        cam2 = MagicMock(spec=Camera)
        
        cam1.camera_id = 0
        cam2.camera_id = 1
        
        # Setup get_frame behavior
        def get_frame_cam1(timeout=0.1):
            return CameraFrame(
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
                timestamp=100.0, # Base time
                frame_number=1,
                camera_id=0
            )
            
        def get_frame_cam2(timeout=0.1):
            return CameraFrame(
                image=np.zeros((720, 1280, 3), dtype=np.uint8),
                timestamp=100.1, # 100ms later (desync > 50ms)
                frame_number=1,
                camera_id=1
            )
            
        cam1.get_frame.side_effect = get_frame_cam1
        cam2.get_frame.side_effect = get_frame_cam2
        
        # Inject into manager
        mgr.cameras = {0: cam1, 1: cam2}
        
        # Test
        frames = mgr.get_synchronized_frames(max_time_diff=0.05, timeout=0.1)
        
        # Should initiate retry logic, and since we return same frames, 
        # it might fail or return them anyway depending on retry implementation.
        # But it should NOT crash.
        
    def test_partial_failure(self, mock_camera_cls):
        """Test when one camera returns None."""
        mgr = CameraManager()
        
        cam1 = MagicMock(spec=Camera)
        cam2 = MagicMock(spec=Camera)
        
        cam1.get_frame.return_value = CameraFrame(
            image=np.zeros((10,10,3)), timestamp=100.0, frame_number=1, camera_id=0
        )
        cam2.get_frame.return_value = None # Fails
        
        mgr.cameras = {0: cam1, 1: cam2}
        
        frames = mgr.get_synchronized_frames()
        assert frames is None
        
    def test_timeout_recovery(self, mock_camera_cls):
        """Test transient failure recovery."""
        mgr = CameraManager()
        cam = MagicMock(spec=Camera)
        mgr.cameras = {0: cam}
        
        valid_frame = CameraFrame(
            image=np.zeros((10,10,3)), timestamp=100.0, frame_number=1, camera_id=0
        )
        
        # Sequence: Success, Failure, Success
        cam.get_frame.side_effect = [valid_frame, None, valid_frame]
        
        assert mgr.get_synchronized_frames() is not None
        assert mgr.get_synchronized_frames() is None
        assert mgr.get_synchronized_frames() is not None
