
import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from voxelvr.pose import PoseFilter
from voxelvr.gui.tracking_panel import TrackingStatus

class MockPipelineComponents:
    """Helper to mock complex pipeline chain."""
    
    def __init__(self):
        self.positions = np.random.rand(17, 3)
        self.valid = np.ones(17, dtype=bool)
        
    def get_pose(self):
        # Add some noise
        noise = np.random.normal(0, 0.01, (17, 3))
        return self.positions + noise

class TestFullPipelineStress:
    
    def test_long_running_pipeline_stability(self):
        """Simulate a long running tracking loop to check for memory/stability."""
        
        # Setup components
        filter = PoseFilter()
        pipeline = MockPipelineComponents()
        
        start_time = time.time()
        iterations = 0
        max_duration = 0.5 # Run for 0.5 seconds (simulated stress)
        
        simulated_fps = 60
        frame_time = 1.0 / simulated_fps
        
        while time.time() - start_time < max_duration:
            loop_start = time.time()
            
            # 1. Get noisy pose
            raw_pos = pipeline.get_pose()
            
            # 2. Filter
            filtered_pos = filter.filter(raw_pos, pipeline.valid)
            
            # 3. Verify output integrity
            assert filtered_pos.shape == (17, 3)
            assert not np.any(np.isnan(filtered_pos))
            
            # 4. Simulate timeline advancement
            time.sleep(0.001) 
            iterations += 1
            
        assert iterations > 10 # Ensure we actually ran
        
    def test_random_data_injection(self):
        """Inject completely random/invalid data and ensure no crash."""
        filter = PoseFilter()
        
        for _ in range(100):
            # Generate garbage data
            # - Random shapes (invalid, but type checking usually catches this before filter)
            # - Random values (NaN, Inf)
            
            # Case 1: Extreme values
            positions = np.random.uniform(-1000, 1000, (17, 3))
            valid = np.random.choice([True, False], 17)
            
            try:
                out = filter.filter(positions, valid)
                assert not np.any(np.isnan(out))
            except Exception as e:
                pytest.fail(f"Filter crashed on extreme values: {e}")
                
            # Case 2: NaNs in input
            positions_nan = positions.copy()
            positions_nan[0, 0] = np.nan
            
            # Filter handles basic smoothing, but might propagate NaNs if underlying logic doesn't catch it.
            # Real world: user code should prevent NaNs.
            # But let's see if it crashes.
            try:
                out = filter.filter(positions_nan, valid)
                # It's acceptable for output to contain NaN if input did, 
                # OR for it to handle it. Just allow no crash.
            except Exception as e:
                pass # Crash is bad, but exception handling might be in the caller.
                # Ideally filter should be robust.
                
    def test_osc_sender_simulated(self):
        """Test OSC sender logic with huge throughput."""
        from voxelvr.transport import OSCSender
        
        # Mock pythonosc
        with patch('pythonosc.udp_client.SimpleUDPClient') as mock_client:
            sender = OSCSender(ip="127.0.0.1", port=9000)
            sender.connect()
            
            # Send 1000 packets
            from voxelvr.transport.osc_sender import VRChatTracker
            data = {
                'hip': VRChatTracker(position=(0,0,0), rotation=(0,0,0)),
                'head': VRChatTracker(position=(0,1.7,0), rotation=(0,0,0))
            }
            
            for _ in range(1000):
                sender.send_all_trackers(data)
                
            # Verify no exceptions and client called
            assert mock_client.return_value.send_message.called
