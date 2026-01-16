
import pytest
from unittest.mock import MagicMock
from voxelvr.gui.tracking_panel import TrackingPanel, TrackingState

class TestTrackingError:
    """Tests to reproduce and fix tracking state errors."""

    def test_state_callback_exception_handling(self):
        """
        Reproduce 'State callback error'.
        
        The error message in the logs confirms that `_notify_state_change` catches exceptions 
        from callbacks and prints them. We want to verify this behavior and ensure it doesn't 
        crash the app, but also we want to find out why it might be happening.
        
        If the user sees "State callback error: IDLE", it means an exception was raised
        with the message "IDLE" or string representation "IDLE".
        
        This suggests maybe an Enum is being raised or printed?
        Or maybe a callback is failing when state is IDLE (or STOPPED)?
        """
        panel = TrackingPanel()
        
        # Mock a callback that raises an exception
        mock_callback = MagicMock(side_effect=Exception("IDLE"))
        panel.add_state_callback(mock_callback)
        
        # Trigger state change
        # This should print "State callback error: IDLE" to stdout but NOT raise exception
        try:
            panel.request_start()
        except Exception as e:
            pytest.fail(f"TrackingPanel crashed on callback error: {e}")
            
        mock_callback.assert_called()

    def test_simulated_app_callback_logic(self):
        """
        Simulate app.py's callback logic to reproduce the error.
        
        Hypothesis: app.py uses TrackingState.IDLE which does not exist.
        """
        def bad_callback(state):
            # This is what app.py does at line 766
            if state == TrackingState.IDLE:
                pass
                
        panel = TrackingPanel()
        panel.add_state_callback(bad_callback)
        
        # Trigger any state change to fire the callback
        try:
            panel.request_start()
        except AttributeError as e:
            # We expect an AttributeError here if IDLE doesn't exist
            print(f"Caught expected error: {e}")
            assert "IDLE" in str(e)
            
        # If the panel catches it, it prints it.
        # But wait, panel.request_start() sets state to STARTING.
        # The callback fires.
        # Inside callback, TrackingState.IDLE is accessed.
        # Raise AttributeError.
        # Panel catches it and prints "State callback error: <e>"
        
        # We can't capture stdout easily here without capsys, but the fact it doesn't crash 
        # is what we know. We want to verify `TrackingState.IDLE` is indeed invalid.
        
        with pytest.raises(AttributeError):
             _ = TrackingState.IDLE
