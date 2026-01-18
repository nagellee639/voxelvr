"""
OSC Status Indicator

Monitors and displays the status of OSC connections to VRChat or other clients.
"""

import time
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from collections import deque


class ConnectionState(Enum):
    """OSC connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class OSCStats:
    """Statistics for OSC communication."""
    messages_sent: int = 0
    messages_per_second: float = 0.0
    last_send_time: float = 0.0
    bytes_sent: int = 0
    errors: int = 0


class OSCStatusIndicator:
    """
    Monitors OSC connection status and provides real-time statistics.
    
    Features:
    - Connection state tracking (connected/connecting/disconnected/error)
    - Message rate monitoring
    - Activity detection
    - State change callbacks
    """
    
    def __init__(
        self,
        activity_timeout: float = 2.0,
        rate_window: float = 1.0,
    ):
        """
        Initialize the OSC status indicator.
        
        Args:
            activity_timeout: Seconds of inactivity before marking as disconnected
            rate_window: Window size for message rate calculation
        """
        self.activity_timeout = activity_timeout
        self.rate_window = rate_window
        
        self._state = ConnectionState.DISCONNECTED
        self._target_ip = "127.0.0.1"
        self._target_port = 9000
        
        # Statistics
        self._messages_sent = 0
        self._bytes_sent = 0
        self._errors = 0
        self._last_send_time = 0.0
        
        # Rate calculation
        self._send_times: deque = deque(maxlen=100)
        
        # State change callbacks
        self._callbacks: List[Callable[[ConnectionState], None]] = []
    
    @property
    def state(self) -> ConnectionState:
        """Get current connection state, checking for timeout."""
        if self._state == ConnectionState.CONNECTED:
            if time.time() - self._last_send_time > self.activity_timeout:
                self._set_state(ConnectionState.DISCONNECTED)
        return self._state
    
    @property
    def target(self) -> str:
        """Get target address string."""
        return f"{self._target_ip}:{self._target_port}"
    
    def _set_state(self, new_state: ConnectionState) -> None:
        """Set state and notify callbacks if changed."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            for callback in self._callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    print(f"OSC status callback error: {e}")
    
    def add_state_callback(self, callback: Callable[[ConnectionState], None]) -> None:
        """Add callback for state changes."""
        self._callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable[[ConnectionState], None]) -> None:
        """Remove a state callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def set_target(self, ip: str, port: int) -> None:
        """Update target address."""
        self._target_ip = ip
        self._target_port = port
    
    def on_connect(self) -> None:
        """Call when OSC client connects successfully."""
        self._set_state(ConnectionState.CONNECTING)
    
    def on_message_sent(self, byte_count: int = 0) -> None:
        """
        Call when an OSC message is sent successfully.
        
        Args:
            byte_count: Number of bytes in the message
        """
        now = time.time()
        self._messages_sent += 1
        self._bytes_sent += byte_count
        self._last_send_time = now
        self._send_times.append(now)
        
        # Mark as connected after successful send
        if self._state != ConnectionState.CONNECTED:
            self._set_state(ConnectionState.CONNECTED)
    
    def on_error(self, error: Optional[str] = None) -> None:
        """
        Call when an OSC error occurs.
        
        Args:
            error: Optional error message
        """
        self._errors += 1
        self._set_state(ConnectionState.ERROR)
        if error:
            print(f"OSC error: {error}")
    
    def on_disconnect(self) -> None:
        """Call when OSC client disconnects."""
        self._set_state(ConnectionState.DISCONNECTED)
    
    def get_stats(self) -> OSCStats:
        """Get current OSC statistics."""
        # Calculate messages per second
        now = time.time()
        cutoff = now - self.rate_window
        recent_sends = [t for t in self._send_times if t > cutoff]
        mps = len(recent_sends) / self.rate_window if self.rate_window > 0 else 0
        
        return OSCStats(
            messages_sent=self._messages_sent,
            messages_per_second=mps,
            last_send_time=self._last_send_time,
            bytes_sent=self._bytes_sent,
            errors=self._errors,
        )
    
    def get_state_color(self) -> tuple:
        """
        Get RGB color for current state (for UI display).
        
        Returns:
            (R, G, B) tuple with values 0-255
        """
        state = self.state  # Property updates state based on timeout
        
        if state == ConnectionState.CONNECTED:
            return (0, 200, 100)  # Green
        elif state == ConnectionState.CONNECTING:
            return (255, 200, 0)  # Yellow
        elif state == ConnectionState.ERROR:
            return (255, 80, 80)  # Red
        else:  # DISCONNECTED
            return (128, 128, 128)  # Gray
    
    def get_state_text(self) -> str:
        """Get human-readable state text."""
        state = self.state
        stats = self.get_stats()
        
        if state == ConnectionState.CONNECTED:
            return f"Connected ({stats.messages_per_second:.0f} msg/s)"
        elif state == ConnectionState.CONNECTING:
            return "Connecting..."
        elif state == ConnectionState.ERROR:
            return f"Error ({self._errors} errors)"
        else:
            return "Disconnected"
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._messages_sent = 0
        self._bytes_sent = 0
        self._errors = 0
        self._send_times.clear()
