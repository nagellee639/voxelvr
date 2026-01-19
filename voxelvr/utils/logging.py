"""
VoxelVR Logging Utility

Provides timestamped console logging for all VoxelVR components.
"""

import sys
from datetime import datetime
from typing import Optional


def log(message: str, level: str = "INFO", component: Optional[str] = None) -> None:
    """
    Print a timestamped log message.
    
    Args:
        message: The message to log
        level: Log level (INFO, WARN, ERROR, DEBUG)
        component: Optional component name (e.g., "OSC", "Tracking")
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
    
    if component:
        prefix = f"[{timestamp}] [{level}] [{component}]"
    else:
        prefix = f"[{timestamp}] [{level}]"
    
    print(f"{prefix} {message}")


def log_info(message: str, component: Optional[str] = None) -> None:
    """Log an info message."""
    log(message, "INFO", component)


def log_warn(message: str, component: Optional[str] = None) -> None:
    """Log a warning message."""
    log(message, "WARN", component)


def log_error(message: str, component: Optional[str] = None) -> None:
    """Log an error message."""
    log(message, "ERROR", component)


def log_debug(message: str, component: Optional[str] = None) -> None:
    """Log a debug message."""
    log(message, "DEBUG", component)


# Convenience aliases for common components
class ComponentLogger:
    """Logger bound to a specific component."""
    
    def __init__(self, component: str):
        self.component = component
    
    def info(self, message: str) -> None:
        log_info(message, self.component)
    
    def warn(self, message: str) -> None:
        log_warn(message, self.component)
    
    def error(self, message: str) -> None:
        log_error(message, self.component)
    
    def debug(self, message: str) -> None:
        log_debug(message, self.component)


# Pre-configured loggers for common components
osc_log = ComponentLogger("OSC")
tracking_log = ComponentLogger("Tracking")
calibration_log = ComponentLogger("Calibration")
camera_log = ComponentLogger("Camera")
gui_log = ComponentLogger("GUI")
