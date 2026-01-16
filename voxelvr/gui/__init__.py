"""
VoxelVR GUI Module

Provides graphical user interface components for VoxelVR.
"""

from .app import VoxelVRApp
from .camera_panel import CameraPanel, CameraFeedInfo
from .calibration_panel import CalibrationPanel, CalibrationStep, CalibrationState
from .tracking_panel import TrackingPanel, TrackingState, TrackerConfig
from .performance_panel import PerformancePanel, PerformanceMetrics
from .debug_panel import DebugPanel, DebugMetrics
from .osc_status import OSCStatusIndicator, ConnectionState, OSCStats
from .param_optimizer import ParameterOptimizer, FilterProfile, ProfileSettings, PROFILE_PRESETS

__all__ = [
    # Main application
    "VoxelVRApp",
    
    # Panels
    "CameraPanel",
    "CameraFeedInfo",
    "CalibrationPanel",
    "CalibrationStep",
    "CalibrationState",
    "TrackingPanel",
    "TrackingState",
    "TrackerConfig",
    "PerformancePanel",
    "PerformanceMetrics",
    "DebugPanel",
    "DebugMetrics",
    
    # Status and optimization
    "OSCStatusIndicator",
    "ConnectionState",
    "OSCStats",
    "ParameterOptimizer",
    "FilterProfile",
    "ProfileSettings",
    "PROFILE_PRESETS",
]
