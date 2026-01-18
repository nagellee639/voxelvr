"""
Debug Panel

Debug and tuning interface for filter parameters and performance optimization.
"""

import numpy as np
import time
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass
from collections import deque

from .param_optimizer import ParameterOptimizer, FilterProfile, JitterMetrics, LatencyMetrics


@dataclass
class DebugMetrics:
    """Metrics displayed in the debug panel."""
    # Jitter
    jitter_position_mm: float = 0.0
    jitter_velocity: float = 0.0
    max_deviation_mm: float = 0.0
    
    # Latency
    estimated_latency_ms: float = 0.0
    filter_delay_frames: float = 0.0
    
    # Filter state
    current_min_cutoff: float = 1.0
    current_beta: float = 0.5
    current_d_cutoff: float = 1.0
    current_profile: str = "Balanced"
    auto_adjust_enabled: bool = True
    manual_override: bool = False


class DebugPanel:
    """
    Debug and tuning interface.
    
    Features:
    - Filter parameter sliders (min_cutoff, beta, d_cutoff)
    - Real-time jitter visualization
    - Latency measurement display
    - Profile selector (Low Jitter, Balanced, Low Latency, Precision)
    - Auto-adjust toggle with continuous optimization
    - Manual override button
    - Parameter history graph
    """
    
    def __init__(
        self,
        optimizer: Optional[ParameterOptimizer] = None,
    ):
        """
        Initialize debug panel.
        
        Args:
            optimizer: Parameter optimizer instance (creates one if None)
        """
        self.optimizer = optimizer or ParameterOptimizer()
        
        # Metrics
        self._metrics = DebugMetrics()
        
        # Jitter visualization data
        self._jitter_history: deque = deque(maxlen=300)  # 10 seconds at 30fps
        
        # Parameter change callbacks
        self._param_callbacks: List[Callable[[float, float, float], None]] = []
        
        # Connect to optimizer
        self.optimizer.add_callback(self._on_params_changed)
        
        # Update metrics from optimizer
        self._update_from_optimizer()
    
    def _on_params_changed(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        """Called when optimizer parameters change."""
        self._metrics.current_min_cutoff = min_cutoff
        self._metrics.current_beta = beta
        self._metrics.current_d_cutoff = d_cutoff
        
        # Notify external callbacks
        for callback in self._param_callbacks:
            try:
                callback(min_cutoff, beta, d_cutoff)
            except Exception as e:
                print(f"Debug panel callback error: {e}")
    
    def _update_from_optimizer(self) -> None:
        """Update metrics from optimizer state."""
        self._metrics.current_min_cutoff = self.optimizer.min_cutoff
        self._metrics.current_beta = self.optimizer.beta
        self._metrics.current_d_cutoff = self.optimizer.d_cutoff
        self._metrics.current_profile = self.optimizer.current_profile.value
        self._metrics.auto_adjust_enabled = self.optimizer.auto_adjust_enabled
        self._metrics.manual_override = self.optimizer.manual_override
    
    @property
    def metrics(self) -> DebugMetrics:
        """Get current debug metrics."""
        self._update_from_optimizer()
        return self._metrics
    
    def add_param_callback(self, callback: Callable[[float, float, float], None]) -> None:
        """Add callback for parameter changes."""
        self._param_callbacks.append(callback)
    
    def set_min_cutoff(self, value: float) -> None:
        """Set min_cutoff parameter manually."""
        self.optimizer.set_manual_parameters(min_cutoff=value)
        self._update_from_optimizer()
    
    def set_beta(self, value: float) -> None:
        """Set beta parameter manually."""
        self.optimizer.set_manual_parameters(beta=value)
        self._update_from_optimizer()
    
    def set_d_cutoff(self, value: float) -> None:
        """Set d_cutoff parameter manually."""
        self.optimizer.set_manual_parameters(d_cutoff=value)
        self._update_from_optimizer()
    
    def set_profile(self, profile_name: str) -> None:
        """
        Set filter profile by name.
        
        Args:
            profile_name: One of 'low_jitter', 'balanced', 'low_latency', 'precision'
        """
        profile_map = {
            'low_jitter': FilterProfile.LOW_JITTER,
            'balanced': FilterProfile.BALANCED,
            'low_latency': FilterProfile.LOW_LATENCY,
            'precision': FilterProfile.PRECISION,
        }
        
        profile = profile_map.get(profile_name.lower())
        if profile:
            self.optimizer.set_profile(profile)
            self._update_from_optimizer()
    
    def set_auto_adjust(self, enabled: bool) -> None:
        """Enable or disable auto-adjustment."""
        self.optimizer.enable_auto_adjust(enabled)
        self._update_from_optimizer()
    
    def enable_manual_override(self) -> None:
        """Enable manual override mode."""
        self.optimizer.manual_override = True
        self._update_from_optimizer()
    
    def disable_manual_override(self) -> None:
        """Disable manual override and return to auto-adjustment."""
        self.optimizer.disable_manual_override()
        self._update_from_optimizer()
    
    def update(
        self,
        positions: np.ndarray,
        valid_mask: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update debug panel with new pose data.
        
        Args:
            positions: (17, 3) joint positions
            valid_mask: (17,) boolean mask
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update optimizer (handles auto-adjustment internally)
        self.optimizer.update(positions, valid_mask, timestamp)
        
        # Measure jitter
        jitter = self.optimizer.measure_jitter()
        self._metrics.jitter_position_mm = jitter.position_std * 1000
        self._metrics.jitter_velocity = jitter.velocity_std
        self._metrics.max_deviation_mm = jitter.max_deviation * 1000
        
        # Record jitter history for visualization
        self._jitter_history.append((timestamp, self._metrics.jitter_position_mm))
        
        # Estimate latency
        latency = self.optimizer.estimate_latency()
        self._metrics.estimated_latency_ms = latency.estimated_latency_ms
        self._metrics.filter_delay_frames = latency.filter_delay_frames
        
        self._update_from_optimizer()
    
    def get_jitter_history(self) -> List[Tuple[float, float]]:
        """Get jitter history for visualization."""
        return list(self._jitter_history)
    
    def get_parameter_history(self) -> List[Dict]:
        """Get parameter history from optimizer."""
        return list(self.optimizer.parameter_history)
    
    def get_profile_options(self) -> List[Dict]:
        """Get available profile options for UI."""
        return [
            {
                'name': 'Low Jitter',
                'value': 'low_jitter',
                'description': 'Maximum smoothing, higher latency',
            },
            {
                'name': 'Balanced',
                'value': 'balanced',
                'description': 'Good compromise between smoothness and responsiveness',
            },
            {
                'name': 'Low Latency',
                'value': 'low_latency',
                'description': 'Minimal smoothing, fastest response',
            },
            {
                'name': 'Precision',
                'value': 'precision',
                'description': 'Optimized for slow, precise movements',
            },
        ]
    
    def get_slider_ranges(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get slider ranges for parameters.
        
        Returns:
            Dict mapping param name to (min, max, default) tuple
        """
        return {
            'min_cutoff': (0.1, 10.0, 1.0),
            'beta': (0.0, 2.0, 0.5),
            'd_cutoff': (0.1, 5.0, 1.0),
        }
    
    def get_current_values(self) -> Dict[str, float]:
        """Get current parameter values."""
        return {
            'min_cutoff': self.optimizer.min_cutoff,
            'beta': self.optimizer.beta,
            'd_cutoff': self.optimizer.d_cutoff,
        }
    
    def reset_to_defaults(self) -> None:
        """Reset parameters to default (Balanced profile)."""
        self.optimizer.set_profile(FilterProfile.BALANCED)
        self.optimizer.enable_auto_adjust(True)
        self._update_from_optimizer()
    
    def get_graph_data(
        self,
        metric: str,
        max_points: int = 300,
    ) -> Tuple[List[float], List[float]]:
        """
        Get time-series data for graphing.
        
        Args:
            metric: 'jitter' or 'parameters'
            max_points: Maximum points to return
        """
        if metric == 'jitter':
            data = list(self._jitter_history)
        elif metric == 'min_cutoff':
            data = [(d['time'], d['min_cutoff']) for d in self.optimizer.parameter_history]
        elif metric == 'beta':
            data = [(d['time'], d['beta']) for d in self.optimizer.parameter_history]
        else:
            return [], []
        
        if not data:
            return [], []
        
        # Downsample if needed
        if len(data) > max_points:
            step = len(data) // max_points
            data = data[::step]
        
        times = [t for t, _ in data]
        values = [v for _, v in data]
        
        # Normalize times
        if times:
            t0 = times[0]
            times = [t - t0 for t in times]
        
        return times, values
