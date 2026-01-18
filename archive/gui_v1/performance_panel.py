"""
Performance Panel

Real-time performance monitoring dashboard.
"""

import time
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class PerformanceMetrics:
    """Complete set of performance metrics."""
    # FPS measurements
    capture_fps: float = 0.0
    detection_fps: float = 0.0
    triangulation_fps: float = 0.0
    filter_fps: float = 0.0
    total_fps: float = 0.0
    
    # Latency measurements (ms)
    capture_latency_ms: float = 0.0
    detection_latency_ms: float = 0.0
    triangulation_latency_ms: float = 0.0
    filter_latency_ms: float = 0.0
    osc_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Quality metrics
    num_valid_joints: int = 0
    avg_confidence: float = 0.0
    jitter_mm: float = 0.0
    
    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_percent: float = 0.0


class PerformancePanel:
    """
    Performance monitoring dashboard.
    
    Features:
    - Real-time FPS graphs
    - Latency breakdown chart
    - CPU/GPU/Memory usage
    - Frame timing histogram
    - Performance statistics
    """
    
    def __init__(
        self,
        history_seconds: float = 10.0,
        sample_rate: float = 30.0,
    ):
        """
        Initialize performance panel.
        
        Args:
            history_seconds: Seconds of history to keep
            sample_rate: Expected samples per second
        """
        self.history_seconds = history_seconds
        self.sample_rate = sample_rate
        
        max_samples = int(history_seconds * sample_rate)
        
        # FPS history
        self._fps_history: deque = deque(maxlen=max_samples)
        self._capture_fps_history: deque = deque(maxlen=max_samples)
        self._detection_fps_history: deque = deque(maxlen=max_samples)
        self._triangulation_fps_history: deque = deque(maxlen=max_samples)
        
        # Latency history
        self._total_latency_history: deque = deque(maxlen=max_samples)
        self._latency_breakdown_history: deque = deque(maxlen=max_samples)
        
        # Quality history
        self._valid_joints_history: deque = deque(maxlen=max_samples)
        self._confidence_history: deque = deque(maxlen=max_samples)
        self._jitter_history: deque = deque(maxlen=max_samples)
        
        # Resource history
        self._cpu_history: deque = deque(maxlen=max_samples)
        self._memory_history: deque = deque(maxlen=max_samples)
        self._gpu_history: deque = deque(maxlen=max_samples)
        
        # Timing for FPS calculation
        self._last_update_time: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = time.time()
        
        # Current metrics
        self._current_metrics = PerformanceMetrics()
        
        # Frame timing histogram bins (ms)
        self._timing_bins = [0, 5, 10, 16, 20, 33, 50, 100, 200]
        self._timing_histogram: List[int] = [0] * (len(self._timing_bins) + 1)
    
    @property
    def current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self._current_metrics
    
    def update(
        self,
        metrics: Optional[PerformanceMetrics] = None,
        **kwargs,
    ) -> None:
        """
        Update performance metrics.
        
        Can pass a PerformanceMetrics object or individual keyword arguments.
        """
        now = time.time()
        
        if metrics is None:
            # Update individual fields
            for key, value in kwargs.items():
                if hasattr(self._current_metrics, key):
                    setattr(self._current_metrics, key, value)
        else:
            self._current_metrics = metrics
        
        # Update histories
        m = self._current_metrics
        
        self._fps_history.append((now, m.total_fps))
        self._capture_fps_history.append((now, m.capture_fps))
        self._detection_fps_history.append((now, m.detection_fps))
        self._triangulation_fps_history.append((now, m.triangulation_fps))
        
        self._total_latency_history.append((now, m.total_latency_ms))
        self._latency_breakdown_history.append((now, {
            'capture': m.capture_latency_ms,
            'detection': m.detection_latency_ms,
            'triangulation': m.triangulation_latency_ms,
            'filter': m.filter_latency_ms,
            'osc': m.osc_latency_ms,
        }))
        
        self._valid_joints_history.append((now, m.num_valid_joints))
        self._confidence_history.append((now, m.avg_confidence))
        self._jitter_history.append((now, m.jitter_mm))
        
        self._cpu_history.append((now, m.cpu_percent))
        self._memory_history.append((now, m.memory_mb))
        self._gpu_history.append((now, m.gpu_percent))
        
        # Update timing histogram
        if m.total_latency_ms > 0:
            bin_idx = 0
            for i, threshold in enumerate(self._timing_bins):
                if m.total_latency_ms <= threshold:
                    bin_idx = i
                    break
            else:
                bin_idx = len(self._timing_bins)
            self._timing_histogram[bin_idx] += 1
        
        self._frame_count += 1
        self._last_update_time = now
    
    def get_fps_history(self) -> Dict[str, List[Tuple[float, float]]]:
        """Get FPS history for all metrics."""
        return {
            'total': list(self._fps_history),
            'capture': list(self._capture_fps_history),
            'detection': list(self._detection_fps_history),
            'triangulation': list(self._triangulation_fps_history),
        }
    
    def get_latency_history(self) -> List[Tuple[float, float]]:
        """Get total latency history."""
        return list(self._total_latency_history)
    
    def get_latency_breakdown(self) -> Dict[str, float]:
        """Get current latency breakdown."""
        m = self._current_metrics
        return {
            'Capture': m.capture_latency_ms,
            'Detection': m.detection_latency_ms,
            'Triangulation': m.triangulation_latency_ms,
            'Filter': m.filter_latency_ms,
            'OSC': m.osc_latency_ms,
        }
    
    def get_timing_histogram(self) -> Dict[str, int]:
        """Get frame timing histogram."""
        labels = []
        for i, threshold in enumerate(self._timing_bins):
            if i == 0:
                labels.append(f"<{threshold}ms")
            else:
                labels.append(f"{self._timing_bins[i-1]}-{threshold}ms")
        labels.append(f">{self._timing_bins[-1]}ms")
        
        return dict(zip(labels, self._timing_histogram))
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of all metrics."""
        def calc_stats(history: deque) -> Dict[str, float]:
            if not history:
                return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
            
            values = [v for _, v in history]
            return {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
            }
        
        return {
            'fps': calc_stats(self._fps_history),
            'latency_ms': calc_stats(self._total_latency_history),
            'valid_joints': calc_stats(self._valid_joints_history),
            'confidence': calc_stats(self._confidence_history),
            'jitter_mm': calc_stats(self._jitter_history),
            'cpu_percent': calc_stats(self._cpu_history),
            'memory_mb': calc_stats(self._memory_history),
        }
    
    def get_average_fps(self, window_seconds: float = 1.0) -> float:
        """Get average FPS over a time window."""
        if not self._fps_history:
            return 0.0
        
        now = time.time()
        cutoff = now - window_seconds
        
        recent = [v for t, v in self._fps_history if t > cutoff]
        return np.mean(recent) if recent else 0.0
    
    def get_percentile_latency(self, percentile: float = 95.0) -> float:
        """Get latency at a specific percentile."""
        if not self._total_latency_history:
            return 0.0
        
        values = [v for _, v in self._total_latency_history]
        return np.percentile(values, percentile)
    
    def get_uptime(self) -> float:
        """Get total tracking uptime in seconds."""
        return time.time() - self._start_time
    
    def get_total_frames(self) -> int:
        """Get total frames processed."""
        return self._frame_count
    
    def reset(self) -> None:
        """Reset all statistics."""
        self._fps_history.clear()
        self._capture_fps_history.clear()
        self._detection_fps_history.clear()
        self._triangulation_fps_history.clear()
        self._total_latency_history.clear()
        self._latency_breakdown_history.clear()
        self._valid_joints_history.clear()
        self._confidence_history.clear()
        self._jitter_history.clear()
        self._cpu_history.clear()
        self._memory_history.clear()
        self._gpu_history.clear()
        self._timing_histogram = [0] * (len(self._timing_bins) + 1)
        self._frame_count = 0
        self._start_time = time.time()
    
    def get_graph_data(
        self,
        metric: str,
        max_points: int = 300,
    ) -> Tuple[List[float], List[float]]:
        """
        Get time-series data for graphing.
        
        Args:
            metric: Metric name ('fps', 'latency', 'joints', 'confidence', 'jitter', 'cpu', 'memory')
            max_points: Maximum number of points to return
            
        Returns:
            Tuple of (times, values) lists
        """
        history_map = {
            'fps': self._fps_history,
            'latency': self._total_latency_history,
            'joints': self._valid_joints_history,
            'confidence': self._confidence_history,
            'jitter': self._jitter_history,
            'cpu': self._cpu_history,
            'memory': self._memory_history,
            'gpu': self._gpu_history,
        }
        
        history = history_map.get(metric, self._fps_history)
        
        if not history:
            return [], []
        
        # Downsample if needed
        data = list(history)
        if len(data) > max_points:
            step = len(data) // max_points
            data = data[::step]
        
        times = [t for t, _ in data]
        values = [v for _, v in data]
        
        # Normalize times to relative
        if times:
            t0 = times[0]
            times = [t - t0 for t in times]
        
        return times, values
