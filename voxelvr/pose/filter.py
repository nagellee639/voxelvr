"""
Temporal Filtering for Pose Smoothing

Implements the One-Euro Filter and other temporal smoothing
methods to reduce jitter in tracking data.

The One-Euro Filter is particularly effective because:
- It adapts cutoff frequency based on signal velocity
- Reduces jitter during slow movements (aggressive smoothing)
- Maintains responsiveness during fast movements (minimal lag)
"""

import numpy as np
from typing import Optional, Dict, Union, List, Any
from collections import deque
import time


class OneEuroFilter:
    """
    One-Euro Filter for 1D signal smoothing.
    
    Based on: "1€ Filter: A Simple Speed-based Low-pass Filter for 
    Noisy Input in Interactive Systems" by Géry Casiez et al.
    
    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing.
        beta: Speed coefficient. Higher = less lag during fast movement.
        d_cutoff: Cutoff frequency for derivative estimation.
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.t_prev: Optional[float] = None
    
    def _alpha(self, cutoff: float, dt: float) -> float:
        """Compute smoothing factor alpha from cutoff frequency."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def reset(self) -> None:
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def filter(self, x: float, t: Optional[float] = None) -> float:
        """
        Filter a single value.
        
        Args:
            x: Input value
            t: Timestamp (optional, uses real time if not provided)
            
        Returns:
            Filtered value
        """
        if t is None:
            t = time.time()
        
        if self.x_prev is None:
            # First sample
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        # Compute time delta
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-6  # Avoid division by zero
        
        # Estimate derivative
        dx = (x - self.x_prev) / dt
        
        # Filter derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter value
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat


class OneEuroFilter3D:
    """One-Euro Filter for 3D vectors."""
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        self.filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff)
            for _ in range(3)
        ]
    
    def reset(self) -> None:
        for f in self.filters:
            f.reset()
    
    def filter(self, v: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """Filter a 3D vector."""
        return np.array([
            self.filters[i].filter(v[i], t)
            for i in range(3)
        ])


class PoseFilter:
    """
    Filter for full body pose (17 joints x 3D).
    
    Maintains separate One-Euro filters for each joint
    to provide smooth, low-jitter tracking.
    """
    
    def __init__(
        self,
        num_joints: int = 17,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        """
        Initialize pose filter.
        
        Args:
            num_joints: Number of pose joints (default 17 for COCO)
            min_cutoff: Minimum cutoff frequency (lower = more smoothing)
            beta: Speed coefficient (higher = less lag during movement)
            d_cutoff: Cutoff for derivative estimation
        """
        self.num_joints = num_joints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.joint_filters: Dict[int, OneEuroFilter3D] = {}
        
        for i in range(num_joints):
            self.joint_filters[i] = OneEuroFilter3D(min_cutoff, beta, d_cutoff)
    
    def reset(self) -> None:
        """Reset all filters."""
        for f in self.joint_filters.values():
            f.reset()
    
    def filter(
        self,
        positions: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        """
        Filter a full pose.
        
        Args:
            positions: (num_joints, 3) joint positions
            valid_mask: (num_joints,) boolean mask for valid joints
            timestamp: Current timestamp
            
        Returns:
            Filtered (num_joints, 3) positions
        """
        if timestamp is None:
            timestamp = time.time()
        
        if valid_mask is None:
            valid_mask = np.ones(self.num_joints, dtype=bool)
        
        # Robustness check: ignore updates with NaNs
        if np.any(np.isnan(positions)):
            # If input is junk, return previous state or just input (if first frame)
            # Ideally we want to return the last valid filtered state
            # For now, let's just use the current valid state without updating
            if any(f.filters[0].x_prev is not None for f in self.joint_filters.values()):
                # Construct result from current filter states
                result = np.zeros_like(positions)
                for i in range(self.num_joints):
                     # Current estimated position
                     result[i] = [
                         self.joint_filters[i].filters[0].x_prev if self.joint_filters[i].filters[0].x_prev is not None else positions[i,0],
                         self.joint_filters[i].filters[1].x_prev if self.joint_filters[i].filters[1].x_prev is not None else positions[i,1],
                         self.joint_filters[i].filters[2].x_prev if self.joint_filters[i].filters[2].x_prev is not None else positions[i,2],
                     ]
                return result
            else:
                return np.nan_to_num(positions) # Return zeros if first frame is NaN
        
        filtered = positions.copy()
        
        for i in range(min(self.num_joints, len(positions))):
            if valid_mask[i] and not np.any(np.isnan(positions[i])):
                filtered[i] = self.joint_filters[i].filter(positions[i], timestamp)
            # If joint is invalid, we could either:
            # 1. Reset the filter (causes jump when joint becomes valid)
            # 2. Keep last value (maintains continuity)
            # Currently keeping last value by not updating
        
        return filtered
    
    def update_parameters(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        """Update filter parameters (recreates filters)."""
        if min_cutoff is not None:
            self.min_cutoff = min_cutoff
        if beta is not None:
            self.beta = beta
        if d_cutoff is not None:
            self.d_cutoff = d_cutoff
        
        # Recreate filters with new parameters
        for i in range(self.num_joints):
            self.joint_filters[i] = OneEuroFilter3D(
                self.min_cutoff, self.beta, self.d_cutoff
            )
    
    def get_filter_state(self) -> Dict[int, Dict[str, Any]]:
        """
        Get current filter state for each joint.
        
        Returns:
            Dict mapping joint index to filter state info
        """
        state = {}
        for i, f in self.joint_filters.items():
            state[i] = {
                'has_previous': f.filters[0].x_prev is not None,
                'last_value': [filt.x_prev for filt in f.filters] if f.filters[0].x_prev else None,
            }
        return state


class JitterMeasurement:
    """
    Utility for measuring pose jitter in real-time.
    
    Maintains a sliding window of recent positions and
    calculates jitter metrics.
    """
    
    def __init__(
        self,
        num_joints: int = 17,
        window_size: int = 30,  # ~1 second at 30fps
    ):
        """
        Initialize jitter measurement.
        
        Args:
            num_joints: Number of pose joints
            window_size: Number of frames to keep in history
        """
        self.num_joints = num_joints
        self.window_size = window_size
        
        # Position history per joint
        self._history: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(num_joints)
        }
        
        # Velocity history per joint
        self._velocity_history: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(num_joints)
        }
        
        self._last_positions: Optional[np.ndarray] = None
        self._last_timestamp: float = 0.0
    
    def update(
        self,
        positions: np.ndarray,
        valid_mask: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update with new pose data.
        
        Args:
            positions: (num_joints, 3) joint positions
            valid_mask: (num_joints,) boolean mask
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        for i in range(min(self.num_joints, len(positions))):
            if valid_mask[i]:
                self._history[i].append(positions[i].copy())
                
                # Calculate velocity
                if self._last_positions is not None and valid_mask[i]:
                    dt = timestamp - self._last_timestamp
                    if dt > 0:
                        velocity = np.linalg.norm(positions[i] - self._last_positions[i]) / dt
                        self._velocity_history[i].append(velocity)
        
        self._last_positions = positions.copy()
        self._last_timestamp = timestamp
    
    def get_jitter(self, joint_indices: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Calculate jitter metrics.
        
        Args:
            joint_indices: Specific joints to measure (all if None)
            
        Returns:
            Dict with jitter metrics:
            - 'position_std_m': Standard deviation in meters
            - 'position_std_mm': Standard deviation in millimeters
            - 'max_deviation_mm': Maximum deviation from mean in mm
            - 'velocity_std': Velocity variation
        """
        if joint_indices is None:
            joint_indices = list(range(self.num_joints))
        
        all_stds = []
        all_max_devs = []
        all_vel_stds = []
        
        for idx in joint_indices:
            if len(self._history[idx]) >= 5:
                positions = np.array(list(self._history[idx]))
                mean_pos = np.mean(positions, axis=0)
                
                # Per-axis standard deviation, averaged
                std = np.mean(np.std(positions, axis=0))
                all_stds.append(std)
                
                # Maximum deviation from mean
                deviations = np.linalg.norm(positions - mean_pos, axis=1)
                all_max_devs.append(np.max(deviations))
            
            if len(self._velocity_history[idx]) >= 5:
                velocities = np.array(list(self._velocity_history[idx]))
                all_vel_stds.append(np.std(velocities))
        
        return {
            'position_std_m': np.mean(all_stds) if all_stds else 0.0,
            'position_std_mm': np.mean(all_stds) * 1000 if all_stds else 0.0,
            'max_deviation_mm': np.mean(all_max_devs) * 1000 if all_max_devs else 0.0,
            'velocity_std': np.mean(all_vel_stds) if all_vel_stds else 0.0,
        }
    
    def reset(self) -> None:
        """Clear all history."""
        for hist in self._history.values():
            hist.clear()
        for hist in self._velocity_history.values():
            hist.clear()
        self._last_positions = None
        self._last_timestamp = 0.0


class ExponentialMovingAverage:
    """
    Simple exponential moving average filter.
    
    Alternative to One-Euro when parameter tuning is difficult.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor (0-1). Lower = more smoothing.
        """
        self.alpha = alpha
        self.value: Optional[np.ndarray] = None
    
    def reset(self) -> None:
        self.value = None
    
    def filter(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Filter a value (scalar or array)."""
        if self.value is None:
            if isinstance(x, np.ndarray):
                self.value = x.copy()
            else:
                self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        
        if isinstance(self.value, np.ndarray):
            return self.value.copy()
        return self.value


class PoseFilterEMA:
    """EMA-based pose filter for simpler smoothing."""
    
    def __init__(self, num_joints: int = 17, alpha: float = 0.3):
        self.filters = [ExponentialMovingAverage(alpha) for _ in range(num_joints)]
    
    def reset(self) -> None:
        for f in self.filters:
            f.reset()
    
    def filter(
        self,
        positions: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if valid_mask is None:
            valid_mask = np.ones(len(positions), dtype=bool)
        
        filtered = positions.copy()
        for i, (pos, valid) in enumerate(zip(positions, valid_mask)):
            if valid and i < len(self.filters):
                filtered[i] = self.filters[i].filter(pos)
        
        return filtered
