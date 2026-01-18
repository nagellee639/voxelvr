"""
Parameter Optimizer

Provides automatic and profile-based optimization of filter parameters
to balance between jitter reduction and latency.
"""

import numpy as np
import time
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class FilterProfile(Enum):
    """Predefined filter profiles optimizing for different use cases."""
    LOW_JITTER = "low_jitter"
    BALANCED = "balanced"
    LOW_LATENCY = "low_latency"
    PRECISION = "precision"
    CUSTOM = "custom"


@dataclass
class ProfileSettings:
    """Settings for a filter profile."""
    min_cutoff: float
    beta: float
    d_cutoff: float = 1.0
    description: str = ""


# Predefined profile configurations
PROFILE_PRESETS: Dict[FilterProfile, ProfileSettings] = {
    FilterProfile.LOW_JITTER: ProfileSettings(
        min_cutoff=0.5,
        beta=0.1,
        d_cutoff=0.5,
        description="Maximum smoothing, higher latency - best for slow movements"
    ),
    FilterProfile.BALANCED: ProfileSettings(
        min_cutoff=1.0,
        beta=0.5,
        d_cutoff=1.0,
        description="Good compromise between smoothness and responsiveness"
    ),
    FilterProfile.LOW_LATENCY: ProfileSettings(
        min_cutoff=2.0,
        beta=1.5,
        d_cutoff=1.5,
        description="Minimal smoothing, fastest response - may have jitter"
    ),
    FilterProfile.PRECISION: ProfileSettings(
        min_cutoff=0.8,
        beta=0.3,
        d_cutoff=0.8,
        description="Optimized for slow, precise movements"
    ),
}


@dataclass
class JitterMetrics:
    """Metrics for measuring tracking jitter."""
    position_std: float = 0.0  # Standard deviation in mm
    velocity_std: float = 0.0  # Velocity variation
    max_deviation: float = 0.0  # Maximum deviation from mean in mm
    sample_count: int = 0


@dataclass
class LatencyMetrics:
    """Metrics for estimating filter latency."""
    estimated_latency_ms: float = 0.0
    filter_delay_frames: float = 0.0


class ParameterOptimizer:
    """
    Dynamically optimizes filter parameters based on real-time jitter/latency analysis.
    
    Features:
    - Continuous background monitoring of jitter and latency
    - Profile-based presets for quick switching
    - Auto-adjustment mode that adapts to movement patterns
    - Manual override capability
    """
    
    def __init__(
        self,
        initial_profile: FilterProfile = FilterProfile.BALANCED,
        history_size: int = 60,  # 2 seconds at 30fps
        update_interval: float = 0.5,  # Seconds between parameter updates
    ):
        """
        Initialize the parameter optimizer.
        
        Args:
            initial_profile: Starting filter profile
            history_size: Number of frames to keep for analysis
            update_interval: Minimum time between parameter adjustments
        """
        self.current_profile = initial_profile
        self.history_size = history_size
        self.update_interval = update_interval
        
        # Current parameters
        self._min_cutoff = PROFILE_PRESETS[initial_profile].min_cutoff
        self._beta = PROFILE_PRESETS[initial_profile].beta
        self._d_cutoff = PROFILE_PRESETS[initial_profile].d_cutoff
        
        # Auto-adjustment state
        self.auto_adjust_enabled = True
        self.manual_override = False
        
        # Position history for jitter measurement (per joint)
        self._position_history: Dict[int, deque] = {}
        self._velocity_history: Dict[int, deque] = {}
        
        # Timing
        self._last_update_time = 0.0
        self._last_positions: Optional[np.ndarray] = None
        self._last_timestamp: float = 0.0
        
        # Parameter change callbacks
        self._callbacks: List[Callable[[float, float, float], None]] = []
        
        # Parameter history for visualization
        self.parameter_history: deque = deque(maxlen=300)  # 10 seconds at 30fps
        
        # Target metrics based on profile
        self._update_targets()
    
    def _update_targets(self) -> None:
        """Update target metrics based on current profile."""
        if self.current_profile == FilterProfile.LOW_JITTER:
            self._target_jitter_mm = 0.5
            self._target_latency_ms = 100.0
        elif self.current_profile == FilterProfile.LOW_LATENCY:
            self._target_jitter_mm = 3.0
            self._target_latency_ms = 20.0
        elif self.current_profile == FilterProfile.PRECISION:
            self._target_jitter_mm = 1.0
            self._target_latency_ms = 60.0
        else:  # BALANCED or CUSTOM
            self._target_jitter_mm = 1.5
            self._target_latency_ms = 50.0
    
    @property
    def min_cutoff(self) -> float:
        return self._min_cutoff
    
    @property
    def beta(self) -> float:
        return self._beta
    
    @property
    def d_cutoff(self) -> float:
        return self._d_cutoff
    
    def add_callback(self, callback: Callable[[float, float, float], None]) -> None:
        """Add callback for parameter changes. Callback receives (min_cutoff, beta, d_cutoff)."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[float, float, float], None]) -> None:
        """Remove a parameter change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_callbacks(self) -> None:
        """Notify all callbacks of parameter change."""
        for callback in self._callbacks:
            try:
                callback(self._min_cutoff, self._beta, self._d_cutoff)
            except Exception as e:
                print(f"Parameter callback error: {e}")
    
    def set_profile(self, profile: FilterProfile) -> None:
        """
        Switch to a predefined profile.
        
        Args:
            profile: The profile to switch to
        """
        self.current_profile = profile
        self._update_targets()
        
        if profile != FilterProfile.CUSTOM:
            settings = PROFILE_PRESETS[profile]
            # Use smooth=False for explicit profile changes to apply immediately
            self._set_parameters(settings.min_cutoff, settings.beta, settings.d_cutoff, smooth=False)
    
    def set_manual_parameters(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        """
        Manually set filter parameters and enable manual override.
        
        Args:
            min_cutoff: Minimum cutoff frequency (0.1 - 10.0)
            beta: Speed coefficient (0.0 - 2.0)
            d_cutoff: Derivative cutoff (0.1 - 5.0)
        """
        self.manual_override = True
        self.current_profile = FilterProfile.CUSTOM
        
        if min_cutoff is not None:
            self._min_cutoff = np.clip(min_cutoff, 0.1, 10.0)
        if beta is not None:
            self._beta = np.clip(beta, 0.0, 2.0)
        if d_cutoff is not None:
            self._d_cutoff = np.clip(d_cutoff, 0.1, 5.0)
        
        self._notify_callbacks()
    
    def enable_auto_adjust(self, enabled: bool = True) -> None:
        """Enable or disable automatic parameter adjustment."""
        self.auto_adjust_enabled = enabled
        if enabled:
            self.manual_override = False
    
    def disable_manual_override(self) -> None:
        """Disable manual override and return to auto-adjustment."""
        self.manual_override = False
        self.set_profile(self.current_profile)
    
    def _set_parameters(
        self,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
        smooth: bool = True,
    ) -> None:
        """Set parameters with optional smoothing to avoid jumps."""
        if smooth:
            # Smooth transition (exponential moving average)
            alpha = 0.1
            self._min_cutoff = alpha * min_cutoff + (1 - alpha) * self._min_cutoff
            self._beta = alpha * beta + (1 - alpha) * self._beta
            self._d_cutoff = alpha * d_cutoff + (1 - alpha) * self._d_cutoff
        else:
            self._min_cutoff = min_cutoff
            self._beta = beta
            self._d_cutoff = d_cutoff
        
        # Record history
        self.parameter_history.append({
            'time': time.time(),
            'min_cutoff': self._min_cutoff,
            'beta': self._beta,
            'd_cutoff': self._d_cutoff,
        })
        
        self._notify_callbacks()
    
    def update(
        self,
        positions: np.ndarray,
        valid_mask: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update the optimizer with new pose data.
        
        Call this every frame to enable jitter measurement and auto-adjustment.
        
        Args:
            positions: (N, 3) array of joint positions
            valid_mask: (N,) boolean mask for valid joints
            timestamp: Current timestamp (uses time.time() if not provided)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update position history for valid joints
        for i, (pos, valid) in enumerate(zip(positions, valid_mask)):
            if valid:
                if i not in self._position_history:
                    self._position_history[i] = deque(maxlen=self.history_size)
                    self._velocity_history[i] = deque(maxlen=self.history_size)
                
                self._position_history[i].append(pos.copy())
                
                # Calculate velocity if we have previous positions
                if self._last_positions is not None and valid_mask[i]:
                    dt = timestamp - self._last_timestamp
                    if dt > 0:
                        velocity = (pos - self._last_positions[i]) / dt
                        self._velocity_history[i].append(np.linalg.norm(velocity))
        
        self._last_positions = positions.copy()
        self._last_timestamp = timestamp
        
        # Auto-adjust if enabled and not in manual override
        if self.auto_adjust_enabled and not self.manual_override:
            if timestamp - self._last_update_time >= self.update_interval:
                self._auto_adjust()
                self._last_update_time = timestamp
    
    def _auto_adjust(self) -> None:
        """Perform automatic parameter adjustment based on current metrics."""
        jitter = self.measure_jitter()
        
        if jitter.sample_count < 10:
            return  # Not enough data
        
        # Calculate average velocity (movement speed)
        avg_velocity = 0.0
        velocity_count = 0
        for hist in self._velocity_history.values():
            if len(hist) > 0:
                avg_velocity += np.mean(list(hist))
                velocity_count += 1
        
        if velocity_count > 0:
            avg_velocity /= velocity_count
        
        # Adjust parameters based on movement and jitter
        current_jitter_mm = jitter.position_std * 1000  # Convert to mm
        
        if current_jitter_mm > self._target_jitter_mm * 1.5:
            # Too much jitter - increase smoothing
            new_min_cutoff = max(0.1, self._min_cutoff * 0.95)
            new_beta = max(0.0, self._beta * 0.95)
        elif current_jitter_mm < self._target_jitter_mm * 0.5:
            # Very smooth - can reduce smoothing for less latency
            new_min_cutoff = min(10.0, self._min_cutoff * 1.05)
            new_beta = min(2.0, self._beta * 1.05)
        else:
            # Within target range - minor adjustments based on velocity
            if avg_velocity > 1.0:  # Fast movement
                # Increase beta for less lag during movement
                new_min_cutoff = self._min_cutoff
                new_beta = min(2.0, self._beta * 1.02)
            else:  # Slow or stationary
                # Increase smoothing
                new_min_cutoff = max(0.1, self._min_cutoff * 0.99)
                new_beta = self._beta
        
        self._set_parameters(new_min_cutoff, new_beta, self._d_cutoff, smooth=True)
    
    def measure_jitter(self, joint_indices: Optional[List[int]] = None) -> JitterMetrics:
        """
        Measure current jitter from position history.
        
        Args:
            joint_indices: Specific joints to measure (all if None)
            
        Returns:
            JitterMetrics with current jitter measurements
        """
        if joint_indices is None:
            joint_indices = list(self._position_history.keys())
        
        all_stds = []
        all_max_devs = []
        total_samples = 0
        
        for idx in joint_indices:
            if idx in self._position_history and len(self._position_history[idx]) >= 5:
                positions = np.array(list(self._position_history[idx]))
                mean_pos = np.mean(positions, axis=0)
                
                # Calculate per-axis standard deviation
                std = np.std(positions, axis=0)
                all_stds.append(np.mean(std))  # Average across axes
                
                # Maximum deviation from mean
                deviations = np.linalg.norm(positions - mean_pos, axis=1)
                all_max_devs.append(np.max(deviations))
                
                total_samples += len(positions)
        
        if not all_stds:
            return JitterMetrics()
        
        # Velocity jitter
        velocity_stds = []
        for idx in joint_indices:
            if idx in self._velocity_history and len(self._velocity_history[idx]) >= 5:
                velocities = np.array(list(self._velocity_history[idx]))
                velocity_stds.append(np.std(velocities))
        
        return JitterMetrics(
            position_std=np.mean(all_stds),
            velocity_std=np.mean(velocity_stds) if velocity_stds else 0.0,
            max_deviation=np.mean(all_max_devs),
            sample_count=total_samples,
        )
    
    def estimate_latency(self) -> LatencyMetrics:
        """
        Estimate the filter latency based on current parameters.
        
        Returns:
            LatencyMetrics with latency estimates
        """
        # Approximate latency based on One-Euro filter characteristics
        # Lower min_cutoff = more smoothing = more latency
        # Higher beta = less latency during movement
        
        # Base latency from min_cutoff (inverse relationship)
        base_latency_ms = (1.0 / self._min_cutoff) * 30  # Approximate
        
        # Beta reduces effective latency during movement
        effective_latency_ms = base_latency_ms * (1.0 - 0.3 * self._beta)
        
        # Estimate frame delay
        frame_delay = effective_latency_ms / 33.3  # At 30fps
        
        return LatencyMetrics(
            estimated_latency_ms=max(0, effective_latency_ms),
            filter_delay_frames=max(0, frame_delay),
        )
    
    def get_profile_info(self, profile: Optional[FilterProfile] = None) -> ProfileSettings:
        """Get information about a profile."""
        if profile is None:
            profile = self.current_profile
        
        if profile == FilterProfile.CUSTOM:
            return ProfileSettings(
                min_cutoff=self._min_cutoff,
                beta=self._beta,
                d_cutoff=self._d_cutoff,
                description="Custom user-defined settings"
            )
        
        return PROFILE_PRESETS[profile]
    
    def reset_history(self) -> None:
        """Clear position/velocity history. Call when tracking is restarted."""
        self._position_history.clear()
        self._velocity_history.clear()
        self._last_positions = None
        self._last_timestamp = 0.0
