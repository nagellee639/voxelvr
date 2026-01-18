"""
Calibration Panel

Step-by-step calibration wizard with ChArUco board support.
"""

import numpy as np
import time
from typing import Optional, Dict, List, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CalibrationStep(Enum):
    """Calibration workflow steps."""
    IDLE = "idle"
    EXPORT_BOARD = "export_board"
    CALIBRATION = "calibration"  # Unified intrinsics + extrinsics
    COMPLETE = "complete"


@dataclass
class CameraCalibrationStatus:
    """Calibration status for a single camera."""
    camera_id: int
    board_visible: bool = False
    
    # Intrinsic progress
    intrinsic_frames_captured: int = 0
    intrinsic_frames_required: int = 20
    intrinsic_complete: bool = False
    intrinsic_result: Optional[Any] = None  # CameraIntrinsics when complete
    intrinsic_error: float = 0.0
    intrinsic_computing: bool = False
    
    # Pairwise extrinsics (frames captured with each other camera)
    pairwise_frames: Dict[int, int] = field(default_factory=dict)
    pairwise_required: int = 10  # Frames needed per camera pair



@dataclass
class ExtrinsicsProgress:
    """Overall extrinsics calibration progress (all cameras simultaneously)."""
    all_cameras_frames: int = 0
    all_cameras_required: int = 30
    complete: bool = False
    result: Optional[Any] = None  # MultiCameraCalibration when complete
    computing: bool = False


@dataclass
class CalibrationState:
    """Overall calibration state."""
    current_step: CalibrationStep = CalibrationStep.IDLE
    cameras: Dict[int, CameraCalibrationStatus] = field(default_factory=dict)
    extrinsics: ExtrinsicsProgress = field(default_factory=ExtrinsicsProgress)
    all_cameras_visible: bool = False
    error_message: str = ""
    is_running: bool = False
    auto_capture: bool = True  # Default to true for convenience


class CalibrationPanel:
    """
    Calibration wizard UI component.
    
    Provides step-by-step guidance for:
    1. Exporting printable ChArUco board
    2. Intrinsic calibration per camera
    3. Extrinsic calibration for multi-camera setup
    4. Verification and quality metrics
    """
    
    def __init__(
        self,
        charuco_squares_x: int = 5,
        charuco_squares_y: int = 5,
        charuco_square_length: float = 0.04,
        charuco_marker_length: float = 0.03,
        charuco_dict: str = "DICT_6X6_250",
        intrinsic_frames_required: int = 20,
        extrinsic_frames_required: int = 30,
    ):
        """
        Initialize calibration panel.
        
        Args:
            charuco_squares_x: Number of squares in X direction
            charuco_squares_y: Number of squares in Y direction
            charuco_square_length: Square size in meters
            charuco_marker_length: Marker size in meters
            charuco_dict: ArUco dictionary name
            intrinsic_frames_required: Frames needed for intrinsic calibration
            extrinsic_frames_required: Frames needed for extrinsic calibration
        """
        self.charuco_squares_x = charuco_squares_x
        self.charuco_squares_y = charuco_squares_y
        self.charuco_square_length = charuco_square_length
        self.charuco_marker_length = charuco_marker_length
        self.charuco_dict = charuco_dict
        self.intrinsic_frames_required = intrinsic_frames_required
        self.extrinsic_frames_required = extrinsic_frames_required
        
        # State
        self._state = CalibrationState()
        self._state.extrinsics.all_cameras_required = extrinsic_frames_required
        self._camera_ids: List[int] = []
        
        # Callbacks
        self._step_callbacks: List[Callable[[CalibrationStep], None]] = []
        self._progress_callbacks: List[Callable[[CalibrationState], None]] = []
        
        # Captured data storage
        # intrinsic_frames: Dict[cam_id, List[Dict with 'frame', 'corners', 'ids']]
        self._intrinsic_frames: Dict[int, List[Dict[str, Any]]] = {}
        # extrinsic_captures: List of synchronized multi-camera captures (legacy - all cameras)
        self._extrinsic_captures: List[Dict[str, Any]] = []
        # pairwise_captures: Dict[(cam_a, cam_b), List[Dict with 'frames', 'detections']]]
        self._pairwise_captures: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        self._pairwise_frames_required = 10  # Frames needed per camera pair
        
        # Board detection results
        self._board = None
        self._aruco_dict = None
        
        # Auto-capture state
        self._last_capture_time: Dict[int, float] = {}
        self._last_capture_corners: Dict[int, np.ndarray] = {}
        self._capture_cooldown = 1.0  # Seconds
        self._movement_threshold = 50.0  # Pixel distance sum
        
        # Background computation
        self._computation_threads: Dict[int, Any] = {}  # cam_id -> thread
    
    @property
    def state(self) -> CalibrationState:
        """Get current calibration state."""
        return self._state
    
    @property
    def current_step(self) -> CalibrationStep:
        """Get current calibration step."""
        return self._state.current_step
    
    def set_cameras(self, camera_ids: List[int]) -> None:
        """
        Set the cameras to calibrate.
        
        Args:
            camera_ids: List of camera IDs
        """
        self._camera_ids = camera_ids
        self._state.cameras = {
            cam_id: CameraCalibrationStatus(
                camera_id=cam_id,
                intrinsic_frames_required=self.intrinsic_frames_required,
            )
            for cam_id in camera_ids
        }
    
    def add_step_callback(self, callback: Callable[[CalibrationStep], None]) -> None:
        """Add callback for step changes."""
        self._step_callbacks.append(callback)
    
    def add_progress_callback(self, callback: Callable[[CalibrationState], None]) -> None:
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)
    
    def _notify_step_change(self) -> None:
        """Notify callbacks of step change."""
        for callback in self._step_callbacks:
            try:
                callback(self._state.current_step)
            except Exception as e:
                print(f"Step callback error: {e}")
    
    def _notify_progress(self) -> None:
        """Notify callbacks of progress update."""
        for callback in self._progress_callbacks:
            try:
                callback(self._state)
            except Exception as e:
                print(f"Progress callback error: {e}")
    
    def get_step_instructions(self) -> str:
        """Get user instructions for current step."""
        step = self._state.current_step
        
        if step == CalibrationStep.IDLE:
            return "Click 'Start Calibration' to begin the calibration wizard."
        
        elif step == CalibrationStep.EXPORT_BOARD:
            return (
                "Step 1: Export and Print ChArUco Board\n\n"
                f"Board size: {self.charuco_squares_x}x{self.charuco_squares_y} squares\n"
                f"Square size: {self.charuco_square_length * 100:.1f} cm\n"
                f"Marker size: {self.charuco_marker_length * 100:.1f} cm\n\n"
                "1. Click 'Export PDF' to save the calibration board\n"
                "2. Print at 100% scale (no scaling)\n"
                "3. Measure a square to verify it matches the expected size\n"
                "4. Attach the board to a flat, rigid surface\n"
                "5. Click 'Start Calibration' when ready"
            )
        
        elif step == CalibrationStep.CALIBRATION:
            visible_count = sum(1 for s in self._state.cameras.values() if s.board_visible)
            return (
                "Step 2: Calibration (Intrinsics + Extrinsics)\n\n"
                f"Cameras seeing board: {visible_count}/{len(self._camera_ids)}\n\n"
                "Move the ChArUco board around your tracking space:\n"
                "• Hold it in front of each camera from different angles\n"
                "• Include corners and edges of each camera's view\n"
                "• When multiple cameras see it, we capture extrinsics too\n"
                "• For best results, show it to all cameras together\n\n"
                "Progress bars below show what still needs to be captured.\n"
                "Frames are captured automatically based on movement."
            )
        
        elif step == CalibrationStep.COMPLETE:
            return (
                "Calibration Complete!\n\n"
                "Your cameras are now calibrated and ready for tracking.\n"
                "You can close this panel and start tracking."
            )
        
        
        return ""
    
    def start_calibration(self) -> None:
        """Start the calibration wizard."""
        if len(self._camera_ids) == 0:
            self._state.error_message = "No cameras configured"
            return
        
        self._state.is_running = True
        self._state.current_step = CalibrationStep.EXPORT_BOARD
        self._state.error_message = ""
        self._notify_step_change()
    
    def begin_calibration(self) -> None:
        """Begin the actual calibration capture phase."""
        self._state.current_step = CalibrationStep.CALIBRATION
        self._intrinsic_frames = {cam_id: [] for cam_id in self._camera_ids}
        self._extrinsic_captures = []
        self._notify_step_change()
    
    def cancel_calibration(self) -> None:
        """Cancel the current calibration."""
        self._state.is_running = False
        self._state.current_step = CalibrationStep.IDLE
        self._intrinsic_frames.clear()
        self._extrinsic_captures.clear()
        self._notify_step_change()
    
    
    def finish_calibration(self) -> None:
        """Mark calibration as complete."""
        self._state.current_step = CalibrationStep.COMPLETE
        self._state.is_running = False
        self._notify_step_change()
    
    
    def toggle_auto_capture(self) -> None:
        """Toggle auto-capture mode."""
        self._state.auto_capture = not self._state.auto_capture
        self._notify_progress()
    
    def should_auto_capture(self, camera_id: int, corners: np.ndarray, require_all_cameras: bool = False) -> bool:
        """
        Check if a frame should be auto-captured.
        
        Args:
            camera_id: ID of camera provided
            corners: Detected corners
            require_all_cameras: If True, checks if ALL cameras see the board now
        """
        if not self._state.auto_capture:
            return False
            
        now = time.time()
        
        # If requiring all cameras, we check global state
        if require_all_cameras and not self._state.all_cameras_visible:
            return False
            
        # Check global cooldown if requiring all cameras (batch capture)
        # Otherwise check per-camera cooldown
        last_time = self._last_capture_time.get(camera_id, 0)
        if now - last_time < self._capture_cooldown:
            return False
            
        # Check movement
        # If this camera recorded a previous capture, check if we moved enough
        if camera_id in self._last_capture_corners:
            last_corners = self._last_capture_corners[camera_id]
            
            # Simple check: if number of corners differs significantly, it's a new pose
            if len(corners) != len(last_corners):
                return True
                
            # Check displacement (average movement of corners)
            # Only compare matching shape
            try:
                # Euclidean distance
                dist = np.linalg.norm(corners - last_corners, axis=2)
                mean_dist = np.mean(dist)
                
                if mean_dist < self._movement_threshold:
                     return False
                     
            except Exception:
                # Shape mismatch or other error
                return True

        return True
    
    def export_board_pdf(self, output_path: Path) -> bool:
        """
        Export the ChArUco board as a PDF.
        
        Args:
            output_path: Path to save the PDF
            
        Returns:
            True if successful
        """
        try:
            from ..calibration.charuco import generate_charuco_pdf_file
            
            # Ensure PDF extension
            pdf_path = output_path.with_suffix('.pdf')
            
            generate_charuco_pdf_file(
                pdf_path,
                self.charuco_squares_x,
                self.charuco_squares_y,
                self.charuco_square_length,
                self.charuco_marker_length,
                self.charuco_dict,
            )
            
            print(f"Exported calibration board to {pdf_path}")
            return True
            
        except Exception as e:
            self._state.error_message = f"Failed to export board: {e}"
            print(f"Export error: {e}")
            return False
    
    
    def process_frame_detections(
        self,
        detections: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process detection results from all cameras and automatically capture frames.
        
        Args:
            detections: Dict mapping camera_id to detection result with keys:
                - 'success': bool
                - 'corners': np.ndarray (if success)
                - 'ids': np.ndarray (if success)
                - 'frame': np.ndarray (optional, for capture)
        
        Returns:
            Dict with capture information:
                - 'intrinsics_captured': List[int] (camera IDs)
                - 'pairwise_captured': List[Tuple[int, int]] (camera ID pairs)
                - 'all_cameras_captured': bool
        """
        if self._state.current_step != CalibrationStep.CALIBRATION:
            return {'intrinsics_captured': [], 'pairwise_captured': [], 'all_cameras_captured': False}
        
        if not self._state.auto_capture:
            return {'intrinsics_captured': [], 'pairwise_captured': [], 'all_cameras_captured': False}
        
        now = time.time()
        result = {
            'intrinsics_captured': [],
            'pairwise_captured': [],
            'all_cameras_captured': False
        }
        
        # Update visibility status for all cameras
        visible_cameras = []
        for cam_id in self._camera_ids:
            det = detections.get(cam_id, {})
            is_visible = det.get('success', False)
            self._state.cameras[cam_id].board_visible = is_visible
            if is_visible:
                visible_cameras.append(cam_id)
        
        self._state.all_cameras_visible = len(visible_cameras) == len(self._camera_ids)
        
        # Process each visible camera for intrinsic capture
        for cam_id in visible_cameras:
            det = detections[cam_id]
            corners = det.get('corners')
            
            if corners is None or len(corners) < 4:
                continue
            
            # Check if this camera needs more intrinsic frames
            status = self._state.cameras[cam_id]
            if status.intrinsic_frames_captured >= status.intrinsic_frames_required:
                continue  # Already have enough
            
            if status.intrinsic_computing:
                continue  # Computation in progress
            
            # Check cooldown
            last_time = self._last_capture_time.get(cam_id, 0)
            if now - last_time < self._capture_cooldown:
                continue
            
            # Check movement
            if cam_id in self._last_capture_corners:
                last_corners = self._last_capture_corners[cam_id]
                if len(corners) == len(last_corners):
                    try:
                        dist = np.linalg.norm(corners - last_corners, axis=2)
                        mean_dist = np.mean(dist)
                        if mean_dist < self._movement_threshold:
                            continue  # Not enough movement
                    except Exception:
                        pass  # Shape mismatch, treat as new pose
            
            # Capture for intrinsics!
            frame = det.get('frame')
            ids = det.get('ids')
            
            if frame is not None and ids is not None:
                self._intrinsic_frames[cam_id].append({
                    'frame': frame.copy(),
                    'corners': corners.copy(),
                    'ids': ids.copy(),
                })
                
                status.intrinsic_frames_captured = len(self._intrinsic_frames[cam_id])
                self._last_capture_time[cam_id] = now
                self._last_capture_corners[cam_id] = corners.copy()
                
                result['intrinsics_captured'].append(cam_id)
                
                # Check if we should trigger computation
                if status.intrinsic_frames_captured >= status.intrinsic_frames_required:
                    self._trigger_intrinsic_computation(cam_id)
        
        # Process pairwise extrinsics (when 2+ cameras see the board)
        if len(visible_cameras) >= 2:
            # For simplicity, we'll use the first visible camera's cooldown as reference
            ref_cam = visible_cameras[0]
            ref_time = self._last_capture_time.get(ref_cam, 0)
            
            if now - ref_time >= self._capture_cooldown:
                # Store captures for all visible pairs
                for i, cam1 in enumerate(visible_cameras):
                    for cam2 in visible_cameras[i+1:]:
                        # Use ordered tuple as key (smaller camera ID first)
                        pair_key = (min(cam1, cam2), max(cam1, cam2))
                        
                        if pair_key not in self._pairwise_captures:
                            self._pairwise_captures[pair_key] = []
                        
                        # Store frames for this pair
                        pair_frames = {
                            cam1: detections[cam1].get('frame'),
                            cam2: detections[cam2].get('frame'),
                        }
                        
                        # Only store if we have valid frames
                        if pair_frames[cam1] is not None and pair_frames[cam2] is not None:
                            self._pairwise_captures[pair_key].append({
                                'frames': {k: v.copy() for k, v in pair_frames.items()},
                                'detections': {
                                    cam1: detections[cam1],
                                    cam2: detections[cam2],
                                },
                                'timestamp': now,
                            })
                            
                            result['pairwise_captured'].append((cam1, cam2))
                            
                            # Track pairwise progress in camera status
                            if cam2 not in self._state.cameras[cam1].pairwise_frames:
                                self._state.cameras[cam1].pairwise_frames[cam2] = 0
                            if cam1 not in self._state.cameras[cam2].pairwise_frames:
                                self._state.cameras[cam2].pairwise_frames[cam1] = 0
                            
                            self._state.cameras[cam1].pairwise_frames[cam2] += 1
                            self._state.cameras[cam2].pairwise_frames[cam1] += 1
                
                # Check if we have enough pairwise data to trigger computation
                self._check_and_trigger_pairwise_computation()
        
        # Legacy: Process overall extrinsics (when ALL cameras see the board)
        # This is kept as a fallback/optimization when all cameras can see the board
        if self._state.all_cameras_visible:
            if self._state.extrinsics.all_cameras_frames < self._state.extrinsics.all_cameras_required:
                # Check global cooldown (use first camera)
                if len(visible_cameras) > 0:
                    ref_cam = visible_cameras[0]
                    ref_time = self._last_capture_time.get(ref_cam, 0)
                    
                    if now - ref_time >= self._capture_cooldown:
                        # Capture synchronized frame set
                        frames_data = {}
                        detections_data = {}
                        
                        for cam_id in visible_cameras:
                            det = detections[cam_id]
                            frames_data[cam_id] = det.get('frame')
                            detections_data[cam_id] = det
                        
                        self._extrinsic_captures.append({
                            'frames': frames_data,
                            'detections': detections_data,
                            'timestamp': now,
                        })
                        
                        self._state.extrinsics.all_cameras_frames = len(self._extrinsic_captures)
                        result['all_cameras_captured'] = True
                        
                        # Update all camera timestamps
                        for cam_id in visible_cameras:
                            self._last_capture_time[cam_id] = now
        
        self._notify_progress()
        return result
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get current calibration progress for UI display.
        
        Returns dict with progress information for all cameras and overall extrinsics.
        """
        cameras_progress = {}
        
        for cam_id, status in self._state.cameras.items():
            intrinsic_percent = min(100, int(100 * status.intrinsic_frames_captured / status.intrinsic_frames_required))
            
            # Calculate average pairwise progress
            pairwise_total = 0
            pairwise_count = 0
            for other_cam, count in status.pairwise_frames.items():
                pairwise_total += min(100, int(100 * count / status.pairwise_required))
                pairwise_count += 1
            
            pairwise_percent = pairwise_total // pairwise_count if pairwise_count > 0 else 0
            
            cameras_progress[cam_id] = {
                'intrinsic_percent': intrinsic_percent,
                'intrinsic_frames': status.intrinsic_frames_captured,
                'intrinsic_required': status.intrinsic_frames_required,
                'intrinsic_complete': status.intrinsic_complete,
                'intrinsic_computing': status.intrinsic_computing,
                'pairwise_percent': pairwise_percent,
                'board_visible': status.board_visible,
            }
        
        # Build pairwise progress grid
        pairwise_progress = {}
        camera_ids = sorted(self._camera_ids)
        
        for i, cam_a in enumerate(camera_ids):
            for cam_b in camera_ids[i+1:]:
                pair_key = (cam_a, cam_b)
                captures = self._pairwise_captures.get(pair_key, [])
                frames_captured = len(captures)
                frames_required = self._pairwise_frames_required
                percent = min(100, int(100 * frames_captured / frames_required))
                is_ready = frames_captured >= frames_required
                
                pairwise_progress[pair_key] = {
                    'frames': frames_captured,
                    'required': frames_required,
                    'percent': percent,
                    'ready': is_ready,
                }
        
        # Check connectivity
        from ..calibration.pairwise_extrinsics import check_connectivity, PairwiseCalibrationResult
        
        mock_results = []
        for pair_key, pair_data in pairwise_progress.items():
            if pair_data['ready']:
                mock_results.append(PairwiseCalibrationResult(
                    camera_a=pair_key[0],
                    camera_b=pair_key[1],
                    transform_a_to_b=None,
                    reprojection_error=0.0,
                    num_frames_used=pair_data['frames'],
                ))
        
        is_connected, disconnected_cameras = check_connectivity(mock_results, camera_ids)
        
        overall_extrinsics_percent = min(100, int(100 * self._state.extrinsics.all_cameras_frames / self._state.extrinsics.all_cameras_required))
        all_intrinsics_complete = all(s.intrinsic_complete for s in self._state.cameras.values())
        all_ready = all_intrinsics_complete and self._state.extrinsics.complete
        
        return {
            'cameras': cameras_progress,
            'pairwise': pairwise_progress,
            'pairwise_connected': is_connected,
            'pairwise_disconnected_cameras': list(disconnected_cameras),
            'overall_extrinsics_percent': overall_extrinsics_percent,
            'overall_extrinsics_frames': self._state.extrinsics.all_cameras_frames,
            'overall_extrinsics_required': self._state.extrinsics.all_cameras_required,
            'overall_extrinsics_complete': self._state.extrinsics.complete,
            'overall_extrinsics_computing': self._state.extrinsics.computing,
            'all_intrinsics_complete': all_intrinsics_complete,
            'all_ready': all_ready,
            'all_cameras_visible': self._state.all_cameras_visible,
        }
    
    def _trigger_intrinsic_computation(self, camera_id: int) -> None:
        """Trigger background computation of intrinsic calibration for a camera."""
        import threading
        from ..calibration.intrinsics import calibrate_intrinsics
        from ..config import CalibrationConfig
        
        status = self._state.cameras[camera_id]
        status.intrinsic_computing = True
        
        def compute():
            try:
                # Extract frames from captured data
                frames = [cap['frame'] for cap in self._intrinsic_frames[camera_id]]
                
                # Create config
                config = CalibrationConfig(
                    charuco_squares_x=self.charuco_squares_x,
                    charuco_squares_y=self.charuco_squares_y,
                    charuco_square_length=self.charuco_square_length,
                    charuco_marker_length=self.charuco_marker_length,
                    charuco_dict=self.charuco_dict,
                )
                
                # Compute
                result = calibrate_intrinsics(frames, config, camera_id, f"Camera {camera_id}")
                
                if result:
                    status.intrinsic_complete = True
                    status.intrinsic_result = result
                    status.intrinsic_error = result.reprojection_error
                    print(f"Camera {camera_id} intrinsics complete! Error: {result.reprojection_error:.3f}")
                else:
                    print(f"Camera {camera_id} intrinsics computation failed")
                
            except Exception as e:
                print(f"Error computing intrinsics for camera {camera_id}: {e}")
            finally:
                status.intrinsic_computing = False
                self._notify_progress()
        
        thread = threading.Thread(target=compute, daemon=True)
        self._computation_threads[camera_id] = thread
        thread.start()
    
    def _check_and_trigger_pairwise_computation(self) -> None:
        """Check if we have enough pairwise data and trigger computation if ready."""
        # Check if all cameras have completed intrinsics
        if not all(s.intrinsic_complete for s in self._state.cameras.values()):
            return
            
        # Already computing or complete?
        if self._state.extrinsics.computing or self._state.extrinsics.complete:
            return
        
        # Check if we have enough pairs to form a spanning tree
        # For N cameras, we need at least N-1 connected pairs
        from ..calibration.pairwise_extrinsics import check_connectivity, PairwiseCalibrationResult
        
        # Build mock results to check connectivity
        mock_results = []
        for (cam_a, cam_b), captures in self._pairwise_captures.items():
            if len(captures) >= self._pairwise_frames_required:
                # Enough frames for this pair
                mock_results.append(PairwiseCalibrationResult(
                    camera_a=cam_a,
                    camera_b=cam_b,
                    transform_a_to_b=None,  # Not needed for connectivity check
                    reprojection_error=0.0,
                    num_frames_used=len(captures),
                ))
        
        is_connected, disconnected = check_connectivity(mock_results, self._camera_ids)
        
        if is_connected:
            print("All camera pairs calibrated! Triggering pairwise extrinsics computation...")
            self._trigger_pairwise_extrinsic_computation()
        else:
            # Could add partial feedback here
            pass
    
    def _trigger_pairwise_extrinsic_computation(self) -> None:
        """Trigger background computation of pairwise extrinsic calibration."""
        import threading
        from ..calibration.pairwise_extrinsics import calibrate_pairwise_extrinsics
        from ..config import CalibrationConfig
        
        self._state.extrinsics.computing = True
        
        def compute():
            try:
                # Wait for all intrinsics to complete (should already be done)
                while not all(s.intrinsic_complete for s in self._state.cameras.values()):
                    time.sleep(0.5)
                
                # Prepare intrinsics map
                intrinsics_map = {
                    s.intrinsic_result.camera_id: s.intrinsic_result 
                    for s in self._state.cameras.values() 
                    if s.intrinsic_result is not None
                }
                
                # Prepare pairwise captures (convert to expected format)
                pairwise_captures = {}
                for pair_key, captures in self._pairwise_captures.items():
                    if len(captures) >= self._pairwise_frames_required:
                        pairwise_captures[pair_key] = [cap['frames'] for cap in captures]
                
                # Create config
                config = CalibrationConfig(
                    charuco_squares_x=self.charuco_squares_x,
                    charuco_squares_y=self.charuco_squares_y,
                    charuco_square_length=self.charuco_square_length,
                    charuco_marker_length=self.charuco_marker_length,
                    charuco_dict=self.charuco_dict,
                )
                
                # Compute using pairwise approach
                result = calibrate_pairwise_extrinsics(
                    pairwise_captures, intrinsics_map, config
                )
                
                if result:
                    self._state.extrinsics.complete = True
                    self._state.extrinsics.result = result
                    print("Pairwise extrinsics calibration complete!")
                else:
                    print("Pairwise extrinsics calibration failed")
                
            except Exception as e:
                print(f"Error computing pairwise extrinsics: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._state.extrinsics.computing = False
                self._notify_progress()
        
        thread = threading.Thread(target=compute, daemon=True)
        thread.start()
    
    def _trigger_extrinsic_computation(self) -> None:
        """Trigger background computation of overall extrinsic calibration (legacy - all cameras)."""
        import threading
        from ..calibration.extrinsics import calibrate_extrinsics
        from ..config import CalibrationConfig
        
        self._state.extrinsics.computing = True
        
        def compute():
            try:
                # Wait for all intrinsics to complete
                while not all(s.intrinsic_complete for s in self._state.cameras.values()):
                    time.sleep(0.5)
                
                # Prepare intrinsics list
                intrinsics_list = [s.intrinsic_result for s in self._state.cameras.values()]
                
                # Prepare captures (convert to expected format)
                captures = []
                for cap_data in self._extrinsic_captures:
                    captures.append(cap_data['frames'])
                
                # Create config
                config = CalibrationConfig(
                    charuco_squares_x=self.charuco_squares_x,
                    charuco_squares_y=self.charuco_squares_y,
                    charuco_square_length=self.charuco_square_length,
                    charuco_marker_length=self.charuco_marker_length,
                    charuco_dict=self.charuco_dict,
                )
                
                # Compute
                result = calibrate_extrinsics(captures, intrinsics_list, config)
                
                if result:
                    self._state.extrinsics.complete = True
                    self._state.extrinsics.result = result
                    print(f"Extrinsics calibration complete!")
                else:
                    print(f"Extrinsics calibration failed")
                
            except Exception as e:
                print(f"Error computing extrinsics: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._state.extrinsics.computing = False
                self._notify_progress()
        
        thread = threading.Thread(target=compute, daemon=True)
        thread.start()
    
    
    
    def update_board_detection(
        self,
        camera_id: int,
        is_visible: bool,
        corners: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update board detection status for a camera.
        
        Args:
            camera_id: Camera ID
            is_visible: Whether board is visible
            corners: Detected ChArUco corners
            ids: Corner IDs
        """
        if camera_id not in self._state.cameras:
            return
        
        self._state.cameras[camera_id].board_visible = is_visible
        
        # Check if all cameras see the board
        self._state.all_cameras_visible = all(
            s.board_visible for s in self._state.cameras.values()
        )
        
        self._notify_progress()
    
    def capture_intrinsic_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        corners: np.ndarray,
        ids: np.ndarray,
    ) -> bool:
        """
        Capture a frame for intrinsic calibration.
        
        Args:
            camera_id: Camera ID
            frame: Image frame
            corners: Detected ChArUco corners
            ids: Corner IDs
            
        Returns:
            True if captured successfully
        """
        if self._state.current_step != CalibrationStep.INTRINSIC_CAPTURE:
            return False
        
        if camera_id not in self._intrinsic_frames:
            self._intrinsic_frames[camera_id] = []
        
        # Store frame and detection data
        self._intrinsic_frames[camera_id].append({
            'frame': frame.copy(),
            'corners': corners.copy(),
            'ids': ids.copy(),
        })
        
        self._state.cameras[camera_id].intrinsic_frames = len(self._intrinsic_frames[camera_id])
        
        # Update auto-capture state
        self._last_capture_time[camera_id] = time.time()
        self._last_capture_corners[camera_id] = corners.copy()
        
        self._notify_progress()
        
        return True
    
    def capture_extrinsic_frame(
        self,
        frames: Dict[int, np.ndarray],
        detections: Dict[int, Dict[str, np.ndarray]],
    ) -> bool:
        """
        Capture a synchronized frame set for extrinsic calibration.
        
        Args:
            frames: Dict mapping camera_id to frame
            detections: Dict mapping camera_id to detection data
            
        Returns:
            True if captured successfully
        """
        if self._state.current_step != CalibrationStep.EXTRINSIC_CAPTURE:
            return False
        
        # Relaxed check: app.py handles the "at least one" logic now
        # if not self._state.all_cameras_visible:
        #     return False
        
        self._extrinsic_captures.append({
            'frames': {k: v.copy() for k, v in frames.items()},
            'detections': detections,
            'timestamp': time.time(),
        })
        
        self._state.extrinsic_frames = len(self._extrinsic_captures)
        self._notify_progress()
        
        return True
    
    def get_intrinsic_frames(self, camera_id: int) -> List[Dict]:
        """Get captured intrinsic frames for a camera."""
        return self._intrinsic_frames.get(camera_id, [])
    
    def get_extrinsic_captures(self) -> List[Dict]:
        """Get captured extrinsic frame sets."""
        return self._extrinsic_captures
    
    def set_intrinsic_result(
        self,
        camera_id: int,
        reprojection_error: float,
    ) -> None:
        """Set intrinsic calibration result for a camera."""
        if camera_id in self._state.cameras:
            self._state.cameras[camera_id].intrinsic_complete = True
            self._state.cameras[camera_id].intrinsic_error = reprojection_error
            self._notify_progress()
    
    def set_extrinsic_result(
        self,
        camera_id: int,
        reprojection_error: float,
    ) -> None:
        """Set extrinsic calibration result for a camera."""
        if camera_id in self._state.cameras:
            self._state.cameras[camera_id].extrinsic_complete = True
            self._state.cameras[camera_id].extrinsic_error = reprojection_error
            self._notify_progress()
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of calibration quality."""
        intrinsic_errors = [
            s.intrinsic_error for s in self._state.cameras.values()
            if s.intrinsic_complete
        ]
        extrinsic_errors = [
            s.extrinsic_error for s in self._state.cameras.values()
            if s.extrinsic_complete
        ]
        
        return {
            'num_cameras': len(self._camera_ids),
            'intrinsic_complete': sum(1 for s in self._state.cameras.values() if s.intrinsic_complete),
            'extrinsic_complete': sum(1 for s in self._state.cameras.values() if s.extrinsic_complete),
            'avg_intrinsic_error': np.mean(intrinsic_errors) if intrinsic_errors else 0.0,
            'avg_extrinsic_error': np.mean(extrinsic_errors) if extrinsic_errors else 0.0,
            'max_intrinsic_error': max(intrinsic_errors) if intrinsic_errors else 0.0,
            'max_extrinsic_error': max(extrinsic_errors) if extrinsic_errors else 0.0,
        }
