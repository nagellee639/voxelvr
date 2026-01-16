"""
Calibration Panel

Step-by-step calibration wizard with ChArUco board support.
"""

import numpy as np
import time
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CalibrationStep(Enum):
    """Calibration workflow steps."""
    IDLE = "idle"
    EXPORT_BOARD = "export_board"
    INTRINSIC_CAPTURE = "intrinsic_capture"
    INTRINSIC_COMPUTE = "intrinsic_compute"
    EXTRINSIC_CAPTURE = "extrinsic_capture"
    EXTRINSIC_COMPUTE = "extrinsic_compute"
    VERIFICATION = "verification"
    COMPLETE = "complete"


@dataclass
class CameraCalibrationStatus:
    """Calibration status for a single camera."""
    camera_id: int
    board_visible: bool = False
    intrinsic_frames: int = 0
    intrinsic_required: int = 20
    intrinsic_complete: bool = False
    intrinsic_error: float = 0.0
    extrinsic_complete: bool = False
    extrinsic_error: float = 0.0


@dataclass
class CalibrationState:
    """Overall calibration state."""
    current_step: CalibrationStep = CalibrationStep.IDLE
    cameras: Dict[int, CameraCalibrationStatus] = field(default_factory=dict)
    extrinsic_frames: int = 0
    extrinsic_required: int = 30
    all_cameras_visible: bool = False
    error_message: str = ""
    is_running: bool = False


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
        self._camera_ids: List[int] = []
        self._current_camera_idx = 0
        
        # Callbacks
        self._step_callbacks: List[Callable[[CalibrationStep], None]] = []
        self._progress_callbacks: List[Callable[[CalibrationState], None]] = []
        
        # Captured data storage
        self._intrinsic_frames: Dict[int, List[np.ndarray]] = {}
        self._extrinsic_captures: List[Dict[int, Any]] = []
        
        # Board detection results
        self._board = None
        self._aruco_dict = None
    
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
                intrinsic_required=self.intrinsic_frames_required,
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
                "5. Click 'Next' when ready"
            )
        
        elif step == CalibrationStep.INTRINSIC_CAPTURE:
            cam_id = self._camera_ids[self._current_camera_idx]
            status = self._state.cameras[cam_id]
            return (
                f"Step 2: Intrinsic Calibration - Camera {cam_id}\n\n"
                f"Captured: {status.intrinsic_frames}/{status.intrinsic_required} frames\n\n"
                "1. Hold the ChArUco board in front of this camera\n"
                "2. Move the board to different positions and angles\n"
                "3. Include corners and edges of the camera view\n"
                "4. Press SPACE or click 'Capture' when the board is visible\n"
                "5. Each position should show the board at a different angle"
            )
        
        elif step == CalibrationStep.INTRINSIC_COMPUTE:
            return "Computing intrinsic parameters... Please wait."
        
        elif step == CalibrationStep.EXTRINSIC_CAPTURE:
            visible_count = sum(1 for s in self._state.cameras.values() if s.board_visible)
            return (
                "Step 3: Extrinsic Calibration (Multi-Camera)\n\n"
                f"Captured: {self._state.extrinsic_frames}/{self._state.extrinsic_required} frames\n"
                f"Cameras seeing board: {visible_count}/{len(self._camera_ids)}\n\n"
                "1. Hold the ChArUco board in the center of your play area\n"
                "2. Make sure ALL cameras can see the board simultaneously\n"
                "3. Wave the board slowly around the center area\n"
                "4. Frames are captured automatically when all cameras see the board"
            )
        
        elif step == CalibrationStep.EXTRINSIC_COMPUTE:
            return "Computing extrinsic parameters... Please wait."
        
        elif step == CalibrationStep.VERIFICATION:
            return (
                "Step 4: Verification\n\n"
                "Calibration complete! Review the quality metrics below.\n\n"
                "A reprojection error under 0.5 pixels is excellent.\n"
                "Under 1.0 pixels is good.\n"
                "Over 2.0 pixels may cause tracking issues."
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
    
    def cancel_calibration(self) -> None:
        """Cancel the current calibration."""
        self._state.is_running = False
        self._state.current_step = CalibrationStep.IDLE
        self._intrinsic_frames.clear()
        self._extrinsic_captures.clear()
        self._notify_step_change()
    
    def next_step(self) -> None:
        """Advance to the next calibration step."""
        step = self._state.current_step
        
        if step == CalibrationStep.EXPORT_BOARD:
            self._state.current_step = CalibrationStep.INTRINSIC_CAPTURE
            self._current_camera_idx = 0
            self._intrinsic_frames = {cam_id: [] for cam_id in self._camera_ids}
        
        elif step == CalibrationStep.INTRINSIC_CAPTURE:
            # Check if we have enough frames for current camera
            cam_id = self._camera_ids[self._current_camera_idx]
            status = self._state.cameras[cam_id]
            
            if status.intrinsic_frames >= self.intrinsic_frames_required:
                # Compute intrinsics for this camera
                self._state.current_step = CalibrationStep.INTRINSIC_COMPUTE
        
        elif step == CalibrationStep.INTRINSIC_COMPUTE:
            # Move to next camera or extrinsic calibration
            self._current_camera_idx += 1
            if self._current_camera_idx < len(self._camera_ids):
                self._state.current_step = CalibrationStep.INTRINSIC_CAPTURE
            else:
                self._state.current_step = CalibrationStep.EXTRINSIC_CAPTURE
                self._state.extrinsic_frames = 0
        
        elif step == CalibrationStep.EXTRINSIC_CAPTURE:
            if self._state.extrinsic_frames >= self.extrinsic_frames_required:
                self._state.current_step = CalibrationStep.EXTRINSIC_COMPUTE
        
        elif step == CalibrationStep.EXTRINSIC_COMPUTE:
            self._state.current_step = CalibrationStep.VERIFICATION
        
        elif step == CalibrationStep.VERIFICATION:
            self._state.current_step = CalibrationStep.COMPLETE
            self._state.is_running = False
        
        self._notify_step_change()
    
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
        
        # Only capture if all cameras see the board
        if not self._state.all_cameras_visible:
            return False
        
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
