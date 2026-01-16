"""
2D Pose Detection

Uses a lightweight pose estimation model (ONNX format) for
extracting 2D keypoints from each camera view.

For initial testing, uses OpenCV's DNN module with a MoveNet/PoseNet model.
Can be upgraded to RTMPose ONNX for better accuracy.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import urllib.request
import os


# Model URLs for download
MOVENET_URL = "https://storage.googleapis.com/tfhub-modules/google/movenet/singlepose/lightning/4.tar.gz"
MOVENET_ONNX_URL = "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx"


@dataclass
class Keypoints2D:
    """2D keypoint detection result."""
    # Keypoint positions (17, 2) for COCO format
    positions: np.ndarray  # Shape: (num_keypoints, 2) - (x, y) in pixels
    
    # Confidence scores (17,)
    confidences: np.ndarray  # Shape: (num_keypoints,)
    
    # Image dimensions used for detection
    image_width: int
    image_height: int
    
    # Camera ID this detection came from
    camera_id: int = 0
    
    # Timestamp of the frame
    timestamp: float = 0.0
    
    # Detection confidence threshold used
    threshold: float = 0.3
    
    @property
    def num_keypoints(self) -> int:
        return len(self.positions)
    
    def get_valid_keypoints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get keypoints above confidence threshold."""
        mask = self.confidences >= self.threshold
        indices = np.where(mask)[0]
        return indices, self.positions[mask], self.confidences[mask]
    
    def to_normalized(self) -> np.ndarray:
        """Convert to normalized coordinates [0, 1]."""
        normalized = self.positions.copy()
        normalized[:, 0] /= self.image_width
        normalized[:, 1] /= self.image_height
        return normalized


# COCO keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso to hips
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]


class PoseDetector2D:
    """
    2D pose detector using ONNX Runtime.
    
    Supports multiple backends:
    - DirectML (Windows AMD/Intel/NVIDIA)
    - CUDA (NVIDIA Linux/Windows)
    - CPU (fallback)
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        backend: str = "auto",
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize the pose detector.
        
        Args:
            model_path: Path to ONNX model file
            backend: Execution backend ("auto", "directml", "cuda", "cpu")
            confidence_threshold: Minimum keypoint confidence
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.backend = backend
        
        # Will be set after model loading
        self.input_height = 192
        self.input_width = 192
        
    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load the ONNX model.
        
        Args:
            model_path: Path to ONNX model (downloads default if None)
            
        Returns:
            True if model loaded successfully
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("Error: onnxruntime not installed. Run: pip install onnxruntime-directml")
            return False
        
        if model_path is None:
            model_path = self._get_default_model()
        
        if model_path is None or not model_path.exists():
            print(f"Model not found at {model_path}")
            return False
        
        # Select execution providers based on backend
        providers = self._get_providers()
        print(f"Using ONNX Runtime providers: {providers}")
        
        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            
            # Get input details
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape
            self.input_type = input_info.type  # e.g. 'tensor(int32)' or 'tensor(float)'
            
            # Handle dynamic dimensions
            if len(self.input_shape) >= 4:
                h = self.input_shape[1] if isinstance(self.input_shape[1], int) else 192
                w = self.input_shape[2] if isinstance(self.input_shape[2], int) else 192
                self.input_height = h
                self.input_width = w
            
            print(f"Model loaded: input shape={self.input_shape}, type={self.input_type}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def _get_providers(self) -> List[str]:
        """Get ONNX Runtime execution providers."""
        import onnxruntime as ort
        
        available = ort.get_available_providers()
        
        if self.backend == "auto":
            # Prefer DirectML on Windows, CUDA on Linux, fallback to CPU
            if "DmlExecutionProvider" in available:
                return ["DmlExecutionProvider", "CPUExecutionProvider"]
            elif "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "ROCMExecutionProvider" in available:
                return ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                return ["CPUExecutionProvider"]
        elif self.backend == "directml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif self.backend == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.backend == "rocm":
            return ["ROCMExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    
    def _get_default_model(self) -> Optional[Path]:
        """Download and return path to default model."""
        cache_dir = Path.home() / ".voxelvr" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = cache_dir / "movenet_lightning.onnx"
        
        if not model_path.exists():
            print("Downloading MoveNet model...")
            try:
                urllib.request.urlretrieve(MOVENET_ONNX_URL, model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Failed to download model: {e}")
                return None
        
        return model_path
    
    def detect(self, image: np.ndarray, camera_id: int = 0, timestamp: float = 0.0) -> Optional[Keypoints2D]:
        """
        Detect 2D pose in an image.
        
        Args:
            image: BGR image (H, W, 3)
            camera_id: Camera identifier
            timestamp: Frame timestamp
            
        Returns:
            Keypoints2D object or None if detection failed
        """
        if self.session is None:
            if not self.load_model(self.model_path):
                return None
        
        orig_height, orig_width = image.shape[:2]
        
        # Preprocess: resize and normalize
        input_tensor = self._preprocess(image)
        
        # Run inference
        try:
            outputs = self.session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            print(f"Inference failed: {e}")
            return None
        
        # Parse output
        keypoints = self._postprocess(outputs, orig_width, orig_height)
        
        if keypoints is None:
            return None
        
        return Keypoints2D(
            positions=keypoints[:, :2],
            confidences=keypoints[:, 2],
            image_width=orig_width,
            image_height=orig_height,
            camera_id=camera_id,
            timestamp=timestamp,
            threshold=self.confidence_threshold,
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Resize to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Check expected input type
        if hasattr(self, 'input_type') and 'int32' in self.input_type:
             # MoveNet specific: [1, 192, 192, 3] int32 tensor with values [0, 255]
             tensor = np.expand_dims(rgb, axis=0).astype(np.int32)
             return tensor
        
        # Default: float32 normalized [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and ensure NHWC format (for MoveNet)
        # Some models expect NCHW - adjust if needed
        tensor = np.expand_dims(normalized, axis=0)
        
        return tensor
    
    def _postprocess(self, outputs: List[np.ndarray], orig_width: int, orig_height: int) -> Optional[np.ndarray]:
        """
        Parse model output to keypoints.
        
        Args:
            outputs: Model outputs
            orig_width: Original image width
            orig_height: Original image height
            
        Returns:
            Array of shape (17, 3) with (x, y, confidence) for each keypoint
        """
        # MoveNet output format: (1, 1, 17, 3) with [y, x, score]
        output = outputs[0]
        
        # Handle different output shapes
        if output.ndim == 4:
            # (batch, 1, 17, 3)
            keypoints_raw = output[0, 0]  # (17, 3)
        elif output.ndim == 3:
            # (batch, 17, 3)
            keypoints_raw = output[0]  # (17, 3)
        else:
            print(f"Unexpected output shape: {output.shape}")
            return None
        
        # MoveNet outputs [y, x, score] in normalized [0, 1] coordinates
        keypoints = np.zeros((17, 3), dtype=np.float32)
        
        for i in range(min(17, len(keypoints_raw))):
            y_norm, x_norm, score = keypoints_raw[i]
            keypoints[i, 0] = x_norm * orig_width  # x in pixels
            keypoints[i, 1] = y_norm * orig_height  # y in pixels
            keypoints[i, 2] = score  # confidence
        
        return keypoints
    
    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: Keypoints2D,
        draw_skeleton: bool = True,
    ) -> np.ndarray:
        """
        Draw keypoints on image.
        
        Args:
            image: Input image (will be copied)
            keypoints: Detected keypoints
            draw_skeleton: Whether to draw skeleton connections
            
        Returns:
            Image with keypoints drawn
        """
        output = image.copy()
        
        positions = keypoints.positions
        confidences = keypoints.confidences
        
        # Draw skeleton first (so joints appear on top)
        if draw_skeleton:
            for start_idx, end_idx in COCO_SKELETON:
                if (confidences[start_idx] >= keypoints.threshold and 
                    confidences[end_idx] >= keypoints.threshold):
                    pt1 = tuple(positions[start_idx].astype(int))
                    pt2 = tuple(positions[end_idx].astype(int))
                    cv2.line(output, pt1, pt2, (0, 255, 0), 2)
        
        # Draw joints
        for i, (pos, conf) in enumerate(zip(positions, confidences)):
            if conf >= keypoints.threshold:
                x, y = int(pos[0]), int(pos[1])
                # Color based on confidence (red=low, green=high)
                green = int(min(255, conf * 255))
                color = (0, green, 255 - green)
                cv2.circle(output, (x, y), 5, color, -1)
                cv2.circle(output, (x, y), 5, (0, 0, 0), 1)
        
        return output


class PoseDetectorCPU:
    """
    Fallback CPU-only pose detector using OpenCV DNN.
    
    Uses a simpler model that works without ONNX Runtime.
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.net = None
        
    def detect(self, image: np.ndarray, camera_id: int = 0, timestamp: float = 0.0) -> Optional[Keypoints2D]:
        """Basic detection using OpenCV's pose estimation."""
        # This is a placeholder - OpenCV's built-in pose detection requires
        # additional model files. For now, we'll use a simple skeleton detection
        # based on color/motion or defer to the ONNX detector.
        print("CPU-only detector not yet implemented - please install onnxruntime")
        return None
