"""
AprilTag Detection and Tracking

Provides AprilTag detection for wearable markers to improve tracking precision.
Supports pose estimation and PDF generation for printing AprilTag sheets.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class AprilTagDetection:
    """A detected AprilTag in an image."""
    tag_id: int
    corners: np.ndarray  # (4, 2) corner points
    center: np.ndarray   # (2,) center point
    
    @property
    def size_pixels(self) -> float:
        """Approximate size in pixels (average edge length)."""
        edges = [
            np.linalg.norm(self.corners[0] - self.corners[1]),
            np.linalg.norm(self.corners[1] - self.corners[2]),
            np.linalg.norm(self.corners[2] - self.corners[3]),
            np.linalg.norm(self.corners[3] - self.corners[0]),
        ]
        return np.mean(edges)


# Mapping of common AprilTag families
APRILTAG_FAMILIES = {
    'tag36h11': cv2.aruco.DICT_APRILTAG_36h11,
    'tag25h9': cv2.aruco.DICT_APRILTAG_25h9,
    'tag16h5': cv2.aruco.DICT_APRILTAG_16h5,
}


# Wearable marker body locations
WEARABLE_LOCATIONS = [
    'chest',
    'left_wrist',
    'right_wrist',
    'left_ankle',
    'right_ankle',
    'waist_left',
    'waist_right',
]

# Default tag IDs for each body location
DEFAULT_TAG_IDS = {
    'chest': 0,
    'left_wrist': 1,
    'right_wrist': 2,
    'left_ankle': 3,
    'right_ankle': 4,
    'waist_left': 5,
    'waist_right': 6,
}


def get_apriltag_dictionary(family: str = 'tag36h11') -> cv2.aruco.Dictionary:
    """Get OpenCV ArUco dictionary for AprilTag family."""
    if family not in APRILTAG_FAMILIES:
        raise ValueError(f"Unknown AprilTag family: {family}. "
                        f"Available: {list(APRILTAG_FAMILIES.keys())}")
    return cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILIES[family])


def detect_apriltags(
    frame: np.ndarray,
    tag_family: str = 'tag36h11',
    refine_corners: bool = True,
) -> List[AprilTagDetection]:
    """
    Detect AprilTags in an image.
    
    Args:
        frame: Input image (grayscale or BGR)
        tag_family: AprilTag family name
        refine_corners: Whether to refine corner positions
        
    Returns:
        List of detected AprilTags
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Get dictionary and detector
    dictionary = get_apriltag_dictionary(tag_family)
    parameters = cv2.aruco.DetectorParameters()
    
    # Enable corner refinement for better accuracy
    if refine_corners:
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None:
        return []
    
    detections = []
    for i, tag_id in enumerate(ids.flatten()):
        corner_pts = corners[i][0]  # Shape: (4, 2)
        center = np.mean(corner_pts, axis=0)
        
        detections.append(AprilTagDetection(
            tag_id=int(tag_id),
            corners=corner_pts,
            center=center,
        ))
    
    return detections


def estimate_apriltag_pose(
    detection: AprilTagDetection,
    tag_size: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate 3D pose of a detected AprilTag.
    
    Args:
        detection: AprilTag detection
        tag_size: Physical size of tag in meters
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        Tuple of (rvec, tvec) - rotation and translation vectors
    """
    # Object points: corners of the tag in tag-local coordinates
    half_size = tag_size / 2
    obj_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0],
    ], dtype=np.float32)
    
    # Image points from detection
    img_points = detection.corners.astype(np.float32)
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        obj_points, 
        img_points, 
        camera_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    if not success:
        raise ValueError("Failed to estimate AprilTag pose")
    
    return rvec, tvec


def draw_apriltag_detections(
    frame: np.ndarray,
    detections: List[AprilTagDetection],
    draw_ids: bool = True,
    draw_axes: bool = False,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    tag_size: float = 0.05,
) -> np.ndarray:
    """
    Draw detected AprilTags on an image.
    
    Args:
        frame: Input image (will be copied)
        detections: List of detections
        draw_ids: Whether to draw tag IDs
        draw_axes: Whether to draw 3D axes (requires camera params)
        camera_matrix: Camera intrinsic matrix (for axes)
        dist_coeffs: Distortion coefficients (for axes)
        tag_size: Tag size in meters (for axes)
        
    Returns:
        Annotated image
    """
    result = frame.copy()
    
    for det in detections:
        # Draw corners
        corners = det.corners.astype(np.int32)
        cv2.polylines(result, [corners], True, (0, 255, 0), 2)
        
        # Draw center
        center = tuple(det.center.astype(np.int32))
        cv2.circle(result, center, 5, (0, 0, 255), -1)
        
        # Draw ID
        if draw_ids:
            cv2.putText(
                result, 
                str(det.tag_id),
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
        
        # Draw 3D axes
        if draw_axes and camera_matrix is not None and dist_coeffs is not None:
            try:
                rvec, tvec = estimate_apriltag_pose(det, tag_size, camera_matrix, dist_coeffs)
                cv2.drawFrameAxes(result, camera_matrix, dist_coeffs, rvec, tvec, tag_size * 0.5)
            except ValueError:
                pass
    
    return result


def generate_apriltag_image(
    tag_id: int,
    size_pixels: int = 200,
    tag_family: str = 'tag36h11',
    border_bits: int = 1,
) -> np.ndarray:
    """
    Generate an AprilTag image.
    
    Args:
        tag_id: Tag ID to generate
        size_pixels: Output image size in pixels
        tag_family: AprilTag family name
        border_bits: Border size in bits
        
    Returns:
        Grayscale tag image
    """
    dictionary = get_apriltag_dictionary(tag_family)
    tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, size_pixels)
    return tag_img


def export_apriltag_sheet_pdf(
    output_path: Path,
    tag_ids: Optional[List[int]] = None,
    tag_size_mm: float = 50.0,
    tag_family: str = 'tag36h11',
    page_size_mm: Tuple[float, float] = (210, 297),  # A4
    margin_mm: float = 15.0,
    include_labels: bool = True,
) -> bool:
    """
    Export printable AprilTag sheet as PDF.
    
    Args:
        output_path: Path to save PDF
        tag_ids: List of tag IDs to include (default: wearable markers 0-6)
        tag_size_mm: Size of each tag in millimeters
        tag_family: AprilTag family name
        page_size_mm: Page size in mm (width, height)
        margin_mm: Page margin in mm
        include_labels: Whether to include text labels
        
    Returns:
        True if successful
    """
    if tag_ids is None:
        tag_ids = list(range(7))  # Default wearable markers
    
    # Calculate layout
    page_width_mm, page_height_mm = page_size_mm
    usable_width = page_width_mm - 2 * margin_mm
    usable_height = page_height_mm - 2 * margin_mm
    
    # Space between tags
    spacing_mm = 10
    label_height_mm = 8 if include_labels else 0
    cell_height = tag_size_mm + label_height_mm + spacing_mm
    cell_width = tag_size_mm + spacing_mm
    
    cols = int(usable_width / cell_width)
    rows = int(usable_height / cell_height)
    
    if cols == 0 or rows == 0:
        raise ValueError("Tags too large for page size")
    
    # DPI for PDF (300 is standard print quality)
    dpi = 300
    mm_to_px = dpi / 25.4
    
    page_width_px = int(page_width_mm * mm_to_px)
    page_height_px = int(page_height_mm * mm_to_px)
    margin_px = int(margin_mm * mm_to_px)
    tag_size_px = int(tag_size_mm * mm_to_px)
    cell_width_px = int(cell_width * mm_to_px)
    cell_height_px = int(cell_height * mm_to_px)
    
    # Create pages
    pages = []
    tags_per_page = cols * rows
    
    for page_idx in range(0, len(tag_ids), tags_per_page):
        page_tags = tag_ids[page_idx:page_idx + tags_per_page]
        
        # White background
        page = np.ones((page_height_px, page_width_px), dtype=np.uint8) * 255
        
        for i, tag_id in enumerate(page_tags):
            row = i // cols
            col = i % cols
            
            x = margin_px + col * cell_width_px
            y = margin_px + row * cell_height_px
            
            # Generate tag
            tag_img = generate_apriltag_image(tag_id, tag_size_px, tag_family)
            
            # Place tag
            page[y:y+tag_size_px, x:x+tag_size_px] = tag_img
            
            # Add label
            if include_labels:
                # Get body location name if available
                location = None
                for loc, tid in DEFAULT_TAG_IDS.items():
                    if tid == tag_id:
                        location = loc
                        break
                
                label = f"ID:{tag_id}"
                if location:
                    label = f"{location.replace('_', ' ').title()} ({tag_id})"
                
                # Draw label (convert to BGR for text, then back)
                page_bgr = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
                font_scale = 0.4 * (dpi / 100)
                label_y = y + tag_size_px + int(5 * mm_to_px)
                cv2.putText(
                    page_bgr,
                    label,
                    (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    max(1, int(dpi / 150))
                )
                page = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        
        pages.append(page)
    
    # Save as PDF (using imageio or PIL if available, fallback to PNG)
    try:
        from PIL import Image
        
        pil_pages = [Image.fromarray(p) for p in pages]
        pil_pages[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_pages[1:] if len(pil_pages) > 1 else [],
            resolution=dpi,
        )
        return True
    except ImportError:
        # Fallback: save as PNG with same name
        png_path = output_path.with_suffix('.png')
        if len(pages) == 1:
            cv2.imwrite(str(png_path), pages[0])
        else:
            # Save first page only
            cv2.imwrite(str(png_path), pages[0])
            print(f"Warning: PIL not available, saved first page only to {png_path}")
        return True


@dataclass
class WearableMarkerConfig:
    """Configuration for a wearable AprilTag marker."""
    location: str  # Body location name
    tag_id: int
    tag_size: float = 0.05  # 5cm default
    
    # Expected position relative to body part (for validation)
    expected_offset: Optional[np.ndarray] = None


class WearableMarkerTracker:
    """
    Track wearable AprilTag markers on the body.
    
    Integrates with skeleton tracking to improve precision on key joints.
    """
    
    def __init__(
        self,
        markers: Optional[List[WearableMarkerConfig]] = None,
        tag_family: str = 'tag36h11',
    ):
        """
        Initialize wearable marker tracker.
        
        Args:
            markers: List of marker configurations (default: standard wearable set)
            tag_family: AprilTag family name
        """
        if markers is None:
            markers = [
                WearableMarkerConfig('chest', 0),
                WearableMarkerConfig('left_wrist', 1),
                WearableMarkerConfig('right_wrist', 2),
                WearableMarkerConfig('left_ankle', 3),
                WearableMarkerConfig('right_ankle', 4),
                WearableMarkerConfig('waist_left', 5),
                WearableMarkerConfig('waist_right', 6),
            ]
        
        self.markers = {m.tag_id: m for m in markers}
        self.tag_family = tag_family
        
        # Map tag IDs to COCO keypoint indices for fusion
        self._tag_to_keypoint = {
            # Wrists
            1: 9,   # left_wrist -> LEFT_WRIST
            2: 10,  # right_wrist -> RIGHT_WRIST
            # Ankles
            3: 15,  # left_ankle -> LEFT_ANKLE
            4: 16,  # right_ankle -> RIGHT_ANKLE
        }
    
    def detect_markers(
        self,
        frame: np.ndarray,
    ) -> Dict[int, AprilTagDetection]:
        """
        Detect wearable markers in a frame.
        
        Returns:
            Dictionary mapping tag_id to detection (only known markers)
        """
        all_detections = detect_apriltags(frame, self.tag_family)
        
        return {
            det.tag_id: det 
            for det in all_detections 
            if det.tag_id in self.markers
        }
    
    def get_marker_poses(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Get 3D poses of all detected wearable markers.
        
        Returns:
            Dictionary mapping tag_id to (rvec, tvec)
        """
        detections = self.detect_markers(frame)
        poses = {}
        
        for tag_id, det in detections.items():
            marker = self.markers[tag_id]
            try:
                rvec, tvec = estimate_apriltag_pose(
                    det, 
                    marker.tag_size, 
                    camera_matrix, 
                    dist_coeffs
                )
                poses[tag_id] = (rvec, tvec)
            except ValueError:
                continue
        
        return poses
    
    def fuse_with_skeleton(
        self,
        skeleton_positions: np.ndarray,
        skeleton_valid: np.ndarray,
        marker_positions: Dict[int, np.ndarray],
        marker_weight: float = 0.7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse AprilTag marker positions with skeleton tracking.
        
        Markers are weighted higher than skeleton for joints they track.
        
        Args:
            skeleton_positions: (17, 3) skeleton joint positions
            skeleton_valid: (17,) boolean mask
            marker_positions: tag_id -> (3,) 3D position
            marker_weight: Weight for marker positions (0-1)
            
        Returns:
            Tuple of (fused_positions, fused_valid)
        """
        fused = skeleton_positions.copy()
        valid = skeleton_valid.copy()
        
        for tag_id, pos in marker_positions.items():
            if tag_id not in self._tag_to_keypoint:
                continue
            
            keypoint_idx = self._tag_to_keypoint[tag_id]
            
            if skeleton_valid[keypoint_idx]:
                # Weighted average
                skel_weight = 1.0 - marker_weight
                fused[keypoint_idx] = (
                    marker_weight * pos + 
                    skel_weight * skeleton_positions[keypoint_idx]
                )
            else:
                # Use marker position directly
                fused[keypoint_idx] = pos
            
            valid[keypoint_idx] = True
        
        return fused, valid
