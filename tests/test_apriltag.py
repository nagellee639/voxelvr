"""
Tests for AprilTag Detection Module

Tests AprilTag detection, pose estimation, and PDF export.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from voxelvr.calibration.apriltag import (
    AprilTagDetection,
    detect_apriltags,
    estimate_apriltag_pose,
    draw_apriltag_detections,
    generate_apriltag_image,
    export_apriltag_sheet_pdf,
    get_apriltag_dictionary,
    WearableMarkerConfig,
    WearableMarkerTracker,
    DEFAULT_TAG_IDS,
    WEARABLE_LOCATIONS,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def synthetic_apriltag_image() -> np.ndarray:
    """Create a synthetic image containing AprilTag markers."""
    # Create white background
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Generate a tag and place it in the image
    tag_img = generate_apriltag_image(0, size_pixels=100)
    
    # Place tag in center
    x_offset = (640 - 100) // 2
    y_offset = (480 - 100) // 2
    
    img[y_offset:y_offset+100, x_offset:x_offset+100] = \
        cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR)
    
    return img


@pytest.fixture
def camera_matrix() -> np.ndarray:
    """Synthetic camera matrix."""
    return np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1],
    ], dtype=np.float64)


@pytest.fixture
def dist_coeffs() -> np.ndarray:
    """Zero distortion coefficients."""
    return np.zeros(5, dtype=np.float64)


# ============================================================================
# Detection Tests
# ============================================================================

class TestAprilTagDetection:
    """Tests for AprilTag detection."""
    
    def test_detect_known_tag(self, synthetic_apriltag_image):
        """Test detection of a known AprilTag."""
        detections = detect_apriltags(synthetic_apriltag_image)
        
        # Should detect exactly one tag
        assert len(detections) == 1
        
        det = detections[0]
        assert det.tag_id == 0
        assert det.corners.shape == (4, 2)
        assert det.center.shape == (2,)
        
        # Center should be roughly in image center
        assert abs(det.center[0] - 320) < 20
        assert abs(det.center[1] - 240) < 20
    
    def test_detect_multiple_tags(self):
        """Test detection of multiple AprilTags."""
        # Create image with two tags
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        tag0 = generate_apriltag_image(0, size_pixels=80)
        tag1 = generate_apriltag_image(1, size_pixels=80)
        
        # Place tags side by side
        img[200:280, 150:230] = cv2.cvtColor(tag0, cv2.COLOR_GRAY2BGR)
        img[200:280, 410:490] = cv2.cvtColor(tag1, cv2.COLOR_GRAY2BGR)
        
        detections = detect_apriltags(img)
        
        assert len(detections) == 2
        tag_ids = {d.tag_id for d in detections}
        assert tag_ids == {0, 1}
    
    def test_detect_no_tags(self):
        """Test detection returns empty list when no tags present."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        detections = detect_apriltags(img)
        assert len(detections) == 0
    
    def test_detect_grayscale_input(self, synthetic_apriltag_image):
        """Test detection works with grayscale input."""
        gray = cv2.cvtColor(synthetic_apriltag_image, cv2.COLOR_BGR2GRAY)
        detections = detect_apriltags(gray)
        assert len(detections) == 1
    
    def test_detection_size_property(self, synthetic_apriltag_image):
        """Test AprilTagDetection size_pixels property."""
        detections = detect_apriltags(synthetic_apriltag_image)
        assert len(detections) == 1
        
        size = detections[0].size_pixels
        # Should be close to 100 pixels (the generated tag size)
        assert 90 < size < 110


class TestAprilTagPoseEstimation:
    """Tests for pose estimation."""
    
    def test_estimate_pose_basic(
        self, synthetic_apriltag_image, camera_matrix, dist_coeffs
    ):
        """Test basic pose estimation."""
        detections = detect_apriltags(synthetic_apriltag_image)
        assert len(detections) == 1
        
        rvec, tvec = estimate_apriltag_pose(
            detections[0],
            tag_size=0.05,  # 5cm tag
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        
        assert rvec.shape == (3, 1)
        assert tvec.shape == (3, 1)
        
        # Tag should be in front of camera (positive Z)
        assert tvec[2] > 0
    
    def test_estimate_pose_different_sizes(self, camera_matrix, dist_coeffs):
        """Test pose estimation with different tag sizes."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        tag = generate_apriltag_image(0, size_pixels=100)
        img[190:290, 270:370] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
        
        detections = detect_apriltags(img)
        
        # Smaller physical size should give smaller distance
        _, tvec_small = estimate_apriltag_pose(
            detections[0], tag_size=0.03, 
            camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
        )
        
        _, tvec_large = estimate_apriltag_pose(
            detections[0], tag_size=0.10,
            camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
        )
        
        # Larger tag should appear further away
        assert tvec_large[2] > tvec_small[2]


class TestDrawDetections:
    """Tests for visualization functions."""
    
    def test_draw_detections(self, synthetic_apriltag_image):
        """Test drawing detections doesn't crash."""
        detections = detect_apriltags(synthetic_apriltag_image)
        
        result = draw_apriltag_detections(
            synthetic_apriltag_image,
            detections,
            draw_ids=True,
        )
        
        assert result.shape == synthetic_apriltag_image.shape
        # Should be modified (not same as input)
        assert not np.array_equal(result, synthetic_apriltag_image)
    
    def test_draw_with_axes(
        self, synthetic_apriltag_image, camera_matrix, dist_coeffs
    ):
        """Test drawing with 3D axes."""
        detections = detect_apriltags(synthetic_apriltag_image)
        
        result = draw_apriltag_detections(
            synthetic_apriltag_image,
            detections,
            draw_ids=True,
            draw_axes=True,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            tag_size=0.05,
        )
        
        assert result.shape == synthetic_apriltag_image.shape


# ============================================================================
# Image Generation Tests
# ============================================================================

class TestTagGeneration:
    """Tests for AprilTag image generation."""
    
    def test_generate_tag_basic(self):
        """Test basic tag generation."""
        tag = generate_apriltag_image(0, size_pixels=100)
        
        assert tag.shape == (100, 100)
        assert tag.dtype == np.uint8
        
        # Should be binary (black and white only)
        unique = np.unique(tag)
        assert len(unique) == 2
        assert 0 in unique
        assert 255 in unique
    
    def test_generate_different_ids(self):
        """Test generating different tag IDs produces different images."""
        tag0 = generate_apriltag_image(0, size_pixels=100)
        tag1 = generate_apriltag_image(1, size_pixels=100)
        
        assert not np.array_equal(tag0, tag1)
    
    def test_generate_different_sizes(self):
        """Test generating different sizes."""
        tag_small = generate_apriltag_image(0, size_pixels=50)
        tag_large = generate_apriltag_image(0, size_pixels=200)
        
        assert tag_small.shape == (50, 50)
        assert tag_large.shape == (200, 200)


# ============================================================================
# PDF Export Tests
# ============================================================================

class TestPDFExport:
    """Tests for PDF export functionality."""
    
    def test_export_apriltag_sheet_pdf(self):
        """Test PDF export creates a file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "apriltags.pdf"
            
            result = export_apriltag_sheet_pdf(
                output_path,
                tag_ids=[0, 1, 2],
                tag_size_mm=50.0,
            )
            
            assert result is True
            # Either PDF or PNG should exist (depending on PIL availability)
            assert output_path.exists() or output_path.with_suffix('.png').exists()
    
    def test_export_default_wearable_markers(self):
        """Test export with default wearable marker IDs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "wearable_markers.pdf"
            
            result = export_apriltag_sheet_pdf(
                output_path,
                tag_ids=None,  # Use defaults
                include_labels=True,
            )
            
            assert result is True
    
    def test_export_custom_page_size(self):
        """Test export with custom page size."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "letter.pdf"
            
            result = export_apriltag_sheet_pdf(
                output_path,
                tag_ids=[0, 1, 2, 3],
                page_size_mm=(215.9, 279.4),  # US Letter
            )
            
            assert result is True


# ============================================================================
# Wearable Marker Tracker Tests
# ============================================================================

class TestWearableMarkerTracker:
    """Tests for wearable marker tracking."""
    
    def test_tracker_initialization(self):
        """Test tracker initializes with default markers."""
        tracker = WearableMarkerTracker()
        
        assert len(tracker.markers) == 7
        assert 0 in tracker.markers  # chest
        assert 1 in tracker.markers  # left_wrist
    
    def test_tracker_custom_markers(self):
        """Test tracker with custom marker configuration."""
        markers = [
            WearableMarkerConfig('hip', 10),
            WearableMarkerConfig('head', 11),
        ]
        tracker = WearableMarkerTracker(markers=markers)
        
        assert len(tracker.markers) == 2
        assert 10 in tracker.markers
        assert 11 in tracker.markers
    
    def test_detect_markers(self, synthetic_apriltag_image):
        """Test marker detection filters to known markers."""
        tracker = WearableMarkerTracker()
        
        # synthetic_apriltag_image has tag ID 0 (chest)
        markers = tracker.detect_markers(synthetic_apriltag_image)
        
        assert len(markers) == 1
        assert 0 in markers
    
    def test_fuse_with_skeleton(self):
        """Test fusion of marker and skeleton positions."""
        tracker = WearableMarkerTracker()
        
        # Skeleton positions
        skeleton = np.random.randn(17, 3)
        valid = np.ones(17, dtype=bool)
        
        # Marker position (left wrist = tag 1 -> keypoint 9)
        marker_positions = {
            1: np.array([1.0, 2.0, 3.0]),  # left_wrist
        }
        
        fused, fused_valid = tracker.fuse_with_skeleton(
            skeleton, valid, marker_positions, marker_weight=0.7
        )
        
        # Wrist position should be weighted toward marker
        expected = 0.7 * marker_positions[1] + 0.3 * skeleton[9]
        np.testing.assert_array_almost_equal(fused[9], expected)
    
    def test_fuse_fills_invalid_joints(self):
        """Test that markers can fill in invalid skeleton joints."""
        tracker = WearableMarkerTracker()
        
        skeleton = np.random.randn(17, 3)
        valid = np.ones(17, dtype=bool)
        valid[9] = False  # Left wrist is invalid
        
        marker_positions = {
            1: np.array([1.0, 2.0, 3.0]),
        }
        
        fused, fused_valid = tracker.fuse_with_skeleton(
            skeleton, valid, marker_positions
        )
        
        # Wrist should now be valid and use marker position directly
        assert fused_valid[9] == True
        np.testing.assert_array_equal(fused[9], marker_positions[1])


# ============================================================================
# Dictionary Tests
# ============================================================================

class TestDictionary:
    """Tests for AprilTag dictionary functions."""
    
    def test_get_valid_dictionary(self):
        """Test getting valid dictionary."""
        d = get_apriltag_dictionary('tag36h11')
        assert d is not None
    
    def test_get_invalid_dictionary(self):
        """Test invalid dictionary raises error."""
        with pytest.raises(ValueError):
            get_apriltag_dictionary('invalid_family')
    
    def test_default_tag_ids(self):
        """Test default tag IDs mapping."""
        assert DEFAULT_TAG_IDS['chest'] == 0
        assert DEFAULT_TAG_IDS['left_wrist'] == 1
        assert DEFAULT_TAG_IDS['right_wrist'] == 2
        assert len(DEFAULT_TAG_IDS) == len(WEARABLE_LOCATIONS)
