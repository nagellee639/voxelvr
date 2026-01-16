
import pytest
import cv2
import numpy as np
from pathlib import Path
from voxelvr.pose.detector_2d import PoseDetector2D
from voxelvr.calibration.charuco import detect_charuco, create_charuco_board

class TestRealData:
    """Tests using real or realistic image files from disk."""

    def test_pose_estimation_on_real_image(self):
        """Test pose detection on a real-world image (Messi)."""
        image_path = Path("test_data/external/person.jpg")
        if not image_path.exists():
            pytest.skip("Real person image not found (download failed?)")
            
        img = cv2.imread(str(image_path))
        assert img is not None, "Failed to load person image"
        
        # Initialize detector
        detector = PoseDetector2D(confidence_threshold=0.3)
        success = detector.load_model()
        
        # If model loading fails (e.g. no GPU or model file missing), we might need to skip
        # checking if we can mock or if the user environment has it.
        # Assuming environment has ONNX runtime as previous tests passed.
        if not success:
            pytest.fail("Failed to load PoseDetector2D model")
            
        # Detect
        result = detector.detect(img)
        assert result is not None, "Detection returned None"
        
        # Verification
        # Messi image should have at least one person
        assert len(result.positions) > 0, "No keypoints detected in Messi image"
        
        # Basic check for reasonable confidence
        confidences = result.confidences
        avg_conf = np.mean(confidences[confidences > 0])
        print(f"Average confidence on Messi: {avg_conf:.2f}")
        assert avg_conf > 0.3, "Confidence too low for real image"

    def test_calibration_on_generated_image(self):
        """Test calibration detection on perspective-warped synthetic image."""
        # Use one of the generated images
        image_path = Path("test_data/calibration/calib_00.jpg")
        if not image_path.exists():
            # If generated images missing, generate them now?
            pytest.skip("Generated calibration images not found")
            
        img = cv2.imread(str(image_path))
        assert img is not None, "Failed to load calibration image"
        
        # Create board config (must match what generate_test_images.py used)
        # 5x5, 0.04 partial square, etc.
        # Wait, generate_test_images uses: squares_x=5, squares_y=5, ... dictionary="DICT_6X6_250"
        board, aruco_dict = create_charuco_board(
            squares_x=5,
            squares_y=5,
            square_length=0.04,
            marker_length=0.03,
            dictionary="DICT_6X6_250"
        )
        
        # Detect
        result = detect_charuco(img, board, aruco_dict)
        
        assert result['success'] == True, "Failed to detect board in warped image"
        assert len(result['corners']) > 4, "Not enough corners detected"
        
        # Check that we found some markers too
        assert result.get('marker_ids') is not None
        assert len(result['marker_ids']) > 0
