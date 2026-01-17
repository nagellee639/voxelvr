
import cv2
import sys
from voxelvr.pose.detector_2d import PoseDetector2D

def test_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    print(f"Testing pose detection on {image_path}")
    detector = PoseDetector2D(backend="auto")
    if not detector.load_model():
        print("Failed to load model")
        return

    result = detector.detect(img)
    if result:
        print(f"Success! Detected {result.num_keypoints} keypoints")
        print(f"Confidences: {result.confidences}")
        
        # Save visualization
        vis = detector.draw_keypoints(img, result)
        cv2.imwrite("test_detection_vis.jpg", vis)
        print("Saved visualization to test_detection_vis.jpg")
    else:
        print("No pose detected.")

if __name__ == "__main__":
    test_detection(sys.argv[1])
