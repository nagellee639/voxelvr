import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, "/home/lee/voxelvr")

from voxelvr.pose import PoseDetector2D

def main():
    # Find a test image
    image_path = Path("/home/lee/voxelvr/dataset/tracking/20260116_165920/cam_0/000000.jpg")
    if not image_path.exists():
        # Try to find any jpg in the cam_0 folder
        folder = Path("/home/lee/voxelvr/dataset/tracking/20260116_165920/cam_0")
        images = list(folder.glob("*.jpg"))
        if not images:
            print(f"No images found in {folder}")
            return
        image_path = images[0]
    
    print(f"Testing on image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print("Failed to load image")
        return

    print("Loading detector...")
    detector = PoseDetector2D(confidence_threshold=0.3)
    detector.load_model()
    
    print("Running detection...")
    kp = detector.detect(image, camera_id=0)
    
    if kp:
        print(f"\nDetection Results:")
        print(f"Keypoints detected: {len(kp.positions)}")
        print(f"Confidences (avg: {np.mean(kp.confidences):.3f}):")
        for i, conf in enumerate(kp.confidences):
            status = "GOOD" if conf >= 0.3 else "LOW"
            print(f"  Joint {i}: {conf:.3f} [{status}]")
            
        print(f"\nPositions (first 3):")
        for i in range(3):
            print(f"  Joint {i}: {kp.positions[i]}")
    else:
        print("No pose detected.")

if __name__ == "__main__":
    main()
