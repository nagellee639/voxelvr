
import cv2
import sys
import numpy as np

def check_stats(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    print(f"Stats for {image_path}:")
    print(f"  Resolution: {img.shape}")
    print(f"  Mean: {np.mean(img):.2f}")
    print(f"  Std Dev: {np.std(img):.2f}")
    print(f"  Min: {np.min(img)}")
    print(f"  Max: {np.max(img)}")
    
    if np.mean(img) < 10:
        print("  WARNING: Image is very dark/black!")
    
    if np.std(img) < 5:
        print("  WARNING: Image has very low contrast!")

if __name__ == "__main__":
    check_stats(sys.argv[1])
