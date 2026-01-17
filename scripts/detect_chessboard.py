
import cv2
import sys
import numpy as np

def try_detect_chessboard(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Common chessboard sizes (internal corners)
    # usually (cols-1, rows-1)
    sizes = [
        (9, 6), (6, 9),
        (8, 5), (5, 8),
        (7, 5), (5, 7),
        (8, 6), (6, 8),
        (10, 7), (7, 10),
        (9, 7), (7, 9),
    ]

    print(f"Checking {image_path} for chessboard...")

    for nx, ny in sizes:
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            print(f"  SUCCESS: Found chessboard with internal corners {nx}x{ny}")
            return

    print("No chessboard found with standard sizes.")

if __name__ == "__main__":
    try_detect_chessboard(sys.argv[1])
