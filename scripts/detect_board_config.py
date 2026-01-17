
import cv2
import cv2.aruco as aruco
import sys

def try_detect(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    # Trial parameters
    # Common dictionaries
    dicts = [
        ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
        ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
        ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
        ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
        ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
        ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
        ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
        ("DICT_APRILTAG_36h11", cv2.aruco.DICT_APRILTAG_36h11),
    ]
    
    # Common sizes (squares_x, squares_y)
    sizes = [
        (5, 7), (7, 5),
        (5, 8), (8, 5),
        (6, 8), (8, 6),
        (6, 9), (9, 6),
        (7, 10), (10, 7),
        (8, 11), (11, 8),
        (8, 12), (12, 8),
    ]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for dict_name, dict_id in dicts:
        print(f"Checking dictionary {dict_name}...")
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        
        # Now try board configurations
        for nx, ny in sizes:
            board = cv2.aruco.CharucoBoard((nx, ny), 0.04, 0.03, dictionary)
            detector = cv2.aruco.CharucoDetector(board)
            
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 6:
                print(f"  SUCCESS: Found {len(charuco_corners)} corners with board ({nx}x{ny}) and dictionary {dict_name}")
                return # Found a match


if __name__ == "__main__":
    try_detect(sys.argv[1])
