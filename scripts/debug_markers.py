
import cv2
import sys

def debug_markers(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    print(f"Image resolution: {img.shape[1]}x{img.shape[0]}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # List of all standard dictionaries
    dicts = [
        (attr, getattr(cv2.aruco, attr)) 
        for attr in dir(cv2.aruco) 
        if attr.startswith("DICT_")
    ]

    print(f"Testing {len(dicts)} dictionaries for raw markers...")

    found_something = False
    for name, dict_id in dicts:
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if len(corners) > 0:
            print(f"  {name}: Found {len(corners)} markers")
            found_something = True
            
    if not found_something:
        print("No markers found with ANY standard dictionary.")

if __name__ == "__main__":
    debug_markers(sys.argv[1])
