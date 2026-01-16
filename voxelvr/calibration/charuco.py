"""
ChArUco Board Detection and Generation

ChArUco boards combine ArUco markers with a chessboard pattern,
providing robust detection even with partial occlusion.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# ArUco dictionary mapping
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
}


def create_charuco_board(
    squares_x: int = 7,
    squares_y: int = 5,
    square_length: float = 0.04,  # meters
    marker_length: float = 0.03,  # meters
    dictionary: str = "DICT_6X6_250",
) -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    """
    Create a ChArUco board for calibration.
    
    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Length of each square in meters
        marker_length: Length of ArUco marker in meters
        dictionary: ArUco dictionary name
        
    Returns:
        Tuple of (CharucoBoard, Dictionary)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary])
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )
    return board, aruco_dict


def detect_charuco(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    aruco_dict: cv2.aruco.Dictionary,
    camera_matrix: Optional[np.ndarray] = None,
    distortion_coeffs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Detect ChArUco board corners in an image.
    
    Args:
        image: Input BGR image
        board: ChArUco board object
        aruco_dict: ArUco dictionary
        camera_matrix: Optional camera intrinsic matrix (for refinement)
        distortion_coeffs: Optional distortion coefficients
        
    Returns:
        Dictionary with detection results:
        - 'success': bool - whether board was detected
        - 'corners': detected charuco corners (Nx1x2)
        - 'ids': corner IDs
        - 'marker_corners': ArUco marker corners
        - 'marker_ids': ArUco marker IDs
        - 'image_with_markers': image with markers drawn
    """
    result = {
        'success': False,
        'corners': None,
        'ids': None,
        'marker_corners': None,
        'marker_ids': None,
        'image_with_markers': image.copy(),
    }
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect ArUco markers
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    
    if marker_ids is None or len(marker_ids) < 4:
        return result
    
    result['marker_corners'] = marker_corners
    result['marker_ids'] = marker_ids
    
    # Draw detected markers
    cv2.aruco.drawDetectedMarkers(result['image_with_markers'], marker_corners, marker_ids)
    
    # Interpolate ChArUco corners
    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coeffs
    )
    
    if num_corners < 4:
        return result
    
    result['success'] = True
    result['corners'] = charuco_corners
    result['ids'] = charuco_ids
    
    # Draw ChArUco corners
    cv2.aruco.drawDetectedCornersCharuco(
        result['image_with_markers'], 
        charuco_corners, 
        charuco_ids,
        cornerColor=(0, 255, 0)
    )
    
    return result


def generate_charuco_pdf(
    output_path: Path,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length: float = 0.04,
    marker_length: float = 0.03,
    dictionary: str = "DICT_6X6_250",
    dpi: int = 300,
    page_size: Tuple[float, float] = (8.5, 11),  # inches (letter size)
) -> None:
    """
    Generate a printable ChArUco board image.
    
    Args:
        output_path: Path to save the image
        squares_x: Number of squares in X
        squares_y: Number of squares in Y
        square_length: Square size in meters
        marker_length: Marker size in meters
        dictionary: ArUco dictionary name
        dpi: Dots per inch for output
        page_size: Page size in inches (width, height)
    """
    board, aruco_dict = create_charuco_board(
        squares_x, squares_y, square_length, marker_length, dictionary
    )
    
    # Calculate image size
    # Board size in meters
    board_width_m = squares_x * square_length
    board_height_m = squares_y * square_length
    
    # Convert to inches (1 inch = 0.0254 meters)
    board_width_in = board_width_m / 0.0254
    board_height_in = board_height_m / 0.0254
    
    # Calculate pixels
    img_width = int(board_width_in * dpi)
    img_height = int(board_height_in * dpi)
    
    # Generate board image
    board_img = board.generateImage((img_width, img_height), marginSize=0)
    
    # Add border and info text
    border = 50
    canvas = np.ones((img_height + 2*border + 100, img_width + 2*border), dtype=np.uint8) * 255
    canvas[border:border+img_height, border:border+img_width] = board_img
    
    # Add calibration info
    info_text = f"ChArUco Board: {squares_x}x{squares_y} | Square: {square_length*100:.1f}cm | Marker: {marker_length*100:.1f}cm"
    cv2.putText(
        canvas, info_text, 
        (border, img_height + border + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"ChArUco board saved to: {output_path}")
    print(f"Print at 100% scale. Square size: {square_length*100:.1f}cm")


def estimate_pose(
    corners: np.ndarray,
    ids: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Estimate the pose of the ChArUco board.
    
    Args:
        corners: Detected ChArUco corners
        ids: Corner IDs
        board: ChArUco board object
        camera_matrix: Camera intrinsic matrix
        distortion_coeffs: Distortion coefficients
        
    Returns:
        Tuple of (success, rotation_vector, translation_vector)
    """
    if corners is None or len(corners) < 4:
        return False, None, None
    
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        corners, ids, board, camera_matrix, distortion_coeffs, None, None
    )
    
    return success, rvec, tvec
