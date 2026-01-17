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
    squares_x: int = 5,
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
    
    
    # Use CharucoDetector (OpenCV 4.7+)
    # Note: interpolateCornersCharuco is deprecated/moved in newer versions
    
    detector = cv2.aruco.CharucoDetector(board)
    
    # Set detector parameters if needed
    detector_params = cv2.aruco.DetectorParameters()
    detector.setDetectorParameters(detector_params)
    
    # Detect
    # detectBoard returns (charucoCorners, charucoIds, markerCorners, markerIds)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    
    if marker_ids is not None:
        result['marker_corners'] = marker_corners
        result['marker_ids'] = marker_ids
        cv2.aruco.drawDetectedMarkers(result['image_with_markers'], marker_corners, marker_ids)
    
    if charuco_corners is None or len(charuco_corners) < 4:
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


def generate_charuco_img(
    squares_x: int = 5,
    squares_y: int = 5,
    square_length: float = 0.04,
    marker_length: float = 0.03,
    dictionary: str = "DICT_6X6_250",
    dpi: int = 300,
) -> np.ndarray:
    """
    Generate a ChArUco board image for display or legacy PNG export.
    Returns: BGR numpy image
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
    info_text = f"ChArUco: {squares_x}x{squares_y} | Sq: {square_length*100:.1f}cm | Mk: {marker_length*100:.1f}cm"
    cv2.putText(
        canvas, info_text, 
        (border, img_height + border + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    
    return canvas


def generate_charuco_pdf_file(
    output_path: Path,
    squares_x: int = 5,
    squares_y: int = 5,
    square_length: float = 0.04,
    marker_length: float = 0.03,
    dictionary: str = "DICT_6X6_250",
) -> None:
    """
    Generate a printable PDF ChArUco board using FPDF.
    
    Args:
        output_path: Path to save the PDF
        squares_x: Number of squares in X
        squares_y: Number of squares in Y
        square_length: Square size in meters
        marker_length: Marker size in meters
        dictionary: ArUco dictionary name
    """
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf not found. Please install: pip install fpdf")
        return

    # Create PDF (A4 size: 210x297mm, Letter: 215.9x279.4mm)
    # Using Letter size as generic default, but centering the image
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.add_page()
    
    # Original board width in mm
    board_w_mm = squares_x * square_length * 1000
    board_h_mm = squares_y * square_length * 1000
    
    # Generate just the board raw image for precise scaling in PDF
    # We use a high resolution temp image
    board, _ = create_charuco_board(squares_x, squares_y, square_length, marker_length, dictionary)
    w_px = int(squares_x * square_length * 300 / 0.0254) # 300 DPI
    h_px = int(squares_y * square_length * 300 / 0.0254)
    raw_board_img = board.generateImage((w_px, h_px), marginSize=0)
    
    # Save temp image
    temp_img_path = output_path.with_suffix('.temp.png')
    cv2.imwrite(str(temp_img_path), raw_board_img)

    # Center on page
    page_w = 215.9
    page_h = 279.4
    x = (page_w - board_w_mm) / 2
    y = (page_h - board_h_mm) / 2
    
    # Verify it fits
    if x < 0 or y < 0:
        print(f"Warning: Board size ({board_w_mm:.1f}x{board_h_mm:.1f}mm) is too large for page!")
    
    pdf.image(str(temp_img_path), x=x, y=y, w=board_w_mm, h=board_h_mm)
    
    # Add text
    pdf.set_xy(x, y + board_h_mm + 5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"VoxelVR Calibration | {squares_x}x{squares_y} | Square: {square_length*100:.1f}cm | Marker: {marker_length*100:.1f}cm", align='C')
    pdf.ln(5)
    pdf.cell(0, 10, "PRINT AT 100% SCALE (DO NOT SCALE TO FIT)", align='C')
    
    pdf.output(str(output_path))
    
    # Cleanup
    if temp_img_path.exists():
        temp_img_path.unlink()
        
    print(f"ChArUco board PDF saved to: {output_path}")


def generate_charuco_pdf(
    output_path: Path,
    squares_x: int = 5,
    squares_y: int = 5,
    square_length: float = 0.04,
    marker_length: float = 0.03,
    dictionary: str = "DICT_6X6_250",
    dpi: int = 300,
    page_size: Tuple[float, float] = (8.5, 11),
) -> None:
    """
    Legacy wrapper. Checks extension:
    - If .pdf, calls generate_charuco_pdf_file
    - If .png/.jpg, generates image and saves it
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.pdf':
        generate_charuco_pdf_file(
            output_path, squares_x, squares_y, square_length, marker_length, dictionary
        )
    else:
        img = generate_charuco_img(squares_x, squares_y, square_length, marker_length, dictionary, dpi)
        cv2.imwrite(str(output_path), img)
        print(f"ChArUco board image saved to: {output_path}")


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
    
    # Manually match points since estimatePoseCharucoBoard is deprecated/missing
    try:
        # Get all board corners (N_total, 3)
        all_board_corners = board.getChessboardCorners()
        
        # Select the ones we see
        # ids is (N, 1) or (N,), flatten to indexing array
        obj_points = all_board_corners[ids.flatten()]
        
        # corners is (N, 1, 2) or (N, 2), solvePnP handles both
        
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            corners,
            camera_matrix,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        return success, rvec, tvec
        
    except Exception as e:
        print(f"Pose estimation failed: {e}")
        return False, None, None
