"""
Real-World 2D Measurement using Perspective Projection (Manual Distance)

This script measures real-world dimensions of objects using:
1. Camera calibration parameters (from step1_calibrate.py)
2. Manually specified distance from camera to object plane
3. Perspective projection equations

Perspective Projection Equations:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

Where:
    - (u, v) = pixel coordinates in image
    - (cx, cy) = principal point (optical center)
    - (fx, fy) = focal lengths in pixels
    - Z = distance from camera to object plane (you measure this!)
    - (X, Y) = real-world coordinates in mm

Controls:
    Left Click   - Place measurement points
    Right Click  - Clear all points
    r            - Toggle rectangle mode (measure width x height)
    s            - Save current frame with measurements
    c            - Cycle camera
    d            - Change distance to object
    q            - Quit
"""

import cv2
import numpy as np
import os
import json
import math

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION_DIR = os.path.join(SCRIPT_DIR, "calibration_data")
MEASURE_DATA_DIR = os.path.join(SCRIPT_DIR, "measure_data")


class PerspectiveMeasurer:
    """Measure real-world dimensions using perspective projection."""
    
    def __init__(self, camera_matrix, dist_coeffs, distance_mm=300.0):
        """
        Initialize the measurer.
        
        Args:
            camera_matrix: 3x3 intrinsic camera matrix
            dist_coeffs: Distortion coefficients
            distance_mm: Distance from camera to object plane in mm
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.distance_mm = distance_mm
        
        # Extract intrinsic parameters
        self.fx = camera_matrix[0, 0]  # Focal length x
        self.fy = camera_matrix[1, 1]  # Focal length y
        self.cx = camera_matrix[0, 2]  # Principal point x
        self.cy = camera_matrix[1, 2]  # Principal point y
        
        print(f"Camera intrinsics loaded:")
        print(f"  Focal length: fx={self.fx:.2f}, fy={self.fy:.2f} pixels")
        print(f"  Principal point: cx={self.cx:.2f}, cy={self.cy:.2f} pixels")
    
    def pixel_to_world(self, u, v):
        """
        Convert pixel coordinates to real-world coordinates.
        
        Uses perspective projection equations:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
        
        Args:
            u, v: Pixel coordinates
            
        Returns:
            (X, Y): Real-world coordinates in mm relative to optical axis
        """
        X = (u - self.cx) * self.distance_mm / self.fx
        Y = (v - self.cy) * self.distance_mm / self.fy
        return X, Y
    
    def measure_distance(self, p1, p2):
        """
        Measure real-world distance between two pixel points.
        
        Args:
            p1, p2: Pixel coordinates as (u, v) tuples
            
        Returns:
            Distance in mm
        """
        # Convert both points to world coordinates
        X1, Y1 = self.pixel_to_world(p1[0], p1[1])
        X2, Y2 = self.pixel_to_world(p2[0], p2[1])
        
        # Calculate Euclidean distance
        distance = math.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)
        return distance
    
    def measure_rectangle(self, p1, p2):
        """
        Measure width and height of a rectangle defined by two corners.
        
        Args:
            p1, p2: Opposite corner pixel coordinates
            
        Returns:
            (width_mm, height_mm)
        """
        # Get the four corners in pixel space
        u1, v1 = p1
        u2, v2 = p2
        
        # Convert to world coordinates
        X1, Y1 = self.pixel_to_world(u1, v1)
        X2, Y2 = self.pixel_to_world(u2, v2)
        
        # Width and height
        width = abs(X2 - X1)
        height = abs(Y2 - Y1)
        
        return width, height
    
    def pixel_distance(self, p1, p2):
        """Calculate pixel distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def get_pixels_per_mm(self):
        """Calculate approximate pixels per mm at the current distance."""
        # Average of x and y scale factors
        px_per_mm_x = self.fx / self.distance_mm
        px_per_mm_y = self.fy / self.distance_mm
        return (px_per_mm_x + px_per_mm_y) / 2


def load_calibration(calibration_dir=None):
    """Load camera calibration data."""
    if calibration_dir is None:
        calibration_dir = DEFAULT_CALIBRATION_DIR
    
    matrix_path = os.path.join(calibration_dir, "camera_matrix.npy")
    dist_path = os.path.join(calibration_dir, "dist_coeffs.npy")
    
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"Calibration not found at {calibration_dir}\n"
            "Run step1_calibrate.py first!"
        )
    
    camera_matrix = np.load(matrix_path)
    dist_coeffs = np.load(dist_path)
    
    # Load JSON for additional info
    json_path = os.path.join(calibration_dir, "calibration.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            calibration_info = json.load(f)
        print(f"Loaded calibration from {calibration_info['timestamp']}")
        print(f"Reprojection error: {calibration_info['reprojection_error']:.4f} pixels")
    
    return camera_matrix, dist_coeffs


def find_available_cameras(max_index=5):
    """Scan for available camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def get_distance_from_user(current_distance):
    """Prompt user for distance input."""
    print(f"\nCurrent distance: {current_distance:.1f} mm")
    try:
        new_distance = input("Enter new distance to object (mm), or press Enter to keep current: ")
        if new_distance.strip():
            return float(new_distance)
    except ValueError:
        print("Invalid input, keeping current distance.")
    return current_distance


def run_measurement(camera_index=None, initial_distance=300.0):
    """
    Run the interactive measurement tool.
    
    Args:
        camera_index: Camera to use (None for auto-detect)
        initial_distance: Initial distance to object plane in mm
    """
    # Load calibration
    try:
        camera_matrix, dist_coeffs = load_calibration()
    except FileNotFoundError as e:
        print(e)
        return
    
    # Initialize measurer
    measurer = PerspectiveMeasurer(camera_matrix, dist_coeffs, initial_distance)
    
    # Find cameras
    available_cameras = find_available_cameras()
    if not available_cameras:
        print("No cameras found!")
        return
    
    if camera_index is None:
        camera_index = available_cameras[0]
    elif camera_index not in available_cameras:
        print(f"Camera {camera_index} not available, using {available_cameras[0]}")
        camera_index = available_cameras[0]
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get optimal camera matrix for undistortion
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    
    print(f"\nCamera {camera_index} opened ({width}x{height})")
    print(f"Distance to object: {measurer.distance_mm:.1f} mm")
    print(f"Scale: {measurer.get_pixels_per_mm():.2f} pixels/mm")
    print("\n" + "="*50)
    print("MEASUREMENT TOOL (Manual Distance)")
    print("="*50)
    print("Controls:")
    print("  Left Click  - Place measurement point")
    print("  Right Click - Clear all points")
    print("  r - Toggle rectangle mode")
    print("  d - Change distance to object")
    print("  s - Save screenshot")
    print("  c - Cycle camera")
    print("  q - Quit")
    print("="*50)
    
    # State
    points = []
    rectangle_mode = False
    current_cam_idx = available_cameras.index(camera_index)
    screenshot_count = 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            # Undistort the clicked point
            point = np.array([[[x, y]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(point, camera_matrix, dist_coeffs, P=new_camera_matrix)
            ux, uy = undistorted[0][0]
            points.append((int(ux), int(uy)))
            
            if len(points) > 2 and not rectangle_mode:
                points = points[-2:]  # Keep only last 2 points for line measurement
            elif len(points) > 2 and rectangle_mode:
                points = points[-2:]  # Keep only last 2 points for rectangle
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            points = []
    
    cv2.namedWindow('Measurement')
    cv2.setMouseCallback('Measurement', mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Undistort frame
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        display = frame.copy()
        
        # Draw measurement info bar
        cv2.rectangle(display, (0, 0), (width, 50), (40, 40, 40), -1)
        mode_text = "RECTANGLE" if rectangle_mode else "LINE"
        info_text = f"Mode: {mode_text} | Distance: {measurer.distance_mm:.0f}mm | Camera: {camera_index}"
        cv2.putText(display, info_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw points
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
            cv2.circle(display, pt, 8, (0, 255, 0), 2)
        
        # Draw measurements
        if len(points) >= 2:
            p1, p2 = points[-2], points[-1]
            
            if rectangle_mode:
                # Draw rectangle
                cv2.rectangle(display, p1, p2, (0, 255, 255), 2)
                
                # Measure dimensions
                width_mm, height_mm = measurer.measure_rectangle(p1, p2)
                
                # Draw dimension labels
                mid_top = ((p1[0] + p2[0]) // 2, min(p1[1], p2[1]) - 10)
                mid_side = (max(p1[0], p2[0]) + 10, (p1[1] + p2[1]) // 2)
                
                # Width label
                cv2.putText(display, f"{width_mm:.1f}mm", mid_top, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # Height label
                cv2.putText(display, f"{height_mm:.1f}mm", mid_side,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Area
                area = width_mm * height_mm
                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(display, f"Area: {area:.1f}mm2", (center[0]-50, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                # Draw line
                cv2.line(display, p1, p2, (0, 255, 0), 2)
                
                # Measure distance
                distance_mm = measurer.measure_distance(p1, p2)
                pixel_dist = measurer.pixel_distance(p1, p2)
                
                # Draw distance label at midpoint
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(display, f"{distance_mm:.1f}mm", (mid[0]+10, mid[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"({pixel_dist:.0f}px)", (mid[0]+10, mid[1]+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Instructions at bottom
        if len(points) < 2:
            help_text = "Click two points to measure" + (" rectangle" if rectangle_mode else " distance")
            cv2.putText(display, help_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('Measurement', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            rectangle_mode = not rectangle_mode
            points = []
            print(f"Mode: {'Rectangle' if rectangle_mode else 'Line'}")
        
        elif key == ord('d'):
            cv2.destroyAllWindows()
            measurer.distance_mm = get_distance_from_user(measurer.distance_mm)
            print(f"New scale: {measurer.get_pixels_per_mm():.2f} pixels/mm")
            cv2.namedWindow('Measurement')
            cv2.setMouseCallback('Measurement', mouse_callback)
        
        elif key == ord('s'):
            os.makedirs(MEASURE_DATA_DIR, exist_ok=True)
            filename = os.path.join(MEASURE_DATA_DIR, f"measurement_{screenshot_count:03d}.png")
            cv2.imwrite(filename, display)
            print(f"Saved: {filename}")
            screenshot_count += 1
        
        elif key == ord('c') and len(available_cameras) > 1:
            cap.release()
            current_cam_idx = (current_cam_idx + 1) % len(available_cameras)
            camera_index = available_cameras[current_cam_idx]
            cap = cv2.VideoCapture(camera_index)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (width, height), 1, (width, height)
            )
            
            points = []
            print(f"Switched to camera {camera_index}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    # Default distance to object plane (mm)
    # Measure this from your camera lens to the object!
    DEFAULT_DISTANCE = 2500.0  # 2.5 m
    
    print("="*50)
    print("REAL-WORLD MEASUREMENT TOOL (Manual Distance)")
    print("="*50)
    print("\nUsing perspective projection equations:")
    print("  X = (u - cx) * Z / fx")
    print("  Y = (v - cy) * Z / fy")
    print("\nIMPORTANT: Measure the distance from camera to object")
    print("and press 'd' to set it for accurate measurements!")
    print()
    
    # Parse command line arguments
    cam_idx = None
    distance = DEFAULT_DISTANCE
    
    if len(sys.argv) > 1:
        cam_idx = int(sys.argv[1])
    if len(sys.argv) > 2:
        distance = float(sys.argv[2])
    
    run_measurement(camera_index=cam_idx, initial_distance=distance)
