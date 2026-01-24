"""
Camera Calibration using OpenCV

This script captures images of a checkerboard pattern and computes
camera intrinsic parameters and distortion coefficients.

Instructions:
1. Print a checkerboard pattern (or display on a tablet/monitor)
   - Download from: https://github.com/opencv/opencv/blob/master/doc/pattern.png
   - Or use: https://calib.io/pages/camera-calibration-pattern-generator
2. Run this script
3. Show the checkerboard to the camera from different angles/distances
4. Press SPACE to capture when corners are detected (shown in color)
5. Capture 15-20 images from various angles
6. Press 'q' to finish and compute calibration

Controls:
    SPACE - Capture current frame (when corners detected)
    q     - Finish capturing and calibrate
    c     - Cycle to next camera
    ESC   - Exit without calibrating
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION_DIR = os.path.join(SCRIPT_DIR, "calibration_data")


def find_available_cameras(max_index=5):
    """Scan for available camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def calibrate_camera(
    checkerboard_size=(9, 6),  # Inner corners (columns, rows)
    square_size=25.0,          # Size of square in mm (for real-world scale)
    camera_index=None,
    output_dir=None
):
    # Use default calibration directory (under script's folder) if not specified
    if output_dir is None:
        output_dir = DEFAULT_CALIBRATION_DIR
    """
    Perform camera calibration using a checkerboard pattern.
    
    Args:
        checkerboard_size: Number of inner corners (width, height)
        square_size: Size of each square in mm
        camera_index: Camera to use (None for auto-detect)
        output_dir: Directory to save calibration results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find available cameras
    available_cameras = find_available_cameras()
    if not available_cameras:
        print("Error: No cameras found!")
        return None
    
    print(f"Available cameras: {available_cameras}")
    
    if camera_index is None:
        camera_index = available_cameras[0]
    elif camera_index not in available_cameras:
        print(f"Camera {camera_index} not available, using {available_cameras[0]}")
        camera_index = available_cameras[0]
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    
    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nCamera {camera_index} opened ({width}x{height})")
    print(f"Checkerboard size: {checkerboard_size[0]}x{checkerboard_size[1]} inner corners")
    print(f"Square size: {square_size}mm")
    print("\n" + "="*50)
    print("Show the checkerboard pattern to the camera.")
    print("When corners are detected, they'll be drawn in color.")
    print("="*50)
    print("\nControls:")
    print("  SPACE - Capture frame (when corners detected)")
    print("  q     - Finish and calibrate")
    print("  c     - Switch camera")
    print("  ESC   - Exit without calibrating")
    print()
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to actual size
    
    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    
    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    captured_count = 0
    current_cam_idx = available_cameras.index(camera_index)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
        
        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(
            gray, checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # Create display frame
        display = frame.copy()
        
        if found:
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners on display frame
            cv2.drawChessboardCorners(display, checkerboard_size, corners_refined, found)
            
            status_color = (0, 255, 0)  # Green - ready to capture
            status_text = "CORNERS DETECTED - Press SPACE to capture"
        else:
            status_color = (0, 0, 255)  # Red - no corners
            status_text = "Searching for checkerboard..."
        
        # Draw status bar
        cv2.rectangle(display, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, status_color, 2)
        
        # Draw capture count
        count_text = f"Captured: {captured_count}/15+ | Camera: {camera_index}"
        cv2.putText(display, count_text, (width - 300, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        
        cv2.imshow('Camera Calibration', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\nCalibration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        
        elif key == ord('q'):
            if captured_count < 5:
                print(f"\nNeed at least 5 captures (have {captured_count}). Keep capturing or press ESC to cancel.")
            else:
                print(f"\nFinishing with {captured_count} captures...")
                break
        
        elif key == ord('c') and len(available_cameras) > 1:
            cap.release()
            current_cam_idx = (current_cam_idx + 1) % len(available_cameras)
            camera_index = available_cameras[current_cam_idx]
            cap = cv2.VideoCapture(camera_index)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Switched to camera {camera_index}")
        
        elif key == ord(' ') and found:
            # Capture this frame
            obj_points.append(objp)
            img_points.append(corners_refined)
            captured_count += 1
            
            # Save the captured image
            img_path = os.path.join(output_dir, f"capture_{captured_count:02d}.png")
            cv2.imwrite(img_path, frame)
            print(f"Captured {captured_count}: {img_path}")
            
            # Flash effect
            cv2.imshow('Camera Calibration', np.ones_like(display) * 255)
            cv2.waitKey(100)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if captured_count < 5:
        print("Not enough captures for calibration.")
        return None
    
    # Perform calibration
    print("\nCalibrating camera...")
    print("This may take a moment...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (width, height), None, None
    )
    
    if not ret:
        print("Calibration failed!")
        return None
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], 
                                                camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        total_error += error
    mean_error = total_error / len(obj_points)
    
    # Display results
    print("\n" + "="*50)
    print("CALIBRATION RESULTS")
    print("="*50)
    print(f"\nReprojection Error: {mean_error:.4f} pixels")
    print("(Lower is better, < 0.5 is good, < 0.3 is excellent)")
    
    print(f"\nCamera Matrix:\n{camera_matrix}")
    print(f"\nDistortion Coefficients:\n{dist_coeffs.ravel()}")
    
    # Extract focal length and principal point
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"\nFocal Length: fx={fx:.2f}, fy={fy:.2f} pixels")
    print(f"Principal Point: cx={cx:.2f}, cy={cy:.2f} pixels")
    
    # Save calibration data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as NumPy files
    np.save(os.path.join(output_dir, "camera_matrix.npy"), camera_matrix)
    np.save(os.path.join(output_dir, "dist_coeffs.npy"), dist_coeffs)
    
    # Save as JSON for readability
    calibration_data = {
        "timestamp": timestamp,
        "camera_index": camera_index,
        "resolution": {"width": width, "height": height},
        "checkerboard_size": list(checkerboard_size),
        "square_size_mm": square_size,
        "num_captures": captured_count,
        "reprojection_error": mean_error,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "focal_length": {"fx": fx, "fy": fy},
        "principal_point": {"cx": cx, "cy": cy}
    }
    
    json_path = os.path.join(output_dir, "calibration.json")
    with open(json_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\nCalibration saved to: {output_dir}/")
    print(f"  - camera_matrix.npy")
    print(f"  - dist_coeffs.npy")
    print(f"  - calibration.json")
    
    # Show undistortion preview
    show_undistortion_preview(camera_index, camera_matrix, dist_coeffs)
    
    return calibration_data


def show_undistortion_preview(camera_index, camera_matrix, dist_coeffs):
    """Show live preview of original vs undistorted image."""
    print("\n" + "="*50)
    print("UNDISTORTION PREVIEW")
    print("Press any key to exit preview")
    print("="*50)
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Undistort
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop to ROI
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted_cropped = undistorted[y:y+h, x:x+w]
        else:
            undistorted_cropped = undistorted
        
        # Resize for side-by-side display
        display_height = 480
        scale = display_height / height
        
        original_small = cv2.resize(frame, None, fx=scale, fy=scale)
        undistorted_small = cv2.resize(undistorted, None, fx=scale, fy=scale)
        
        # Add labels
        cv2.putText(original_small, "Original", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(undistorted_small, "Undistorted", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Side by side
        combined = np.hstack([original_small, undistorted_small])
        
        cv2.imshow('Calibration Preview (Press any key to exit)', combined)
        
        if cv2.waitKey(1) & 0xFF != 255:
            break
    
    cap.release()
    cv2.destroyAllWindows()


def load_calibration(calibration_dir=None):
    """Load saved calibration data."""
    if calibration_dir is None:
        calibration_dir = DEFAULT_CALIBRATION_DIR
    camera_matrix = np.load(os.path.join(calibration_dir, "camera_matrix.npy"))
    dist_coeffs = np.load(os.path.join(calibration_dir, "dist_coeffs.npy"))
    return camera_matrix, dist_coeffs


if __name__ == "__main__":
    import sys
    
    # Default checkerboard: 9x6 inner corners, 25mm squares
    # Adjust these values to match YOUR checkerboard pattern!
    CHECKERBOARD_SIZE = (9, 6)  # (columns, rows) of INNER corners
    SQUARE_SIZE = 25.0          # Size of each square in mm
    
    # Parse command line camera index
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    print("="*50)
    print("CAMERA CALIBRATION")
    print("="*50)
    print(f"\nExpecting checkerboard with {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE}mm")
    print("\nIf your checkerboard is different, edit CHECKERBOARD_SIZE at the")
    print("bottom of this script. Count INNER corners, not squares!")
    print("\nExample: A 10x7 squares board has 9x6 inner corners")
    print()
    
    calibrate_camera(
        checkerboard_size=CHECKERBOARD_SIZE,
        square_size=SQUARE_SIZE,
        camera_index=cam_idx
    )
