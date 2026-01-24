import os
import cv2
import sys

# Suppress OpenCV's noisy warnings about non-existent backends/sensors
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


def find_available_cameras(max_index=10):
    """Scan for available camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def test_camera(camera_index=None):
    """
    Test camera feed with live preview.
    
    Controls:
        q - Quit
        s - Save screenshot
        c - Cycle to next available camera
    """
    # Auto-detect cameras if no index specified
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("Error: No cameras found on this system.")
        return
    
    print(f"Available cameras: {available_cameras}")
    
    # Use specified index or first available
    if camera_index is None:
        camera_index = available_cameras[0]
    elif camera_index not in available_cameras:
        print(f"Warning: Camera {camera_index} not available. Using camera {available_cameras[0]} instead.")
        camera_index = available_cameras[0]
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nCamera {camera_index} initialized successfully.")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("  c - Cycle to next camera")
    
    screenshot_count = 0
    current_cam_idx = available_cameras.index(camera_index)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting...")
            break
        
        # Add info overlay
        info_text = f"Camera {camera_index} | {width}x{height}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nExiting...")
            break
        elif key == ord('s'):
            filename = f"screenshot_{screenshot_count:03d}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            screenshot_count += 1
        elif key == ord('c') and len(available_cameras) > 1:
            # Cycle to next camera
            cap.release()
            current_cam_idx = (current_cam_idx + 1) % len(available_cameras)
            camera_index = available_cameras[current_cam_idx]
            cap = cv2.VideoCapture(camera_index)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"\nSwitched to camera {camera_index} ({width}x{height})")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Allow camera index as command line argument
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test_camera(cam_idx)
