"""
Eye Blink Rate Detector

This script detects eye blinks in real-time using a webcam feed and calculates
the blink rate (blinks per second). It supports two modes for comparison:
- 'movie': Record blink rate while watching a movie/video
- 'reading': Record blink rate while reading a document

Theory:
    The Eye Aspect Ratio (EAR) is used to detect blinks. EAR is calculated using
    6 landmark points around each eye:
    
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Where p1-p6 are the eye landmarks (p1 and p4 are the horizontal extremes,
    p2, p3 are upper lid points, and p5, p6 are lower lid points).
    
    When the eye is open, EAR is relatively constant (~0.25-0.30).
    When the eye closes, EAR drops significantly (~0.15 or below).
    A blink is detected when EAR falls below a threshold for consecutive frames.

Usage:
    # Run a 1-minute movie watching session
    python blink_detector.py --mode movie
    
    # Run a 1-minute reading session
    python blink_detector.py --mode reading
    
    # Run a 30-second session
    python blink_detector.py --mode movie --duration 30
    
    # Compare results from all sessions
    python blink_detector.py --compare
    
    # Select a specific camera
    python blink_detector.py --mode movie --camera 1

Controls:
    q - Quit the session early
    c - Cycle through available cameras
    r - Reset blink counter

Author: CSC8830 Assignment
"""

import os
import cv2
import csv
import argparse
import numpy as np
import mediapipe as mp
from datetime import datetime
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
SESSIONS_DIR = os.path.join(OUTPUT_DIR, "sessions")
SESSIONS_FILE = os.path.join(SESSIONS_DIR, "blink_sessions.csv")

# Ensure output directories exist
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Eye Aspect Ratio threshold and consecutive frames for blink detection
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 2

# MediaPipe Face Mesh eye landmark indices
# Left eye landmarks (from user's perspective, right side of image)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye landmarks (from user's perspective, left side of image)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def find_available_cameras(max_index: int = 10) -> List[int]:
    """
    Find all available camera indices.
    
    Args:
        max_index: Maximum camera index to check
        
    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for a single eye.
    
    The EAR formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Args:
        eye_landmarks: Array of 6 eye landmark points [(x, y), ...]
                      ordered as [p1, p2, p3, p4, p5, p6] where:
                      p1 = left corner, p4 = right corner (horizontal)
                      p2, p3 = upper lid points
                      p5, p6 = lower lid points
    
    Returns:
        Eye Aspect Ratio value
    """
    # Compute euclidean distances
    # Vertical distances
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # p2 - p6
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # p3 - p5
    
    # Horizontal distance
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])   # p1 - p4
    
    # Calculate EAR
    if h == 0:
        return 0.0
    ear = (v1 + v2) / (2.0 * h)
    return ear


def get_eye_landmarks(face_landmarks, indices: List[int], 
                      frame_width: int, frame_height: int) -> np.ndarray:
    """
    Extract eye landmark coordinates from face mesh landmarks.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        indices: List of landmark indices for the eye
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        
    Returns:
        Array of (x, y) coordinates for each landmark
    """
    landmarks = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        landmarks.append([x, y])
    return np.array(landmarks)


def draw_eye_landmarks(frame: np.ndarray, landmarks: np.ndarray, 
                       color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    """
    Draw eye landmarks on the frame.
    
    Args:
        frame: Video frame to draw on
        landmarks: Array of (x, y) landmark coordinates
        color: BGR color tuple for the landmarks
    """
    for point in landmarks:
        cv2.circle(frame, tuple(point), 2, color, -1)
    
    # Draw eye contour
    hull = cv2.convexHull(landmarks)
    cv2.drawContours(frame, [hull], -1, color, 1)


def save_session(mode: str, duration: float, blink_count: int) -> None:
    """
    Save session data to CSV file.
    
    Args:
        mode: Session mode ('movie' or 'reading')
        duration: Actual session duration in seconds
        blink_count: Number of blinks detected
    """
    blinks_per_second = (blink_count / duration) if duration > 0 else 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(SESSIONS_FILE)
    
    with open(SESSIONS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'mode', 'duration_sec', 'blink_count', 'blinks_per_second'])
        writer.writerow([timestamp, mode, f"{duration:.1f}", blink_count, f"{blinks_per_second:.3f}"])
    
    print(f"\n✓ Session saved to {SESSIONS_FILE}")


def load_sessions() -> List[dict]:
    """
    Load all session data from CSV file.
    
    Returns:
        List of session dictionaries
    """
    sessions = []
    if not os.path.exists(SESSIONS_FILE):
        return sessions
    
    with open(SESSIONS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append({
                'timestamp': row['timestamp'],
                'mode': row['mode'],
                'duration_sec': float(row['duration_sec']),
                'blink_count': int(row['blink_count']),
                'blinks_per_second': float(row['blinks_per_second'])
            })
    return sessions


def generate_comparison_chart() -> None:
    """
    Generate a comparison chart of blink rates between movie and reading modes.
    """
    sessions = load_sessions()
    
    if not sessions:
        print("⚠ No session data found. Run some sessions first!")
        return
    
    # Separate by mode
    movie_rates = [s['blinks_per_second'] for s in sessions if s['mode'] == 'movie']
    reading_rates = [s['blinks_per_second'] for s in sessions if s['mode'] == 'reading']
    
    if not movie_rates and not reading_rates:
        print("⚠ No valid session data found.")
        return
    
    # Calculate statistics
    movie_avg = np.mean(movie_rates) if movie_rates else 0
    movie_std = np.std(movie_rates) if len(movie_rates) > 1 else 0
    reading_avg = np.mean(reading_rates) if reading_rates else 0
    reading_std = np.std(reading_rates) if len(reading_rates) > 1 else 0
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart comparing averages
    modes = ['Movie\nWatching', 'Document\nReading']
    averages = [movie_avg, reading_avg]
    stds = [movie_std, reading_std]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(modes, averages, yerr=stds, capsize=5, color=colors, edgecolor='black')
    ax1.set_ylabel('Blinks per Second', fontsize=12)
    ax1.set_title('Average Eye Blink Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(averages) * 1.3 if max(averages) > 0 else 0.5)
    
    # Add value labels on bars
    for bar, avg, std in zip(bars, averages, stds):
        height = bar.get_height()
        ax1.annotate(f'{avg:.1f}±{std:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Session count annotations
    ax1.annotate(f'n={len(movie_rates)}', xy=(0, -0.15), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=10, color='gray')
    ax1.annotate(f'n={len(reading_rates)}', xy=(1, -0.15), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=10, color='gray')
    
    # Scatter plot of all sessions
    if movie_rates:
        ax2.scatter([1] * len(movie_rates), movie_rates, c='#3498db', s=100, 
                   alpha=0.6, label=f'Movie (n={len(movie_rates)})', edgecolors='black')
    if reading_rates:
        ax2.scatter([2] * len(reading_rates), reading_rates, c='#e74c3c', s=100,
                   alpha=0.6, label=f'Reading (n={len(reading_rates)})', edgecolors='black')
    
    ax2.set_xlim(0.5, 2.5)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Movie', 'Reading'])
    ax2.set_ylabel('Blinks per Second', fontsize=12)
    ax2.set_title('Individual Session Results', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(OUTPUT_DIR, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 50)
    print("EYE BLINK RATE COMPARISON RESULTS")
    print("=" * 50)
    print(f"\nMovie Watching Sessions: {len(movie_rates)}")
    if movie_rates:
        print(f"  Average: {movie_avg:.3f} blinks/sec (±{movie_std:.3f})")
        print(f"  Range: {min(movie_rates):.3f} - {max(movie_rates):.3f}")
    
    print(f"\nDocument Reading Sessions: {len(reading_rates)}")
    if reading_rates:
        print(f"  Average: {reading_avg:.3f} blinks/sec (±{reading_std:.3f})")
        print(f"  Range: {min(reading_rates):.3f} - {max(reading_rates):.3f}")
    
    if movie_rates and reading_rates:
        diff = reading_avg - movie_avg
        if diff > 0:
            print(f"\n→ Reading shows {diff:.3f} MORE blinks/sec than movie watching")
        else:
            print(f"\n→ Movie watching shows {abs(diff):.3f} MORE blinks/sec than reading")
    
    print(f"\n✓ Chart saved to {chart_path}")
    print("=" * 50)


def run_detection_session(mode: str, duration: int, camera_index: int) -> None:
    """
    Run a blink detection session.
    
    Args:
        mode: Session mode ('movie' or 'reading')
        duration: Session duration in seconds
        camera_index: Camera index to use
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Find available cameras
    available_cameras = find_available_cameras()
    if not available_cameras:
        print("⚠ No cameras found!")
        return
    
    if camera_index not in available_cameras:
        camera_index = available_cameras[0]
        print(f"⚠ Requested camera not available. Using camera {camera_index}")
    
    current_cam_idx = available_cameras.index(camera_index)
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"⚠ Could not open camera {camera_index}")
        return
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n" + "=" * 50)
    print(f"EYE BLINK DETECTION - {mode.upper()} MODE")
    print("=" * 50)
    print(f"Duration: {duration} seconds")
    print(f"Camera: {camera_index} ({frame_width}x{frame_height})")
    print("\nControls:")
    print("  q - Quit session")
    print("  c - Change camera")
    print("  r - Reset blink counter")
    print("=" * 50)
    print("\nStarting session... Look at the camera!")
    
    # Session variables
    blink_count = 0
    frame_counter = 0  # Frames below threshold
    start_time = cv2.getTickCount()
    ear_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Failed to read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Calculate elapsed time
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        remaining = max(0, duration - elapsed)
        
        # Check if session ended
        if elapsed >= duration:
            print("\n✓ Session completed!")
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        ear = 0.0
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get eye landmarks
            left_eye = get_eye_landmarks(face_landmarks, LEFT_EYE_INDICES, 
                                        frame_width, frame_height)
            right_eye = get_eye_landmarks(face_landmarks, RIGHT_EYE_INDICES,
                                         frame_width, frame_height)
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Store EAR history for visualization
            ear_history.append(ear)
            if len(ear_history) > 100:
                ear_history.pop(0)
            
            # Blink detection
            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                frame_counter = 0
            
            # Draw eye landmarks
            draw_eye_landmarks(frame, left_eye, (0, 255, 0))
            draw_eye_landmarks(frame, right_eye, (0, 255, 0))
        
        # Calculate current blink rate
        current_rate = (blink_count / elapsed) if elapsed > 0 else 0
        
        # Create info overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Mode indicator with color
        mode_color = (219, 152, 52) if mode == 'movie' else (60, 76, 231)  # Blue for movie, Red for reading
        cv2.putText(frame, f"Mode: {mode.upper()}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Timer
        mins = int(remaining // 60)
        secs = int(remaining % 60)
        cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blink count
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Blink rate
        cv2.putText(frame, f"Rate: {current_rate:.2f} blinks/sec", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # EAR value with color indicator
        ear_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        
        # Face detection status
        if not face_detected:
            cv2.putText(frame, "No face detected!", (frame_width//2 - 100, frame_height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Progress bar
        progress = elapsed / duration
        bar_width = 300
        bar_height = 10
        bar_x = (frame_width - bar_width) // 2
        bar_y = frame_height - 50
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), mode_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Eye Blink Detector', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⚠ Session ended early by user")
            break
        elif key == ord('c') and len(available_cameras) > 1:
            # Cycle to next camera
            current_cam_idx = (current_cam_idx + 1) % len(available_cameras)
            camera_index = available_cameras[current_cam_idx]
            cap.release()
            cap = cv2.VideoCapture(camera_index)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"→ Switched to camera {camera_index}")
        elif key == ord('r'):
            blink_count = 0
            print("→ Blink counter reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    
    # Calculate final stats
    actual_duration = min(elapsed, duration)
    final_rate = (blink_count / actual_duration) if actual_duration > 0 else 0
    
    # Print results
    print("\n" + "=" * 50)
    print("SESSION RESULTS")
    print("=" * 50)
    print(f"Mode: {mode}")
    print(f"Duration: {actual_duration:.1f} seconds")
    print(f"Total Blinks: {blink_count}")
    print(f"Blink Rate: {final_rate:.3f} blinks/second")
    print("=" * 50)
    
    # Save session
    save_session(mode, actual_duration, blink_count)


def main():
    parser = argparse.ArgumentParser(
        description='Eye Blink Rate Detector - Compare blink rates during movie watching vs reading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python blink_detector.py --mode movie         Run a movie watching session
  python blink_detector.py --mode reading       Run a reading session
  python blink_detector.py --mode movie -d 30   Run a 30-second session
  python blink_detector.py --compare            Compare all session results
  python blink_detector.py --camera 1           Use camera index 1
        """
    )
    
    parser.add_argument('-m', '--mode', type=str, choices=['movie', 'reading'],
                        help='Session mode: "movie" for watching content, "reading" for reading documents')
    parser.add_argument('-d', '--duration', type=int, default=60,
                        help='Session duration in seconds (default: 60)')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Camera index to use (default: 0)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison chart from saved sessions')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List available cameras and exit')
    
    args = parser.parse_args()
    
    # List cameras
    if args.list_cameras:
        cameras = find_available_cameras()
        if cameras:
            print(f"Available cameras: {cameras}")
        else:
            print("No cameras found")
        return
    
    # Generate comparison
    if args.compare:
        generate_comparison_chart()
        return
    
    # Validate mode
    if not args.mode:
        parser.print_help()
        print("\n⚠ Please specify a mode (--mode movie or --mode reading) or use --compare")
        return
    
    # Run detection session
    run_detection_session(args.mode, args.duration, args.camera)


if __name__ == "__main__":
    main()
