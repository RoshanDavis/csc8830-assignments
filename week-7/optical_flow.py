"""
Part A: Optical Flow Computation & Visualization

Computes dense optical flow (Farnebäck) on two user-supplied videos,
visualizes the flow as color-coded videos, and explains what information
can be inferred from optical flow.

Usage:
    python optical_flow.py -v1 video1.mp4 -v2 video2.mp4
    python optical_flow.py -v1 video1.mp4 -v2 video2.mp4 --duration 30
"""

import os
import argparse

import cv2
import numpy as np
import imageio


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def compute_flow_visualization(flow: np.ndarray, frame_gray: np.ndarray) -> np.ndarray:
    """
    Convert a dense optical flow field to an HSV color visualization.

    Encoding:
      - Hue   = direction of motion (angle of the flow vector)
      - Saturation = 255 (constant)
      - Value = magnitude of motion (clamped for visibility)

    Args:
        flow: Dense flow array of shape (H, W, 2) from calcOpticalFlowFarneback.
        frame_gray: Grayscale frame used as background brightness (unused in
                     pure-HSV mode but kept for interface consistency).

    Returns:
        BGR image with flow visualized as color.
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((*frame_gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2        # Hue: 0-180
    hsv[..., 1] = 255                              # Full saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def process_video(video_path: str, output_path: str, duration: float) -> dict:
    """
    Compute and visualize dense optical flow for a single video.

    Args:
        video_path: Path to the input video.
        output_path: Path for the output flow-visualization video.
        duration: Maximum number of seconds to process.

    Returns:
        Dictionary with statistics about the processed video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * duration)

    print(f"\n  Video:      {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps:.1f}")
    print(f"  Processing: up to {duration}s ({max_frames} frames)")

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    frame_count = 1
    flow_magnitudes = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Farnebäck dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flow_vis = compute_flow_visualization(flow, gray)

        # Side-by-side: original | flow visualization
        combined = np.hstack([frame, flow_vis])
        rgb_combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_combined)

        # Collect magnitude stats
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_magnitudes.append(float(np.mean(mag)))

        prev_gray = gray.copy()

    cap.release()
    writer.close()

    stats = {
        "frames_processed": frame_count,
        "duration_s": frame_count / fps,
        "mean_flow_magnitude": float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0,
        "max_flow_magnitude": float(np.max(flow_magnitudes)) if flow_magnitudes else 0.0,
    }

    return stats


def print_flow_explanation():
    """Print an explanation of what optical flow reveals."""
    print("""
==================================================
OPTICAL FLOW — WHAT CAN BE INFERRED?
==================================================

Optical flow computes a 2D motion vector (u, v) for every pixel between
two consecutive frames.  The resulting vector field encodes:

1. DIRECTION OF MOTION
   - The angle of each vector shows where each pixel is moving.
   - In the HSV visualization, hue encodes direction: red = right,
     cyan = left, green = down, magenta = up.

2. SPEED / MAGNITUDE OF MOTION
   - The length (magnitude) of each vector shows how fast a pixel moves.
   - In the visualization, brighter colors = faster motion; dark = static.

3. MOVING VS. STATIC REGIONS
   - Regions with near-zero flow are stationary (background).
   - Regions with large flow vectors correspond to moving objects.

4. OBJECT BOUNDARIES
   - Discontinuities in the flow field often align with object boundaries,
     since an object and its background move differently.

5. CAMERA MOTION
   - If the camera pans or zooms, the entire flow field has a coherent
     global pattern (e.g., uniform rightward flow during a pan).
   - This can be separated from independent object motion.

EVIDENCE IN THE OUTPUT VIDEOS:
   - Static background appears dark (near-zero flow).
   - Moving objects (people, cars, etc.) appear as brightly colored
     regions whose hue shows their direction of travel.
   - Fast-moving objects appear brighter than slow-moving ones.

==================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="Part A: Compute and visualize dense optical flow on two videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python optical_flow.py -v1 video1.mp4 -v2 video2.mp4
    python optical_flow.py -v1 video1.mp4 -v2 video2.mp4 --duration 45
        """
    )

    parser.add_argument("-v1", "--video1", type=str, required=True,
                        help="Path to the first input video")
    parser.add_argument("-v2", "--video2", type=str, required=True,
                        help="Path to the second input video")
    parser.add_argument("-d", "--duration", type=float, default=30.0,
                        help="Seconds of video to process (default: 30)")

    args = parser.parse_args()

    # Resolve paths relative to script directory
    v1 = args.video1 if os.path.isabs(args.video1) else os.path.join(SCRIPT_DIR, args.video1)
    v2 = args.video2 if os.path.isabs(args.video2) else os.path.join(SCRIPT_DIR, args.video2)

    for path in [v1, v2]:
        if not os.path.exists(path):
            print(f"Error: Video not found: {path}")
            return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out1 = os.path.join(OUTPUT_DIR, "flow_video1.mp4")
    out2 = os.path.join(OUTPUT_DIR, "flow_video2.mp4")

    print("=" * 50)
    print("PART A: DENSE OPTICAL FLOW COMPUTATION")
    print("=" * 50)

    # Process video 1
    print("\n--- Video 1 ---")
    stats1 = process_video(v1, out1, args.duration)
    print(f"  Frames processed:     {stats1['frames_processed']}")
    print(f"  Duration processed:   {stats1['duration_s']:.1f}s")
    print(f"  Mean flow magnitude:  {stats1['mean_flow_magnitude']:.3f} px/frame")
    print(f"  Max flow magnitude:   {stats1['max_flow_magnitude']:.3f} px/frame")
    print(f"  Output saved:         {out1}")

    # Process video 2
    print("\n--- Video 2 ---")
    stats2 = process_video(v2, out2, args.duration)
    print(f"  Frames processed:     {stats2['frames_processed']}")
    print(f"  Duration processed:   {stats2['duration_s']:.1f}s")
    print(f"  Mean flow magnitude:  {stats2['mean_flow_magnitude']:.3f} px/frame")
    print(f"  Max flow magnitude:   {stats2['max_flow_magnitude']:.3f} px/frame")
    print(f"  Output saved:         {out2}")

    # Print explanation
    print_flow_explanation()

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
