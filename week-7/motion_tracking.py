"""
Part B: Motion Tracking Equations & Bilinear Interpolation

Derives motion tracking equations from fundamentals, explains bilinear
interpolation, and validates tracking predictions against actual pixel
locations on consecutive frame pairs from two videos.

Usage:
    python motion_tracking.py -v1 video1.mp4 -v2 video2.mp4
    python motion_tracking.py -v1 video1.mp4 -v2 video2.mp4 --frame 60
"""

import os
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# ─────────────────────────────────────────────────────────────────────
# Theoretical Derivations (printed to console)
# ─────────────────────────────────────────────────────────────────────

def print_tracking_derivation():
    """Print the derivation of motion tracking equations from fundamentals."""
    print("""
================================================================
DERIVATION OF MOTION TRACKING EQUATIONS FROM FUNDAMENTALS
================================================================

1. BRIGHTNESS CONSTANCY ASSUMPTION
   --------------------------------
   Assume that the intensity of a moving pixel does not change
   between frames:

       I(x, y, t) = I(x + dx, y + dy, t + dt)              ... (1)

   where (dx, dy) is the displacement of the pixel in time dt.

2. TAYLOR SERIES EXPANSION
   -------------------------
   Expand the right side of (1) using a first-order Taylor series:

       I(x+dx, y+dy, t+dt) ≈ I(x,y,t) + I_x·dx + I_y·dy + I_t·dt

   where I_x = ∂I/∂x, I_y = ∂I/∂y, I_t = ∂I/∂t are the partial
   derivatives of image intensity.

   Substituting into (1) and cancelling I(x,y,t):

       I_x·dx + I_y·dy + I_t·dt = 0

   Dividing by dt:

       I_x·u + I_y·v + I_t = 0                               ... (2)

   where u = dx/dt and v = dy/dt are the optical flow components.

   Equation (2) is the OPTICAL FLOW CONSTRAINT EQUATION.

3. THE APERTURE PROBLEM
   ----------------------
   Equation (2) is a single equation with two unknowns (u, v).
   This is under-determined — we cannot solve for both u and v
   from one pixel alone.

4. LUCAS-KANADE METHOD (LOCAL WINDOW)
   ------------------------------------
   Assume that all pixels within a small window W (e.g., 15×15)
   around pixel (x, y) share the same flow vector (u, v).

   For each pixel (x_i, y_i) in the window, we write:

       I_x(x_i, y_i)·u + I_y(x_i, y_i)·v = −I_t(x_i, y_i)

   Stacking n pixels in the window into a matrix system:

       A · [u, v]^T = b

   where:
       A = [ I_x(x_1, y_1)   I_y(x_1, y_1) ]     b = [ −I_t(x_1, y_1) ]
           [ I_x(x_2, y_2)   I_y(x_2, y_2) ]         [ −I_t(x_2, y_2) ]
           [      ...              ...       ]         [       ...       ]
           [ I_x(x_n, y_n)   I_y(x_n, y_n) ]         [ −I_t(x_n, y_n) ]

   This is an overdetermined system (n >> 2), solved via least squares:

       [u, v]^T = (A^T A)^{-1} A^T b                          ... (3)

   This is the LUCAS-KANADE OPTICAL FLOW SOLUTION.

5. SETTING UP TRACKING FOR TWO FRAMES
   -------------------------------------
   Given frame I_1 at time t and frame I_2 at time t+1:

   Step 1: Compute spatial gradients I_x, I_y using Sobel operators
           on I_1 (or the average of I_1 and I_2).
   Step 2: Compute temporal gradient I_t = I_2 − I_1.
   Step 3: For each feature point, gather I_x, I_y, I_t over the
           local window to form A and b.
   Step 4: Solve for (u, v) via the normal equations (3).
   Step 5: The predicted position of a feature at (x, y) in frame 2
           is (x + u, y + v).

================================================================
""")


def print_bilinear_interpolation():
    """Print the derivation of bilinear interpolation."""
    print("""
================================================================
BILINEAR INTERPOLATION — DERIVATION
================================================================

When the predicted position (x + u, y + v) falls at a sub-pixel
location, we need to interpolate the image intensity.

Given a point (x, y) where x and y are non-integer, let:

    x_0 = floor(x),  x_1 = x_0 + 1
    y_0 = floor(y),  y_1 = y_0 + 1

    a = x − x_0      (fractional part in x)
    b = y − y_0      (fractional part in y)

The four surrounding pixel intensities are:

    Q_00 = I(x_0, y_0)    Q_10 = I(x_1, y_0)
    Q_01 = I(x_0, y_1)    Q_11 = I(x_1, y_1)

Step 1 — Interpolate along x (two 1D linear interpolations):

    R_0 = (1−a)·Q_00 + a·Q_10     (top row)
    R_1 = (1−a)·Q_01 + a·Q_11     (bottom row)

Step 2 — Interpolate along y:

    I(x, y) = (1−b)·R_0 + b·R_1

Expanding:

    I(x, y) = (1−a)(1−b)·Q_00  +  a(1−b)·Q_10
            + (1−a)·b·Q_01     +  a·b·Q_11

This is a weighted average of the four neighbors, where the
weights are the areas of the opposite rectangles.

WHY IT MATTERS FOR TRACKING:
   Optical flow estimates sub-pixel displacements. To compare
   predicted and actual intensities at sub-pixel locations (for
   accuracy validation or iterative refinement), we use bilinear
   interpolation to "read" the image at non-integer coordinates.

================================================================
""")


# ─────────────────────────────────────────────────────────────────────
# Practical Tracking Validation
# ─────────────────────────────────────────────────────────────────────

def extract_frame_pair(video_path: str, frame_index: int):
    """
    Extract two consecutive frames from a video.

    Args:
        video_path: Path to the video file.
        frame_index: Index of the first frame (0-based).

    Returns:
        Tuple of (frame1, frame2) as BGR numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        raise ValueError(
            f"Could not read frames {frame_index} and {frame_index+1} from {video_path}"
        )

    return frame1, frame2


def validate_tracking(frame1, frame2, video_label: str, max_features: int = 20):
    """
    Detect features in frame1, predict their positions in frame2 using
    Lucas-Kanade optical flow, and compare with actual positions found
    via template matching.

    Args:
        frame1: First frame (BGR).
        frame2: Second frame (BGR).
        video_label: Label for display (e.g., "Video 1").
        max_features: Maximum number of features to track.

    Returns:
        Tuple of (annotated_image, results_table).
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect good features to track (Shi-Tomasi corners)
    features = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=30,
        blockSize=7
    )

    if features is None or len(features) == 0:
        print(f"  Warning: No features detected in {video_label}")
        return None, []

    # Lucas-Kanade optical flow — predict positions in frame2
    predicted, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, features, None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # Filter by status (flow successfully found)
    good_mask = status.flatten() == 1
    orig_pts = features[good_mask]
    pred_pts = predicted[good_mask]

    # Validate predictions using template matching
    patch_size = 21
    half = patch_size // 2
    search_radius = 15
    h, w = gray2.shape

    results = []
    for i in range(len(orig_pts)):
        ox, oy = orig_pts[i].ravel()
        px, py = pred_pts[i].ravel()

        # Extract patch from frame1 around original feature
        y_start = max(0, int(oy) - half)
        y_end = min(h, int(oy) + half + 1)
        x_start = max(0, int(ox) - half)
        x_end = min(w, int(ox) + half + 1)
        template = gray1[y_start:y_end, x_start:x_end]

        if template.shape[0] < 5 or template.shape[1] < 5:
            continue

        # Search region in frame2 centered on predicted position
        sy_start = max(0, int(py) - search_radius - half)
        sy_end = min(h, int(py) + search_radius + half + 1)
        sx_start = max(0, int(px) - search_radius - half)
        sx_end = min(w, int(px) + search_radius + half + 1)
        search_region = gray2[sy_start:sy_end, sx_start:sx_end]

        if (search_region.shape[0] < template.shape[0] or
                search_region.shape[1] < template.shape[1]):
            continue

        # Template matching to find actual location
        match = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(match)

        actual_x = sx_start + max_loc[0] + half
        actual_y = sy_start + max_loc[1] + half

        error = np.sqrt((px - actual_x) ** 2 + (py - actual_y) ** 2)

        results.append({
            "index": i,
            "original": (float(ox), float(oy)),
            "predicted": (float(px), float(py)),
            "actual": (float(actual_x), float(actual_y)),
            "error_px": float(error),
        })

    # Create annotated visualization
    vis = frame2.copy()
    for r in results:
        px, py = int(r["predicted"][0]), int(r["predicted"][1])
        ax, ay = int(r["actual"][0]), int(r["actual"][1])
        ox, oy = int(r["original"][0]), int(r["original"][1])

        # Green circle: original position
        cv2.circle(vis, (ox, oy), 5, (0, 255, 0), -1)
        # Blue circle: predicted position
        cv2.circle(vis, (px, py), 5, (255, 0, 0), -1)
        # Red circle: actual position (template match)
        cv2.circle(vis, (ax, ay), 5, (0, 0, 255), -1)
        # Line from original to predicted
        cv2.arrowedLine(vis, (ox, oy), (px, py), (255, 200, 0), 1, tipLength=0.3)

    # Add legend
    cv2.putText(vis, "Green=Original  Blue=Predicted  Red=Actual",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis, results


def print_results_table(results: list, label: str):
    """Print a formatted table of tracking results."""
    if not results:
        print(f"\n  No tracking results for {label}.")
        return

    print(f"\n{'='*72}")
    print(f"  TRACKING VALIDATION — {label}")
    print(f"{'='*72}")
    print(f"  {'#':>3}  {'Original (x,y)':>16}  {'Predicted (x,y)':>16}  "
          f"{'Actual (x,y)':>16}  {'Error(px)':>9}")
    print(f"  {'-'*3}  {'-'*16}  {'-'*16}  {'-'*16}  {'-'*9}")

    errors = []
    for r in results:
        ox, oy = r["original"]
        px, py = r["predicted"]
        ax, ay = r["actual"]
        err = r["error_px"]
        errors.append(err)
        print(f"  {r['index']:3d}  ({ox:7.1f},{oy:6.1f})  "
              f"({px:7.1f},{py:6.1f})  ({ax:7.1f},{ay:6.1f})  {err:9.2f}")

    print(f"  {'-'*66}")
    print(f"  Mean error:  {np.mean(errors):.2f} px")
    print(f"  Max error:   {np.max(errors):.2f} px")
    print(f"  Min error:   {np.min(errors):.2f} px")
    print(f"{'='*72}")


def save_annotated_image(image, filename: str):
    """Save an annotated image to the output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, image)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Part B: Motion tracking derivation & validation on consecutive frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python motion_tracking.py -v1 video1.mp4 -v2 video2.mp4
    python motion_tracking.py -v1 video1.mp4 -v2 video2.mp4 --frame 100
        """
    )

    parser.add_argument("-v1", "--video1", type=str, required=True,
                        help="Path to the first input video")
    parser.add_argument("-v2", "--video2", type=str, required=True,
                        help="Path to the second input video")
    parser.add_argument("-f", "--frame", type=int, default=60,
                        help="Frame index to pick the consecutive pair from (default: 60)")

    args = parser.parse_args()

    # Resolve paths
    v1 = args.video1 if os.path.isabs(args.video1) else os.path.join(SCRIPT_DIR, args.video1)
    v2 = args.video2 if os.path.isabs(args.video2) else os.path.join(SCRIPT_DIR, args.video2)

    for path in [v1, v2]:
        if not os.path.exists(path):
            print(f"Error: Video not found: {path}")
            return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Print theoretical derivations ──
    print_tracking_derivation()
    print_bilinear_interpolation()

    # ── Step 2: Validate tracking on frame pairs ──
    print("=" * 50)
    print("PRACTICAL TRACKING VALIDATION")
    print("=" * 50)

    for idx, (vpath, label) in enumerate([(v1, "Video 1"), (v2, "Video 2")], start=1):
        print(f"\n--- {label}: {os.path.basename(vpath)} ---")
        print(f"  Extracting frames {args.frame} and {args.frame + 1}...")

        frame1, frame2 = extract_frame_pair(vpath, args.frame)
        vis, results = validate_tracking(frame1, frame2, label)

        if vis is not None:
            save_annotated_image(vis, f"tracking_video{idx}.png")
        print_results_table(results, label)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
