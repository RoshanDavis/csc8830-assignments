"""
Uncalibrated Stereo Vision Pipeline
====================================
Estimates object distance using two images from an uncalibrated stereo setup.
Computes: Fundamental Matrix (F), Essential Matrix (E), Rotation (R), Translation (t).
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import argparse
import os
import sys
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these for your setup)
# ─────────────────────────────────────────────
BASELINE_MM = 30          # Distance between the two camera centres (mm)
FOCAL_LENGTH_PX = None       # None → estimated from image width (good first guess)
RANSAC_THRESHOLD = 1.0       # px — RANSAC reprojection threshold for F
MIN_MATCHES = 20             # Minimum good feature matches required


# ─────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────

def load_image(path: str, color: bool = True) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def estimate_focal_length(image_width: int) -> float:
    """Heuristic: focal length ≈ image width (reasonable for most cameras)."""
    return float(image_width)


def build_camera_matrix(focal_px: float, w: int, h: int) -> np.ndarray:
    """Build a plausible intrinsic matrix K."""
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[focal_px, 0,        cx],
                  [0,        focal_px, cy],
                  [0,        0,         1]], dtype=np.float64)
    return K


# ─────────────────────────────────────────────
#  FEATURE DETECTION & MATCHING
# ─────────────────────────────────────────────

def detect_and_match(img_l: np.ndarray, img_r: np.ndarray):
    """SIFT feature detection with Lowe ratio test."""
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des_l, des_r, k=2)

    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    if len(good) < MIN_MATCHES:
        raise RuntimeError(
            f"Only {len(good)} good matches found (need ≥ {MIN_MATCHES}). "
            "Try images with more texture or reduce MIN_MATCHES."
        )

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])

    return kp_l, kp_r, good, pts_l, pts_r


# ─────────────────────────────────────────────
#  FUNDAMENTAL & ESSENTIAL MATRIX
# ─────────────────────────────────────────────

def compute_fundamental_matrix(pts_l, pts_r):
    """Compute F via 8-point algorithm with RANSAC."""
    F, mask = cv2.findFundamentalMat(
        pts_l, pts_r,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        confidence=0.999
    )
    if F is None or F.shape != (3, 3):
        raise RuntimeError("Fundamental matrix estimation failed.")
    mask = mask.ravel().astype(bool)
    inliers_l = pts_l[mask]
    inliers_r = pts_r[mask]
    return F, inliers_l, inliers_r, mask


def compute_essential_matrix(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """E = K^T · F · K"""
    E = K.T @ F @ K
    # Enforce rank-2 constraint via SVD
    U, S, Vt = np.linalg.svd(E)
    S_enforced = np.diag([1.0, 1.0, 0.0])
    E = U @ S_enforced @ Vt
    return E


def decompose_essential_matrix(E: np.ndarray, K: np.ndarray,
                                pts_l: np.ndarray, pts_r: np.ndarray):
    """Recover R and t from E; choose the physically valid solution."""
    _, R, t, pose_mask = cv2.recoverPose(E, pts_l, pts_r, K)
    return R, t, pose_mask


# ─────────────────────────────────────────────
#  TRIANGULATION & DISTANCE ESTIMATION
# ─────────────────────────────────────────────

def triangulate_points(R: np.ndarray, t: np.ndarray,
                        K: np.ndarray,
                        pts_l: np.ndarray, pts_r: np.ndarray) -> np.ndarray:
    """Triangulate matched points to get 3-D coordinates."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts4d = cv2.triangulatePoints(P1, P2, pts_l.T, pts_r.T)
    pts3d = (pts4d[:3] / pts4d[3]).T          # (N, 3)
    return pts3d


def estimate_object_distance(pts3d: np.ndarray,
                              baseline_mm: float,
                              t_norm: float) -> float:
    """
    Scale the metric depth using the known baseline.
    t from recoverPose is a unit vector; we scale by baseline / ||t||_recovered.
    """
    if t_norm < 1e-8:
        raise RuntimeError("Translation vector has near-zero norm.")

    scale = baseline_mm / t_norm                  # mm / arbitrary_unit
    depths = pts3d[:, 2]                          # Z in camera coords
    median_depth_au = float(np.median(depths[depths > 0]))
    distance_mm = median_depth_au * scale
    return distance_mm


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def draw_epipolar_lines(img_l, img_r, pts_l, pts_r, F, n=15):
    """Draw n epipolar lines on both images."""
    h, w = img_l.shape[:2]
    idx = np.random.choice(len(pts_l), min(n, len(pts_l)), replace=False)
    sel_l, sel_r = pts_l[idx], pts_r[idx]

    vis_l = img_l.copy()
    vis_r = img_r.copy()
    colors = [tuple(np.random.randint(50, 255, 3).tolist()) for _ in range(n)]

    lines_r = cv2.computeCorrespondEpilines(sel_l.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines_l = cv2.computeCorrespondEpilines(sel_r.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    for i, (line_r, line_l, c) in enumerate(zip(lines_r, lines_l, colors)):
        x0_r, y0_r = 0, int(-line_r[2] / line_r[1])
        x1_r, y1_r = w, int(-(line_r[2] + line_r[0] * w) / line_r[1])
        cv2.line(vis_r, (x0_r, y0_r), (x1_r, y1_r), c, 1)
        cv2.circle(vis_r, tuple(sel_r[i].astype(int)), 5, c, -1)

        x0_l, y0_l = 0, int(-line_l[2] / line_l[1])
        x1_l, y1_l = w, int(-(line_l[2] + line_l[0] * w) / line_l[1])
        cv2.line(vis_l, (x0_l, y0_l), (x1_l, y1_l), c, 1)
        cv2.circle(vis_l, tuple(sel_l[i].astype(int)), 5, c, -1)

    return cv2.cvtColor(vis_l, cv2.COLOR_BGR2RGB), cv2.cvtColor(vis_r, cv2.COLOR_BGR2RGB)


def save_report_figure(img_l_rgb, img_r_rgb,
                        epi_l, epi_r,
                        pts3d, distance_mm, gt_mm,
                        R, E, F,
                        inliers_l, inliers_r,
                        out_path="stereo_report.png"):
    """Compose the full visual report."""

    fig = plt.figure(figsize=(22, 28), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    text_kw = dict(color="white", fontsize=10)
    title_kw = dict(color="#58a6ff", fontsize=12, fontweight="bold")

    gs = GridSpec(5, 2, figure=fig,
                  hspace=0.45, wspace=0.3,
                  top=0.96, bottom=0.03,
                  left=0.06, right=0.97)

    # ── Row 0: original stereo pair ──────────────────────────────────────────
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_l.imshow(img_l_rgb); ax_l.set_title("Left Image", **title_kw); ax_l.axis("off")
    ax_r.imshow(img_r_rgb); ax_r.set_title("Right Image", **title_kw); ax_r.axis("off")

    # ── Row 1: epipolar lines ────────────────────────────────────────────────
    ax_el = fig.add_subplot(gs[1, 0])
    ax_er = fig.add_subplot(gs[1, 1])
    ax_el.imshow(epi_l); ax_el.set_title("Epipolar Lines — Left", **title_kw); ax_el.axis("off")
    ax_er.imshow(epi_r); ax_er.set_title("Epipolar Lines — Right", **title_kw); ax_er.axis("off")

    # ── Row 2: 3-D point cloud (top & side) ─────────────────────────────────
    ax3d_top = fig.add_subplot(gs[2, 0])
    ax3d_side = fig.add_subplot(gs[2, 1])

    valid = pts3d[:, 2] > 0
    xs, ys, zs = pts3d[valid, 0], pts3d[valid, 1], pts3d[valid, 2]
    sc1 = ax3d_top.scatter(xs, zs, c=zs, cmap="plasma", s=8, alpha=0.7)
    ax3d_top.set_xlabel("X (a.u.)", color="white", fontsize=8)
    ax3d_top.set_ylabel("Z / Depth (a.u.)", color="white", fontsize=8)
    ax3d_top.set_title("3-D Point Cloud — Top View (X-Z)", **title_kw)
    ax3d_top.set_facecolor("#161b22"); ax3d_top.tick_params(colors="white")
    plt.colorbar(sc1, ax=ax3d_top, label="Depth")

    sc2 = ax3d_side.scatter(zs, -ys, c=zs, cmap="plasma", s=8, alpha=0.7)
    ax3d_side.set_xlabel("Z / Depth (a.u.)", color="white", fontsize=8)
    ax3d_side.set_ylabel("Y (a.u.)", color="white", fontsize=8)
    ax3d_side.set_title("3-D Point Cloud — Side View (Z-Y)", **title_kw)
    ax3d_side.set_facecolor("#161b22"); ax3d_side.tick_params(colors="white")

    # ── Row 3: matrix displays ───────────────────────────────────────────────
    def matrix_text(ax, M, title, fmt=".4f"):
        ax.set_facecolor("#161b22")
        ax.set_title(title, **title_kw)
        ax.axis("off")
        rows = M.shape[0]
        lines = ["["]
        for i in range(rows):
            row_str = "  " + "  ".join(f"{v:{fmt}}" for v in M[i])
            lines.append(row_str + (" ," if i < rows - 1 else ""))
        lines.append("]")
        txt = "\n".join(lines)
        ax.text(0.5, 0.5, txt, ha="center", va="center",
                fontfamily="monospace", color="#e6edf3",
                fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", fc="#21262d", ec="#30363d"))

    ax_F = fig.add_subplot(gs[3, 0])
    ax_E = fig.add_subplot(gs[3, 1])
    matrix_text(ax_F, F, "Fundamental Matrix  F  (3×3)")
    matrix_text(ax_E, E, "Essential Matrix  E  (3×3)")

    ax_R = fig.add_subplot(gs[4, 0])
    matrix_text(ax_R, R, "Rotation Matrix  R  (3×3)")

    # ── Row 4 right: distance summary ────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[4, 1])
    ax_dist.set_facecolor("#161b22"); ax_dist.axis("off")
    ax_dist.set_title("Distance Estimation", **title_kw)

    est_m = distance_mm / 1000.0
    gt_m  = gt_mm / 1000.0 if gt_mm else None
    err_pct = abs(est_m - gt_m) / gt_m * 100 if gt_m else None

    summary = (
        f"Estimated Distance : {est_m:.3f} m  ({distance_mm:.1f} mm)\n"
        + (f"Ground Truth       : {gt_m:.3f} m  ({gt_mm:.1f} mm)\n" if gt_m else "")
        + (f"Absolute Error     : {err_pct:.2f} %\n" if err_pct else "")
        + f"\n# Inlier Matches   : {len(inliers_l)}"
    )
    ax_dist.text(0.5, 0.5, summary, ha="center", va="center",
                 fontfamily="monospace", color="#3fb950",
                 fontsize=11, transform=ax_dist.transAxes,
                 bbox=dict(boxstyle="round,pad=0.6", fc="#0f2117", ec="#3fb950", lw=1.5))

    # ── Master title ─────────────────────────────────────────────────────────
    fig.text(0.5, 0.985,
             "Uncalibrated Stereo Vision — Distance Estimation Report",
             ha="center", va="top",
             color="white", fontsize=16, fontweight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Report figure saved → {out_path}")


def save_annotated_setup(img_l_rgb, distance_mm, gt_mm, out_path="annotated_setup.png"):
    """Save left image annotated with estimated & GT distance."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0d1117")
    ax.imshow(img_l_rgb)
    ax.axis("off")

    h, w = img_l_rgb.shape[:2]
    # Draw a bracket in the centre to denote the object
    cx, cy = w // 2, h // 2
    bw, bh = w // 6, h // 5
    rect = patches.FancyBboxPatch(
        (cx - bw // 2, cy - bh // 2), bw, bh,
        boxstyle="round,pad=4", linewidth=2.5,
        edgecolor="#ff6b6b", facecolor="none"
    )
    ax.add_patch(rect)

    label = f"Estimated: {distance_mm/1000:.3f} m"
    if gt_mm:
        label += f"\nGround Truth: {gt_mm/1000:.3f} m"
    ax.text(cx - bw // 2, cy - bh // 2 - 14, label,
            color="white", fontsize=12, fontweight="bold",
            bbox=dict(fc="#ff6b6b", ec="none", pad=3))

    ax.set_title("Annotated Left Image — Selected Object",
                 color="#58a6ff", fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Annotated image saved → {out_path}")


# ─────────────────────────────────────────────
#  DEMO IMAGE GENERATOR (synthetic test scene)
# ─────────────────────────────────────────────

def generate_demo_images(out_dir: str = "."):
    """
    Generate a simple synthetic stereo pair so the pipeline can be
    tested without a real camera. A textured indoor scene is rendered
    with a known disparity shift.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    H, W = 480, 640
    DISPARITY_PX = 16          # pixels — corresponds to a 4.0 m depth at f=640

    def make_scene(shift_x=0):
        img = np.ones((H, W, 3), np.uint8) * 200      # light grey wall
        # Add noise texture
        noise = np.random.randint(0, 30, (H, W, 3), dtype=np.uint8)
        img = np.clip(img.astype(int) + noise - 15, 0, 255).astype(np.uint8)

        # Floor gradient
        for y in range(H // 2, H):
            alpha = (y - H // 2) / (H // 2)
            img[y] = (img[y] * (1 - alpha * 0.3)).astype(np.uint8)

        # Poster on wall (textured rectangle)
        px, py, pw, ph = 180 + shift_x, 80, 120, 160
        img[py:py+ph, px:px+pw] = [30, 80, 160]
        for i in range(5):
            yy = py + 20 + i * 25
            img[yy:yy+8, px+10:px+pw-10] = [220, 220, 180]

        # Object of interest: red box on a table
        bx, by, bw_, bh_ = 270 + shift_x, 260, 80, 70
        img[by:by+bh_, bx:bx+bw_] = [45, 45, 180]
        img[by+5:by+bh_-5, bx+5:bx+bw_-5] = [60, 60, 210]
        # label
        cv2.putText(img, "BOX", (bx + 10, by + bh_ // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Table surface
        img[325:340, 200+shift_x:440+shift_x] = [100, 70, 50]

        # Window (bright)
        img[40:200, 480+shift_x:620+shift_x] = [220, 230, 255]
        for i in range(3):
            x_ = 480 + shift_x + 20 + i * 45
            img[40:200, x_:x_+4] = [150, 150, 130]
        img[120:124, 480+shift_x:620+shift_x] = [150, 150, 130]

        # Some dots/features for SIFT
        for _ in range(60):
            cx_ = random.randint(10, W - 10)
            cy_ = random.randint(10, H - 10)
            r_ = random.randint(3, 8)
            col = tuple(random.randint(30, 200) for _ in range(3))
            cv2.circle(img, (cx_, cy_), r_, col, -1)

        return img

    img_l = make_scene(shift_x=0)
    img_r = make_scene(shift_x=-DISPARITY_PX)

    path_l = os.path.join(out_dir, "left.jpg")
    path_r = os.path.join(out_dir, "right.jpg")
    cv2.imwrite(path_l, img_l)
    cv2.imwrite(path_r, img_r)

    # Known ground truth depth: Z = f * B / d
    # f ≈ W=640, B=100mm, d=16px  → Z ≈ 640*100/16 = 4000 mm = 4.0 m
    gt_mm = float(W) * 100.0 / DISPARITY_PX
    print(f"[demo] Synthetic images generated: {path_l}, {path_r}")
    print(f"[demo] Synthetic ground truth depth: {gt_mm:.1f} mm ({gt_mm/1000:.3f} m)")
    return path_l, path_r, gt_mm


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Uncalibrated Stereo Distance Estimation"
    )
    parser.add_argument("--left",      default=os.path.join(script_dir, "left.jpg"), help="Path to left image")
    parser.add_argument("--right",     default=os.path.join(script_dir, "right.jpg"), help="Path to right image")
    parser.add_argument("--baseline",  type=float, default=BASELINE_MM,
                        help="Baseline distance between cameras in mm (default 100)")
    parser.add_argument("--focal",     type=float, default=None,
                        help="Focal length in pixels (default: estimated from image width)")
    parser.add_argument("--gt",        type=float, default=None,
                        help="Ground truth distance in mm (optional)")
    parser.add_argument("--demo",      action="store_true",
                        help="Generate and use synthetic demo images")
    parser.add_argument("--out",       default=os.path.join(script_dir, "output"), help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Load images ────────────────────────────────────────────────────────
    if args.demo or not (os.path.exists(args.left) and os.path.exists(args.right)):
        print("[info] Images not found or demo requested — generating demo synthetic pair.")
        path_l, path_r, gt_mm = generate_demo_images(script_dir)
        gt_mm = args.gt if args.gt else gt_mm
    else:
        path_l, path_r = args.left, args.right
        gt_mm = args.gt

    img_l = load_image(path_l)
    img_r = load_image(path_r)
    h, w = img_l.shape[:2]

    # ── Camera intrinsics (estimated) ──────────────────────────────────────
    focal_px = args.focal if args.focal else estimate_focal_length(w)
    K = build_camera_matrix(focal_px, w, h)
    print(f"\n[K] Camera Matrix (estimated, focal={focal_px:.1f} px):\n{K}\n")

    # ── Feature matching ──────────────────────────────────────────────────
    print("[1] Detecting and matching features …")
    kp_l, kp_r, good_matches, pts_l, pts_r = detect_and_match(img_l, img_r)
    print(f"    → {len(good_matches)} good matches found")

    # ── Fundamental Matrix ────────────────────────────────────────────────
    print("[2] Computing Fundamental Matrix …")
    F, inliers_l, inliers_r, inlier_mask = compute_fundamental_matrix(pts_l, pts_r)
    print(f"    → {inlier_mask.sum()} inliers after RANSAC")
    print(f"\n[F] Fundamental Matrix:\n{F}\n")

    # ── Essential Matrix ──────────────────────────────────────────────────
    print("[3] Computing Essential Matrix …")
    E = compute_essential_matrix(F, K)
    print(f"\n[E] Essential Matrix:\n{E}\n")

    # ── Rotation & Translation ────────────────────────────────────────────
    print("[4] Recovering Rotation and Translation …")
    R, t, _ = decompose_essential_matrix(E, K, inliers_l, inliers_r)
    print(f"\n[R] Rotation Matrix:\n{R}\n")
    print(f"[t] Translation vector (unit): {t.ravel()}\n")

    t_norm = float(np.linalg.norm(t))

    # ── Triangulation ─────────────────────────────────────────────────────
    print("[5] Triangulating points …")
    pts3d = triangulate_points(R, t, K, inliers_l, inliers_r)

    # ── Distance estimation ───────────────────────────────────────────────
    print("[6] Estimating distance …")
    distance_mm = estimate_object_distance(pts3d, args.baseline, t_norm)
    print(f"\n{'─'*50}")
    print(f"  Estimated distance : {distance_mm/1000:.4f} m  ({distance_mm:.2f} mm)")
    if gt_mm:
        err = abs(distance_mm - gt_mm) / gt_mm * 100
        print(f"  Ground truth       : {gt_mm/1000:.4f} m  ({gt_mm:.2f} mm)")
        print(f"  Absolute error     : {err:.2f} %")
    print(f"{'─'*50}\n")

    # ── Epipolar lines ────────────────────────────────────────────────────
    print("[7] Drawing epipolar lines …")
    epi_l, epi_r = draw_epipolar_lines(img_l, img_r, inliers_l, inliers_r, F)

    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

    # ── Save figures ──────────────────────────────────────────────────────
    report_path    = os.path.join(args.out, "stereo_report.png")
    annotated_path = os.path.join(args.out, "annotated_setup.png")

    save_report_figure(
        img_l_rgb, img_r_rgb,
        epi_l, epi_r,
        pts3d, distance_mm, gt_mm,
        R, E, F,
        inliers_l, inliers_r,
        out_path=report_path
    )
    save_annotated_setup(img_l_rgb, distance_mm, gt_mm, out_path=annotated_path)

    print(f"\n[✓] All outputs saved to: {args.out}/")


if __name__ == "__main__":
    main()