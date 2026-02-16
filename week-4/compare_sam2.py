"""
SAM2 Comparison for Thermal Image Segmentation
===============================================

This script runs SAM2 (Segment Anything Model 2) on thermal images and compares
the results with classical OpenCV-based boundary detection.

SAM2 Overview:
--------------
SAM2 is Meta's state-of-the-art segmentation model that can segment any object
in an image. It uses a vision transformer (ViT) backbone and can generate
high-quality masks with minimal user input.

Comparison Metrics:
-------------------
- IoU (Intersection over Union): intersection / union
  - 1.0 = perfect overlap, 0.0 = no overlap
  
- Dice Coefficient: 2 * intersection / (area1 + area2)
  - Also known as F1 score for segmentation
  - More sensitive to small differences than IoU

- Boundary F1: Precision and recall of boundary pixels
  - Measures how well boundaries align

Usage:
------
    python compare_sam2.py --input images/ --opencv-dir output/
    python compare_sam2.py --input images/ --opencv-dir output/ --model-size large
"""

import os
import sys
import argparse
import glob
from typing import Tuple, List, Dict, Optional
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# SAM2 imports (conditional)
SAM2_AVAILABLE = False
try:
    import torch
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    pass


# SAM2 HuggingFace model IDs
SAM2_HF_MODELS = {
    'tiny': 'facebook/sam2-hiera-tiny',
    'small': 'facebook/sam2-hiera-small',
    'base': 'facebook/sam2-hiera-base-plus',
    'large': 'facebook/sam2-hiera-large',
}


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Ensure binary masks
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Dice coefficient between two binary masks.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Dice coefficient (0.0 to 1.0)
    """
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = mask1.sum()
    area2 = mask2.sum()
    
    if area1 + area2 == 0:
        return 0.0
    
    return 2 * intersection / (area1 + area2)


def calculate_boundary_metrics(
    mask1: np.ndarray, 
    mask2: np.ndarray, 
    tolerance: int = 2
) -> Dict[str, float]:
    """
    Calculate boundary-based metrics.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        tolerance: Pixel tolerance for boundary matching
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # Get boundaries using morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary1 = cv2.morphologyEx(mask1, cv2.MORPH_GRADIENT, kernel)
    boundary2 = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, kernel)
    
    # Dilate boundaries by tolerance
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (2*tolerance+1, 2*tolerance+1))
    dilated1 = cv2.dilate(boundary1, dilate_kernel)
    dilated2 = cv2.dilate(boundary2, dilate_kernel)
    
    # Calculate matches
    matches1 = np.logical_and(boundary1 > 0, dilated2 > 0).sum()
    matches2 = np.logical_and(boundary2 > 0, dilated1 > 0).sum()
    
    total1 = (boundary1 > 0).sum()
    total2 = (boundary2 > 0).sum()
    
    precision = matches1 / total1 if total1 > 0 else 0.0
    recall = matches2 / total2 if total2 > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        'boundary_precision': precision,
        'boundary_recall': recall,
        'boundary_f1': f1
    }


def load_sam2_model(
    model_size: str = 'large',
    device: str = 'auto'
) -> 'SAM2AutomaticMaskGenerator':
    """
    Load SAM2 model for automatic mask generation.
    
    Args:
        model_size: Model size ('tiny', 'small', 'base', 'large')
        device: Device to use ('auto', 'cuda', 'cpu')
        
    Returns:
        SAM2AutomaticMaskGenerator instance
    """
    if not SAM2_AVAILABLE:
        raise ImportError(
            "SAM2 is not installed. Please install it with:\n"
            "  pip install sam2\n"
            "Or clone from: https://github.com/facebookresearch/segment-anything-2"
        )
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading SAM2 {model_size} model on {device}...")
    
    # Get HuggingFace model ID
    model_id = SAM2_HF_MODELS[model_size]
    
    # Build model from HuggingFace (handles downloading automatically)
    sam2 = build_sam2_hf(model_id, device=device)
    
    # Create mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )
    
    print("  Model loaded successfully!")
    return mask_generator


def run_sam2_segmentation(
    mask_generator: 'SAM2AutomaticMaskGenerator',
    image: np.ndarray
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run SAM2 automatic mask generation on an image.
    
    Args:
        mask_generator: SAM2AutomaticMaskGenerator instance
        image: Input image (BGR or RGB)
        
    Returns:
        Tuple of (combined mask, list of mask dictionaries)
    """
    # SAM2 expects RGB
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Generate masks
    start_time = time.time()
    masks = mask_generator.generate(image_rgb)
    elapsed = time.time() - start_time
    
    print(f"    SAM2 generated {len(masks)} masks in {elapsed:.2f}s")
    
    # Sort by predicted IoU and area
    masks = sorted(masks, key=lambda x: x['predicted_iou'] * x['area'], reverse=True)
    
    # Create combined mask (largest/most confident masks)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Take top masks that likely represent animals (larger, high confidence)
    for mask_data in masks[:5]:  # Top 5 masks
        if mask_data['area'] > 500:  # Minimum area
            combined_mask = np.logical_or(combined_mask, mask_data['segmentation'])
    
    combined_mask = (combined_mask * 255).astype(np.uint8)
    
    return combined_mask, masks


def compare_masks(
    opencv_mask: np.ndarray,
    sam2_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compare two masks and compute all metrics.
    
    Args:
        opencv_mask: Mask from OpenCV processing
        sam2_mask: Mask from SAM2
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'iou': calculate_iou(opencv_mask, sam2_mask),
        'dice': calculate_dice(opencv_mask, sam2_mask)
    }
    
    # Add boundary metrics
    boundary_metrics = calculate_boundary_metrics(opencv_mask, sam2_mask)
    metrics.update(boundary_metrics)
    
    return metrics


def create_comparison_visualization(
    original: np.ndarray,
    opencv_mask: np.ndarray,
    sam2_mask: np.ndarray,
    metrics: Dict[str, float],
    output_path: str,
    title: str
) -> None:
    """
    Create a side-by-side comparison visualization.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'OpenCV vs SAM2 Comparison: {title}', fontsize=14)
    
    # Convert BGR to RGB for display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Row 1: Original, OpenCV mask, SAM2 mask
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(opencv_mask, cmap='gray')
    axes[0, 1].set_title('OpenCV Segmentation')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sam2_mask, cmap='gray')
    axes[0, 2].set_title('SAM2 Segmentation')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays and difference
    # OpenCV overlay
    opencv_overlay = original_rgb.copy()
    opencv_overlay[opencv_mask > 0] = opencv_overlay[opencv_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    axes[1, 0].imshow(opencv_overlay.astype(np.uint8))
    axes[1, 0].set_title('OpenCV Overlay')
    axes[1, 0].axis('off')
    
    # SAM2 overlay
    sam2_overlay = original_rgb.copy()
    sam2_overlay[sam2_mask > 0] = sam2_overlay[sam2_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[1, 1].imshow(sam2_overlay.astype(np.uint8))
    axes[1, 1].set_title('SAM2 Overlay')
    axes[1, 1].axis('off')
    
    # Difference visualization
    # Green = OpenCV only, Red = SAM2 only, Yellow = Both
    diff_img = np.zeros((*opencv_mask.shape, 3), dtype=np.uint8)
    opencv_only = np.logical_and(opencv_mask > 0, sam2_mask == 0)
    sam2_only = np.logical_and(sam2_mask > 0, opencv_mask == 0)
    both = np.logical_and(opencv_mask > 0, sam2_mask > 0)
    
    diff_img[opencv_only] = [0, 255, 0]    # Green
    diff_img[sam2_only] = [255, 0, 0]      # Red
    diff_img[both] = [255, 255, 0]         # Yellow
    
    axes[1, 2].imshow(diff_img)
    axes[1, 2].set_title('Difference\n(Green=OpenCV, Red=SAM2, Yellow=Both)')
    axes[1, 2].axis('off')
    
    # Add metrics text
    metrics_text = (
        f"Metrics:\n"
        f"  IoU: {metrics['iou']:.4f}\n"
        f"  Dice: {metrics['dice']:.4f}\n"
        f"  Boundary F1: {metrics['boundary_f1']:.4f}"
    )
    fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def get_image_files(input_path: str) -> List[str]:
    """Get list of image files from input path."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(input_path, ext)))
            files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        return sorted(files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def find_opencv_mask(opencv_dir: str, image_name: str) -> Optional[str]:
    """Find corresponding OpenCV mask for an image."""
    base_name = os.path.splitext(image_name)[0]
    mask_path = os.path.join(opencv_dir, f"{base_name}_mask.png")
    
    if os.path.exists(mask_path):
        return mask_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Compare OpenCV boundary detection with SAM2 segmentation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_sam2.py --input images/ --opencv-dir output/
  python compare_sam2.py --input images/ --opencv-dir output/ --model-size tiny
  python compare_sam2.py --input images/ --opencv-dir output/ --device cpu
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                        default=os.path.join(SCRIPT_DIR, 'images'),
                        help='Input images directory (default: images/)')
    parser.add_argument('--opencv-dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'output'),
                        help='Directory with OpenCV masks (default: output/)')
    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='Output directory (default: same as opencv-dir)')
    parser.add_argument('--model-size', type=str, default='large',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='SAM2 model size (default: large)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run SAM2 on (default: auto)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.opencv_dir
    
    # Check dependencies
    if not SAM2_AVAILABLE:
        print("Error: SAM2 is not available.")
        print("Please install the required dependencies:")
        print("  pip install torch torchvision sam2")
        print("\nOr install from source:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        sys.exit(1)
    
    # Get image files
    try:
        image_files = get_image_files(args.input)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if not image_files:
        print(f"No image files found in: {args.input}")
        sys.exit(1)
    
    print(f"\nSAM2 Comparison")
    print(f"=" * 40)
    print(f"Input images: {args.input}")
    print(f"OpenCV masks: {args.opencv_dir}")
    print(f"Output: {args.output}")
    print(f"Model: SAM2 {args.model_size}")
    print(f"Device: {args.device}")
    print(f"Images found: {len(image_files)}")
    print(f"=" * 40 + "\n")
    
    # Load SAM2 model
    try:
        mask_generator = load_sam2_model(args.model_size, args.device)
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        sys.exit(1)
    
    # Process each image
    all_metrics = []
    
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        print(f"\nProcessing: {image_name}")
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: Could not load image")
            continue
        
        # Find OpenCV mask
        opencv_mask_path = find_opencv_mask(args.opencv_dir, image_name)
        if opencv_mask_path is None:
            print(f"  Warning: No OpenCV mask found. Run thermal_boundary.py first.")
            print(f"    Expected: {args.opencv_dir}/{base_name}_mask.png")
            continue
        
        opencv_mask = cv2.imread(opencv_mask_path, cv2.IMREAD_GRAYSCALE)
        if opencv_mask is None:
            print(f"  Error: Could not load OpenCV mask")
            continue
        
        print(f"  OpenCV mask: {os.path.basename(opencv_mask_path)}")
        
        # Run SAM2
        try:
            sam2_mask, sam2_masks = run_sam2_segmentation(mask_generator, image)
        except Exception as e:
            print(f"  Error running SAM2: {e}")
            continue
        
        # Save SAM2 mask
        sam2_mask_path = os.path.join(args.output, f"{base_name}_sam2_mask.png")
        cv2.imwrite(sam2_mask_path, sam2_mask)
        print(f"  Saved SAM2 mask: {os.path.basename(sam2_mask_path)}")
        
        # Compare masks
        metrics = compare_masks(opencv_mask, sam2_mask)
        metrics['image'] = image_name
        all_metrics.append(metrics)
        
        print(f"  Metrics:")
        print(f"    IoU: {metrics['iou']:.4f}")
        print(f"    Dice: {metrics['dice']:.4f}")
        print(f"    Boundary F1: {metrics['boundary_f1']:.4f}")
        
        # Create comparison visualization
        comparison_path = os.path.join(args.output, f"{base_name}_comparison.png")
        create_comparison_visualization(
            image, opencv_mask, sam2_mask, metrics, comparison_path, base_name
        )
        print(f"  Saved comparison: {os.path.basename(comparison_path)}")
    
    # Save summary
    if all_metrics:
        summary_path = os.path.join(args.output, "comparison_results.txt")
        with open(summary_path, 'w') as f:
            f.write("SAM2 vs OpenCV Comparison Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: SAM2 {args.model_size}\n")
            f.write(f"Images processed: {len(all_metrics)}\n\n")
            
            f.write("Per-Image Results:\n")
            f.write("-" * 50 + "\n")
            for m in all_metrics:
                f.write(f"\n{m['image']}:\n")
                f.write(f"  IoU: {m['iou']:.4f}\n")
                f.write(f"  Dice: {m['dice']:.4f}\n")
                f.write(f"  Boundary F1: {m['boundary_f1']:.4f}\n")
            
            # Calculate averages
            avg_iou = np.mean([m['iou'] for m in all_metrics])
            avg_dice = np.mean([m['dice'] for m in all_metrics])
            avg_boundary_f1 = np.mean([m['boundary_f1'] for m in all_metrics])
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Average Metrics:\n")
            f.write(f"  IoU: {avg_iou:.4f}\n")
            f.write(f"  Dice: {avg_dice:.4f}\n")
            f.write(f"  Boundary F1: {avg_boundary_f1:.4f}\n")
        
        print(f"\n\nSummary saved to: {summary_path}")
        print(f"\nAverage Metrics:")
        print(f"  IoU: {avg_iou:.4f}")
        print(f"  Dice: {avg_dice:.4f}")
        print(f"  Boundary F1: {avg_boundary_f1:.4f}")
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
