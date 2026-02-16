"""
Thermal Image Animal Boundary Detection
========================================

This script detects animal boundaries in thermal imaging camera images using
classical computer vision techniques (no deep learning/machine learning).

Thermal Imaging Principles:
---------------------------
Thermal cameras capture infrared radiation emitted by objects. Animals, being
warm-blooded, emit more infrared radiation than their surroundings, appearing
as brighter (or darker, depending on the color palette) regions in the image.

Pipeline:
---------
1. Load thermal image and convert to grayscale
2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for 
   enhanced contrast in thermal images
3. Apply Gaussian blur to reduce noise
4. Use Otsu's automatic thresholding to segment warm objects from background
5. Apply morphological operations (opening/closing) to clean the mask
6. Find contours and filter by area to isolate animal(s)
7. Draw boundaries and save results

Usage:
------
    python thermal_boundary.py --input images/ --output output/
    python thermal_boundary.py --input images/image.jpg --output output/
    
Controls/Options:
-----------------
    --input         Input image or directory path
    --output        Output directory for results
    --clip-limit    CLAHE clip limit (default: 2.0)
    --blur-size     Gaussian blur kernel size (default: 5)
    --min-area      Minimum contour area to keep (default: 500)
    --invert        Invert threshold (for dark animals on bright background)
"""

import os
import sys
import argparse
import glob
from typing import Tuple, List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocess_thermal_image(
    image: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_grid_size: Tuple[int, int] = (8, 8),
    blur_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess thermal image for segmentation.
    
    Args:
        image: Input BGR image
        clip_limit: CLAHE clip limit for contrast enhancement
        tile_grid_size: CLAHE tile grid size
        blur_size: Gaussian blur kernel size
        
    Returns:
        Tuple of (grayscale image, enhanced image)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE for contrast enhancement
    # This is particularly effective for thermal images with low contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    if blur_size > 0:
        enhanced = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0)
    
    return gray, enhanced


def segment_thermal_image(
    enhanced: np.ndarray,
    invert: bool = False
) -> np.ndarray:
    """
    Segment thermal image using Otsu's thresholding.
    
    Otsu's method automatically calculates the optimal threshold by minimizing
    intra-class variance. This works well for thermal images where animals
    create a bimodal intensity distribution.
    
    Args:
        enhanced: Preprocessed grayscale image
        invert: If True, invert the threshold (for dark objects)
        
    Returns:
        Binary mask
    """
    # Apply Otsu's automatic thresholding
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(enhanced, 0, 255, thresh_type + cv2.THRESH_OTSU)
    
    return binary


def refine_mask(
    binary: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 2
) -> np.ndarray:
    """
    Refine binary mask using morphological operations.
    
    - Opening (erosion followed by dilation): removes small noise
    - Closing (dilation followed by erosion): fills small holes
    
    Args:
        binary: Binary mask
        kernel_size: Size of morphological kernel
        iterations: Number of iterations for each operation
        
    Returns:
        Refined binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening to remove small noise
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Closing to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return closed


def find_animal_contours(
    mask: np.ndarray,
    min_area: int = 500,
    max_contours: int = 10
) -> List[np.ndarray]:
    """
    Find contours in the mask and filter by area.
    
    Args:
        mask: Binary mask
        min_area: Minimum contour area to keep
        max_contours: Maximum number of contours to return
        
    Returns:
        List of contours (sorted by area, largest first)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and sort by size
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            filtered.append((area, cnt))
    
    # Sort by area (largest first) and take top N
    filtered.sort(key=lambda x: x[0], reverse=True)
    result = [cnt for _, cnt in filtered[:max_contours]]
    
    return result


def draw_boundaries(
    image: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw contour boundaries on the image.
    
    Args:
        image: Input image (will be copied)
        contours: List of contours to draw
        color: BGR color for boundaries
        thickness: Line thickness
        
    Returns:
        Image with drawn boundaries
    """
    result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def create_boundary_mask(
    shape: Tuple[int, int],
    contours: List[np.ndarray],
    thickness: int = 2
) -> np.ndarray:
    """
    Create a binary image showing only the boundaries.
    
    Args:
        shape: Output image shape (height, width)
        contours: List of contours
        thickness: Line thickness
        
    Returns:
        Binary boundary image
    """
    boundary = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(boundary, contours, -1, 255, thickness)
    return boundary


def create_filled_mask(
    shape: Tuple[int, int],
    contours: List[np.ndarray]
) -> np.ndarray:
    """
    Create a filled binary mask from contours.
    
    Args:
        shape: Output image shape (height, width)
        contours: List of contours
        
    Returns:
        Filled binary mask
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)  # -1 thickness = fill
    return mask


def process_thermal_image(
    image_path: str,
    output_dir: str,
    clip_limit: float = 2.0,
    blur_size: int = 5,
    min_area: int = 500,
    invert: bool = False,
    save_intermediate: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Process a single thermal image to detect animal boundaries.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
        clip_limit: CLAHE clip limit
        blur_size: Gaussian blur kernel size
        min_area: Minimum contour area
        invert: Invert threshold for dark objects
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Tuple of (original image, mask, contours)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Preprocess
    gray, enhanced = preprocess_thermal_image(image, clip_limit, blur_size=blur_size)
    
    # Segment
    binary = segment_thermal_image(enhanced, invert=invert)
    
    # Refine mask
    refined = refine_mask(binary)
    
    # Find contours
    contours = find_animal_contours(refined, min_area=min_area)
    print(f"  Found {len(contours)} animal region(s)")
    
    # Create output images
    boundary_overlay = draw_boundaries(image, contours, color=(0, 255, 0), thickness=2)
    filled_mask = create_filled_mask(gray.shape, contours)
    boundary_only = create_boundary_mask(gray.shape, contours, thickness=2)
    
    # Generate output filename base
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save boundary overlay
    overlay_path = os.path.join(output_dir, f"{base_name}_boundary.jpg")
    cv2.imwrite(overlay_path, boundary_overlay)
    print(f"  Saved: {os.path.basename(overlay_path)}")
    
    # Save filled mask (for comparison with SAM2)
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, filled_mask)
    print(f"  Saved: {os.path.basename(mask_path)}")
    
    # Save boundary-only image
    boundary_path = os.path.join(output_dir, f"{base_name}_boundary_only.png")
    cv2.imwrite(boundary_path, boundary_only)
    print(f"  Saved: {os.path.basename(boundary_path)}")
    
    if save_intermediate:
        # Save enhanced (CLAHE) image
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
        cv2.imwrite(enhanced_path, enhanced)
        
        # Save binary threshold result
        binary_path = os.path.join(output_dir, f"{base_name}_binary.png")
        cv2.imwrite(binary_path, binary)
        
        # Save refined mask
        refined_path = os.path.join(output_dir, f"{base_name}_refined.png")
        cv2.imwrite(refined_path, refined)
    
    # Create visualization figure
    create_visualization(image, enhanced, binary, refined, filled_mask, 
                        boundary_overlay, output_dir, base_name)
    
    return image, filled_mask, contours


def create_visualization(
    original: np.ndarray,
    enhanced: np.ndarray,
    binary: np.ndarray,
    refined: np.ndarray,
    mask: np.ndarray,
    boundary_overlay: np.ndarray,
    output_dir: str,
    base_name: str
) -> None:
    """
    Create a visualization showing the processing pipeline.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Thermal Image Boundary Detection Pipeline\n{base_name}', fontsize=14)
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    # Enhanced (CLAHE)
    axes[0, 1].imshow(enhanced, cmap='gray')
    axes[0, 1].set_title('2. CLAHE Enhanced')
    axes[0, 1].axis('off')
    
    # Binary threshold
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title("3. Otsu's Threshold")
    axes[0, 2].axis('off')
    
    # Refined mask
    axes[1, 0].imshow(refined, cmap='gray')
    axes[1, 0].set_title('4. Morphological Refinement')
    axes[1, 0].axis('off')
    
    # Final mask
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('5. Final Segmentation Mask')
    axes[1, 1].axis('off')
    
    # Boundary overlay
    axes[1, 2].imshow(cv2.cvtColor(boundary_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('6. Detected Boundaries')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f"{base_name}_pipeline.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(fig_path)}")


def get_image_files(input_path: str) -> List[str]:
    """
    Get list of image files from input path.
    
    Args:
        input_path: File or directory path
        
    Returns:
        List of image file paths
    """
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


def main():
    parser = argparse.ArgumentParser(
        description='Detect animal boundaries in thermal images using classical CV techniques.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python thermal_boundary.py --input images/ --output output/
  python thermal_boundary.py --input images/thermal.jpg --output output/
  python thermal_boundary.py --input images/ --output output/ --invert
  python thermal_boundary.py --input images/ --clip-limit 3.0 --blur-size 7
        """
    )
    
    parser.add_argument('--input', '-i', type=str, 
                        default=os.path.join(SCRIPT_DIR, 'images'),
                        help='Input image or directory path (default: images/)')
    parser.add_argument('--output', '-o', type=str,
                        default=os.path.join(SCRIPT_DIR, 'output'),
                        help='Output directory (default: output/)')
    parser.add_argument('--clip-limit', type=float, default=2.0,
                        help='CLAHE clip limit for contrast enhancement (default: 2.0)')
    parser.add_argument('--blur-size', type=int, default=5,
                        help='Gaussian blur kernel size, must be odd (default: 5)')
    parser.add_argument('--min-area', type=int, default=500,
                        help='Minimum contour area to keep (default: 500)')
    parser.add_argument('--invert', action='store_true',
                        help='Invert threshold (for dark animals on bright background)')
    parser.add_argument('--no-intermediate', action='store_true',
                        help='Do not save intermediate processing results')
    
    args = parser.parse_args()
    
    # Ensure blur size is odd
    if args.blur_size % 2 == 0:
        args.blur_size += 1
        print(f"Note: Blur size adjusted to {args.blur_size} (must be odd)")
    
    # Get image files
    try:
        image_files = get_image_files(args.input)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if not image_files:
        print(f"No image files found in: {args.input}")
        sys.exit(1)
    
    print(f"\nThermal Image Boundary Detection")
    print(f"=" * 40)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Images found: {len(image_files)}")
    print(f"Parameters:")
    print(f"  CLAHE clip limit: {args.clip_limit}")
    print(f"  Blur size: {args.blur_size}")
    print(f"  Min area: {args.min_area}")
    print(f"  Invert: {args.invert}")
    print(f"=" * 40 + "\n")
    
    # Process each image
    for image_path in image_files:
        try:
            process_thermal_image(
                image_path=image_path,
                output_dir=args.output,
                clip_limit=args.clip_limit,
                blur_size=args.blur_size,
                min_area=args.min_area,
                invert=args.invert,
                save_intermediate=not args.no_intermediate
            )
            print()
        except Exception as e:
            print(f"  Error processing {os.path.basename(image_path)}: {e}")
            print()
    
    print(f"Processing complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
