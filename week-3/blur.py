"""
Image Blurring: Spatial vs Frequency Domain Filtering

This script demonstrates that convolution in the spatial domain is equivalent
to multiplication in the frequency (Fourier) domain. It applies Gaussian blur
using both approaches and shows they produce identical results.

Theory:
    The Convolution Theorem states that:
    - Convolution in spatial domain ↔ Multiplication in frequency domain
    - f(x,y) * h(x,y) ↔ F(u,v) · H(u,v)
    
    Where:
    - f(x,y) is the input image
    - h(x,y) is the filter kernel (spatial domain)
    - F(u,v) and H(u,v) are their Fourier transforms
    - * denotes convolution, · denotes element-wise multiplication

Usage:
    python blur.py                      # Use default sample image
    python blur.py --image path/to/img  # Use custom image
    python blur.py --sigma 3.0          # Adjust blur strength
    python blur.py --kernel-size 15     # Adjust kernel size

Controls (when visualization window is open):
    - Press any key to close the window
    - Close the matplotlib window to exit
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import signal

# Script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
SAMPLE_DIR = os.path.join(SCRIPT_DIR, "sample_images")


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian kernel.
    
    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Normalized 2D Gaussian kernel
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size for symmetric kernel
    
    # Create coordinate grid centered at zero
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    
    # 2D Gaussian formula: G(x,y) = exp(-(x² + y²) / (2σ²))
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize so kernel sums to 1 (preserves image brightness)
    kernel = kernel / kernel.sum()
    
    return kernel


def spatial_domain_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply blur using spatial domain convolution.
    
    This uses scipy.signal.convolve2d which performs true 2D convolution:
    output(x,y) = Σ Σ image(x-i, y-j) * kernel(i,j)
    
    Args:
        image: Input grayscale image
        kernel: Convolution kernel
        
    Returns:
        Blurred image (same size as input)
    """
    # scipy.signal.convolve2d performs true convolution
    # mode='same' returns output same size as input
    # boundary='wrap' uses circular/periodic boundary (matches FFT behavior)
    result = signal.convolve2d(image.astype(np.float64), kernel, 
                               mode='same', boundary='wrap')
    return result


def frequency_domain_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply blur using frequency domain multiplication (circular convolution).
    
    This demonstrates the Convolution Theorem by implementing FFT-based
    circular convolution:
    1. Zero-pad kernel to image size (center kernel at origin)
    2. Compute FFT of both image and padded kernel
    3. Multiply the spectra element-wise
    4. Compute inverse FFT to get spatial result
    
    Note: FFT multiplication naturally implements CIRCULAR convolution,
    which is why we use boundary='wrap' in the spatial domain version.
    
    Args:
        image: Input grayscale image
        kernel: Convolution kernel
        
    Returns:
        Blurred image (same size as input)
    """
    img_h, img_w = image.shape
    kern_h, kern_w = kernel.shape
    
    # Create zero-padded kernel matching image dimensions
    padded_kernel = np.zeros((img_h, img_w), dtype=np.float64)
    
    # For circular convolution to match scipy.signal.convolve2d with 'wrap',
    # we need to place the kernel such that its center is at position (0,0)
    # and wraps around the edges
    
    # Center of the kernel
    center_y = kern_h // 2
    center_x = kern_w // 2
    
    # Place each kernel element at the correct wrapped position
    for i in range(kern_h):
        for j in range(kern_w):
            # Calculate position relative to kernel center
            rel_i = i - center_y
            rel_j = j - center_x
            # Wrap to image coordinates
            dest_i = rel_i % img_h
            dest_j = rel_j % img_w
            padded_kernel[dest_i, dest_j] = kernel[i, j]
    
    # Compute 2D FFT of both
    image_fft = fft2(image.astype(np.float64))
    kernel_fft = fft2(padded_kernel)
    
    # Multiply in frequency domain (element-wise)
    # This is equivalent to circular convolution in spatial domain
    result_fft = image_fft * kernel_fft
    
    # Inverse FFT to get spatial domain result
    result = np.real(ifft2(result_fft))
    
    return result


def compute_difference_metrics(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Compute metrics showing how similar two images are.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Dictionary with MSE, max absolute difference, and PSNR
    """
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    
    mse = np.mean(diff**2)
    max_diff = np.max(np.abs(diff))
    
    # PSNR (Peak Signal-to-Noise Ratio) - higher means more similar
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')
    
    return {
        'mse': mse,
        'max_diff': max_diff,
        'psnr': psnr
    }


def visualize_results(original: np.ndarray, 
                     spatial_result: np.ndarray,
                     frequency_result: np.ndarray,
                     kernel: np.ndarray,
                     metrics: dict,
                     save_path: str = None):
    """
    Create visualization comparing spatial and frequency domain results.
    
    Args:
        original: Original input image
        spatial_result: Result from spatial domain convolution
        frequency_result: Result from frequency domain multiplication
        kernel: The Gaussian kernel used
        metrics: Dictionary of comparison metrics
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Convolution Theorem: Spatial Domain vs Frequency Domain', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Images
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(spatial_result, cmap='gray')
    axes[0, 1].set_title('Spatial Domain Blur\n(cv2.filter2D convolution)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(frequency_result, cmap='gray')
    axes[0, 2].set_title('Frequency Domain Blur\n(FFT multiplication)')
    axes[0, 2].axis('off')
    
    # Row 2: Analysis
    # Gaussian kernel visualization
    axes[1, 0].imshow(kernel, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title(f'Gaussian Kernel\n({kernel.shape[0]}x{kernel.shape[1]})')
    axes[1, 0].axis('off')
    
    # Difference image (amplified for visibility)
    diff = np.abs(spatial_result - frequency_result)
    diff_amplified = diff * 1000  # Amplify tiny differences for visualization
    diff_amplified = np.clip(diff_amplified, 0, 255)
    axes[1, 1].imshow(diff_amplified, cmap='hot')
    axes[1, 1].set_title('Difference (amplified 1000x)\n(Should be near-black)')
    axes[1, 1].axis('off')
    
    # FFT magnitude spectrum of original image
    fft_original = fftshift(fft2(original))
    magnitude_spectrum = np.log1p(np.abs(fft_original))
    axes[1, 2].imshow(magnitude_spectrum, cmap='gray')
    axes[1, 2].set_title('FFT Magnitude Spectrum\n(log scale)')
    axes[1, 2].axis('off')
    
    # Add metrics text
    metrics_text = (
        f"Comparison Metrics:\n"
        f"─────────────────────\n"
        f"MSE: {metrics['mse']:.2e}\n"
        f"Max Diff: {metrics['max_diff']:.2e}\n"
        f"PSNR: {metrics['psnr']:.1f} dB\n"
        f"─────────────────────\n"
        f"(Lower MSE = More Similar)\n"
        f"(Higher PSNR = More Similar)"
    )
    fig.text(0.02, 0.02, metrics_text, fontsize=10, fontfamily='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def create_sample_image() -> np.ndarray:
    """
    Create a sample test image with various features.
    
    Returns:
        Grayscale test image
    """
    # Create a 512x512 test image
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add geometric shapes
    # Rectangle
    cv2.rectangle(img, (50, 50), (200, 150), 255, -1)
    
    # Circle
    cv2.circle(img, (350, 100), 60, 200, -1)
    
    # Triangle
    pts = np.array([[256, 200], [150, 350], [362, 350]], np.int32)
    cv2.fillPoly(img, [pts], 180)
    
    # Add text
    cv2.putText(img, 'FFT', (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 
                2, 255, 3, cv2.LINE_AA)
    
    # Add some noise texture
    noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Add gradient
    gradient = np.tile(np.linspace(0, 50, size, dtype=np.uint8), (size, 1))
    img = cv2.add(img, gradient.astype(np.uint8))
    
    return img


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file and convert to grayscale.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Grayscale image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img


def main():
    """Main function to run the spatial vs frequency domain comparison."""
    parser = argparse.ArgumentParser(
        description='Demonstrate spatial vs frequency domain image filtering'
    )
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (uses generated sample if not provided)')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Gaussian blur sigma (default: 5.0)')
    parser.add_argument('--kernel-size', type=int, default=21,
                        help='Kernel size, must be odd (default: 21)')
    parser.add_argument('--no-display', action='store_true',
                        help='Skip visualization, only print metrics')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load or create image
    if args.image:
        print(f"Loading image: {args.image}")
        image = load_image(args.image)
    else:
        print("No image provided, using generated sample image")
        image = create_sample_image()
        # Save the sample image
        sample_path = os.path.join(OUTPUT_DIR, "sample_image.png")
        cv2.imwrite(sample_path, image)
        print(f"Saved sample image to: {sample_path}")
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"Gaussian parameters: sigma={args.sigma}, kernel_size={args.kernel_size}")
    print()
    
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(args.kernel_size, args.sigma)
    print(f"Created Gaussian kernel: {kernel.shape[0]}x{kernel.shape[1]}")
    print(f"Kernel sum: {kernel.sum():.6f} (should be ~1.0)")
    print()
    
    # Apply spatial domain blur
    print("Applying spatial domain convolution (cv2.filter2D)...")
    spatial_result = spatial_domain_blur(image, kernel)
    
    # Apply frequency domain blur
    print("Applying frequency domain multiplication (FFT)...")
    frequency_result = frequency_domain_blur(image, kernel)
    
    # Compute comparison metrics
    print()
    print("=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    
    metrics = compute_difference_metrics(spatial_result, frequency_result)
    
    print(f"Mean Squared Error (MSE):     {metrics['mse']:.6e}")
    print(f"Maximum Absolute Difference:  {metrics['max_diff']:.6e}")
    print(f"Peak Signal-to-Noise Ratio:   {metrics['psnr']:.2f} dB")
    print()
    
    if metrics['mse'] < 1e-10:
        print("✓ Results are IDENTICAL (within numerical precision)")
        print("✓ This proves: Convolution in space = Multiplication in frequency")
    elif metrics['mse'] < 1e-6:
        print("✓ Results are NEARLY IDENTICAL (tiny floating-point differences)")
        print("✓ This demonstrates the Convolution Theorem")
    else:
        print("⚠ Results differ more than expected")
        print("  This may be due to boundary handling differences")
    
    print("=" * 50)
    
    # Save individual results
    cv2.imwrite(os.path.join(OUTPUT_DIR, "spatial_blur.png"), 
                spatial_result.astype(np.uint8))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "frequency_blur.png"), 
                frequency_result.astype(np.uint8))
    print(f"\nSaved results to: {OUTPUT_DIR}")
    
    # Visualize if not disabled
    if not args.no_display:
        visualize_results(
            original=image,
            spatial_result=spatial_result,
            frequency_result=frequency_result,
            kernel=kernel,
            metrics=metrics,
            save_path=os.path.join(OUTPUT_DIR, "comparison.png")
        )


if __name__ == "__main__":
    main()
