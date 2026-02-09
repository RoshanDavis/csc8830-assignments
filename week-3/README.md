# Week 3: Image Blurring - Spatial vs Frequency Domain

This project demonstrates the **Convolution Theorem** — a fundamental concept in signal processing that shows convolution in the spatial domain is mathematically equivalent to multiplication in the frequency (Fourier) domain.

## Theory

### The Convolution Theorem

The convolution theorem states that under suitable conditions:

$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

Where:
- $f * g$ denotes convolution of functions $f$ and $g$
- $\mathcal{F}\{\cdot\}$ denotes the Fourier Transform
- $\cdot$ denotes element-wise (pointwise) multiplication

In the context of image processing:

| Domain | Operation | Description |
|--------|-----------|-------------|
| Spatial | Convolution | Slide kernel over image, sum products |
| Frequency | Multiplication | Element-wise multiply FFT spectra |

### Gaussian Blur

A 2D Gaussian kernel is defined as:

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

Where $\sigma$ (sigma) controls the blur strength — larger values produce more blur.

### Why This Matters

1. **Computational Efficiency**: For large kernels, frequency domain operations can be faster due to FFT's $O(n \log n)$ complexity
2. **Filter Design**: Some filters are easier to design in frequency domain (e.g., ideal low-pass)
3. **Understanding**: Reveals how filtering affects different frequency components of an image

## Setup

### Create Virtual Environment

```bash
cd week-3
python -m venv venv
```

### Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Generated Sample Image)

```bash
python blur.py
```

### With Custom Image

```bash
python blur.py --image path/to/your/image.jpg
```

### Adjust Blur Parameters

```bash
# Increase blur strength (larger sigma)
python blur.py --sigma 5.0

# Use larger kernel
python blur.py --kernel-size 21

# Combine options
python blur.py --image photo.jpg --sigma 3.0 --kernel-size 15
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--image` | None | Path to input image (uses generated sample if not provided) |
| `--sigma` | 2.0 | Gaussian blur standard deviation |
| `--kernel-size` | 11 | Size of the kernel (must be odd) |
| `--no-display` | False | Skip visualization, only print metrics |

## Project Structure

```
week-3/
├── blur.py              # Main demonstration script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── venv/               # Virtual environment (created by user)
├── sample_images/      # Optional: place test images here
└── output/             # Generated output images
    ├── sample_image.png    # Generated test image
    ├── spatial_blur.png    # Result from spatial convolution
    ├── frequency_blur.png  # Result from frequency multiplication
    └── comparison.png      # Side-by-side visualization
```

## Output Explanation

The script produces:

1. **Spatial Domain Blur**: Applied using `scipy.signal.convolve2d()` which performs true 2D convolution with circular boundary handling
2. **Frequency Domain Blur**: Applied using FFT multiplication:
   - Zero-pad kernel to image size (centered at origin)
   - Compute FFT of image: $F = \mathcal{F}\{image\}$
   - Compute FFT of padded kernel: $H = \mathcal{F}\{kernel\}$
   - Multiply: $G = F \cdot H$
   - Inverse FFT: $result = \mathcal{F}^{-1}\{G\}$

3. **Comparison Metrics**:
   - **MSE (Mean Squared Error)**: Should be ~0 (< 1e-20)
   - **Max Difference**: Maximum pixel value difference (< 1e-12)
   - **PSNR**: Higher values indicate more similarity (>300 dB)

## Expected Results

When running the script, you should see output similar to:

```
==================================================
COMPARISON RESULTS
==================================================
Mean Squared Error (MSE):     1.305737e-27
Maximum Absolute Difference:  2.842171e-13
Peak Signal-to-Noise Ratio:   316.97 dB

✓ Results are IDENTICAL (within numerical precision)
✓ This proves: Convolution in space = Multiplication in frequency
==================================================
```

The near-zero MSE (~10⁻²⁷) and extremely high PSNR (>300 dB) demonstrate that both methods produce mathematically identical results, confirming the Convolution Theorem.

## Dependencies

- `opencv-python` - Image I/O and display
- `numpy` - Array operations and FFT (`numpy.fft`)
- `scipy` - True 2D convolution (`scipy.signal.convolve2d`)
- `matplotlib` - Visualization
- `pillow` - Image format support
