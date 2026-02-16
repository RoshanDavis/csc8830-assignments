# Week 4: Thermal Image Animal Boundary Detection

This project implements animal boundary detection in thermal infrared images using classical computer vision techniques (OpenCV), and compares the results with SAM2 (Segment Anything Model 2) deep learning segmentation.

## Theory

### Thermal Imaging

Thermal cameras capture **infrared radiation** emitted by objects based on their temperature. Animals, being warm-blooded (homeotherms), emit more infrared radiation than their cooler surroundings, creating **distinct heat signatures** in thermal images.

Key characteristics of thermal images:
- **Intensity correlates with temperature**: Warmer objects appear brighter (or darker, depending on color palette)
- **Bimodal histogram**: Animals typically create a clear separation from background temperatures
- **Lower resolution**: Thermal sensors typically have lower resolution than visible light cameras
- **Less texture**: Surface texture information is limited compared to visible light

### Classical Segmentation Pipeline

The implemented pipeline uses these techniques:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   
   Standard histogram equalization can amplify noise. CLAHE divides the image into tiles and applies localized equalization:
   
   $$T(r) = \text{round}\left(\frac{(L-1)}{MN} \sum_{j=0}^{r} n_j\right)$$
   
   where $T(r)$ is the transformation function, $L$ is the number of gray levels, $M \times N$ is the tile size, and $n_j$ is the histogram.

2. **Gaussian Blur**
   
   Reduces high-frequency noise while preserving edges:
   
   $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

3. **Otsu's Thresholding**
   
   Automatically determines the optimal threshold by minimizing **intra-class variance**:
   
   $$\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)$$
   
   where $\omega_0, \omega_1$ are class probabilities and $\sigma_0^2, \sigma_1^2$ are class variances.

4. **Morphological Operations**
   
   - **Opening** (erosion → dilation): Removes small noise
   - **Closing** (dilation → erosion): Fills small holes

5. **Contour Detection**
   
   Uses the Suzuki-Abe algorithm to trace boundaries of connected components.

### Comparison Metrics

- **IoU (Intersection over Union)**: $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$
- **Dice Coefficient**: $\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$
- **Boundary F1**: Precision/recall of boundary pixel alignment

## Setup

### Create Virtual Environment

**Windows (PowerShell):**
```powershell
cd week-4
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
cd week-4
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### SAM2 Installation (for comparison)

SAM2 requires PyTorch and the official SAM2 package:

```bash
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**Note**: SAM2 comparison requires a GPU with CUDA for reasonable performance.

## Usage

### 1. Run OpenCV Boundary Detection

Process all images in the `images/` folder:

```bash
python thermal_boundary.py --input images/ --output output/
```

Process a single image:

```bash
python thermal_boundary.py --input images/thermal.jpg --output output/
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | `images/` | Input image or directory |
| `--output`, `-o` | `output/` | Output directory |
| `--clip-limit` | `2.0` | CLAHE clip limit |
| `--blur-size` | `5` | Gaussian blur kernel size |
| `--min-area` | `500` | Minimum contour area |
| `--invert` | `False` | Invert threshold (dark objects) |
| `--no-intermediate` | `False` | Skip intermediate results |

#### Output Files

For each input image `IMAGE.jpg`:

| File | Description |
|------|-------------|
| `IMAGE_boundary.jpg` | Original with green contour overlay |
| `IMAGE_mask.png` | Binary segmentation mask |
| `IMAGE_boundary_only.png` | White boundary on black |
| `IMAGE_pipeline.png` | Processing pipeline visualization |
| `IMAGE_enhanced.jpg` | CLAHE enhanced image |
| `IMAGE_binary.png` | Otsu threshold result |
| `IMAGE_refined.png` | After morphological refinement |

### 2. Run SAM2 Comparison

After running OpenCV detection:

```bash
python compare_sam2.py --input images/ --opencv-dir output/
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | `images/` | Input images directory |
| `--opencv-dir` | `output/` | Directory with OpenCV masks |
| `--output`, `-o` | Same as opencv-dir | Output directory |
| `--model-size` | `large` | SAM2 model: tiny/small/base/large |
| `--device` | `auto` | Device: auto/cuda/cpu |

#### Output Files

| File | Description |
|------|-------------|
| `IMAGE_sam2_mask.png` | SAM2 segmentation mask |
| `IMAGE_comparison.png` | Side-by-side visualization |
| `comparison_results.txt` | Metrics summary |

## Project Structure

```
week-4/
├── thermal_boundary.py    # OpenCV boundary detection
├── compare_sam2.py        # SAM2 comparison script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── images/                # Input thermal images
│   └── *.jpg
├── output/                # Generated results
│   ├── *_boundary.jpg     # Boundary overlays
│   ├── *_mask.png         # Segmentation masks
│   ├── *_pipeline.png     # Pipeline visualizations
│   ├── *_sam2_mask.png    # SAM2 masks
│   ├── *_comparison.png   # Comparison visualizations
│   └── comparison_results.txt
└── checkpoints/           # SAM2 model checkpoints
    └── sam2_hiera_*.pt
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥4.8.0 | Image processing |
| `numpy` | ≥1.24.0 | Array operations |
| `matplotlib` | ≥3.7.0 | Visualization |
| `pillow` | ≥10.0.0 | Image I/O |
| `torch` | ≥2.0.0 | SAM2 backend |
| `torchvision` | ≥0.15.0 | Image transforms |
| `sam2` | ≥1.0 | Segment Anything 2 |

## Tips

### Tuning Parameters

1. **If animals are not fully segmented**:
   - Increase `--clip-limit` (e.g., 3.0 or 4.0)
   - Decrease `--min-area` to capture smaller regions

2. **If there's too much noise**:
   - Increase `--blur-size` (e.g., 7 or 9)
   - Increase `--min-area` to filter small detections

3. **If animals appear dark** (inverted thermal palette):
   - Use `--invert` flag

4. **For low-contrast thermal images**:
   - Try higher `--clip-limit` values (up to 5.0)

### Performance Notes

- OpenCV processing: ~50-200ms per image (CPU)
- SAM2 tiny: ~1-2s per image (GPU)
- SAM2 large: ~3-5s per image (GPU)
- SAM2 on CPU: Very slow, not recommended

## Example Results

The pipeline visualization shows each processing stage:

1. **Original Image**: Raw thermal image input
2. **CLAHE Enhanced**: Improved local contrast
3. **Otsu's Threshold**: Binary segmentation
4. **Morphological Refinement**: Cleaned mask
5. **Final Mask**: Animal segmentation
6. **Detected Boundaries**: Green contour on original

The comparison visualization shows:
- Side-by-side masks (OpenCV vs SAM2)
- Overlay visualizations
- Difference map (green=OpenCV only, red=SAM2 only, yellow=both)
- Quantitative metrics (IoU, Dice, Boundary F1)
