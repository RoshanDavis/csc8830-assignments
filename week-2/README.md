# Week 2: Camera Calibration and Real-World Measurement

This assignment implements camera calibration and real-world 2D measurement using OpenCV's perspective projection equations.

## Assignment Overview

### Objectives
1. **Camera Calibration** - Compute camera intrinsic parameters (focal length, optical center) and distortion coefficients using a checkerboard pattern
2. **Real-World Measurement** - Measure actual dimensions of objects using perspective projection equations

### Perspective Projection Equations

The core equations used to convert pixel coordinates to real-world measurements:

$$X = \frac{(u - c_x) \cdot Z}{f_x}$$

$$Y = \frac{(v - c_y) \cdot Z}{f_y}$$

Where:
- $(u, v)$ = pixel coordinates in image
- $(c_x, c_y)$ = principal point (optical center)
- $(f_x, f_y)$ = focal lengths in pixels
- $Z$ = distance from camera to object plane
- $(X, Y)$ = real-world coordinates in mm

---

## Setup Instructions

### 1. Create Virtual Environment
(This project used Python 3.11.9)
```bash
cd week-2
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Scripts

### `camera_test.py`
Test your camera setup and verify available camera indices.

```bash
python camera_test.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `c` | Cycle to next camera |

---

### `calibrate.py` (Step 1)
Perform camera calibration using a checkerboard pattern.

```bash
python calibrate.py
```

**What you need:**
1. Print a checkerboard pattern (9x6 inner corners recommended)
   - Download from: https://github.com/opencv/opencv/blob/master/doc/pattern.png
2. Show the checkerboard to the camera from 15-20 different angles
3. Press **SPACE** to capture when corners are detected

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | Capture frame (when corners detected) |
| `q` | Finish and calibrate |
| `c` | Switch camera |
| `ESC` | Exit without calibrating |

**Output:** Saves calibration data to `calibration_data/` folder:
- `camera_matrix.npy` - Intrinsic camera matrix
- `dist_coeffs.npy` - Distortion coefficients
- `calibration.json` - Human-readable results

---

### `measure.py` (Step 2)
Measure real-world dimensions of objects using perspective projection.

```bash
python measure.py
```

**Important:** You must measure the distance from your camera to the object plane and set it using the `d` key.

**Controls:**
| Key | Action |
|-----|--------|
| Left Click | Place measurement point |
| Right Click | Clear all points |
| `r` | Toggle rectangle mode (width × height) |
| `d` | Set distance to object (mm) |
| `s` | Save screenshot |
| `c` | Cycle camera |
| `q` | Quit |

**Output:** Screenshots saved to `measure_data/` folder.

---

## Project Structure

```
week-2/
├── calibrate.py          # Camera calibration script
├── measure.py            # Real-world measurement script
├── camera_test.py        # Camera testing utility
├── calibration_data/     # Calibration output files
│   ├── camera_matrix.npy
│   ├── dist_coeffs.npy
│   ├── calibration.json
│   └── capture_*.png     # Calibration images
├── measure_data/         # Measurement screenshots
├── pattern.png           # Checkerboard pattern (if generated)
├── requirements.txt      # Python dependencies
├── venv/                 # Virtual environment
└── README.md
```

---

## Dependencies

- **OpenCV** (`opencv-python`) - Computer vision library
- **NumPy** - Numerical computations
- **Matplotlib** - Plotting (optional)

---

## Tips for Accurate Measurements

1. **Good Calibration**
   - Use 15-20 captures from different angles
   - Cover all areas of the frame, especially corners
   - Keep the checkerboard flat and well-lit
   - Aim for reprojection error < 0.5 pixels

2. **Accurate Distance**
   - Measure the distance from camera lens to object surface precisely
   - Keep the object parallel to the camera sensor plane
   - Objects at image center are most accurate

3. **Consistent Lighting**
   - Avoid shadows and reflections
   - Use even, diffused lighting
