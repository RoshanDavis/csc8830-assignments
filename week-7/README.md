# Week 7: Optical Flow & Motion Tracking

Computes dense optical flow on video, visualizes motion as color-coded video, and validates the Lucas-Kanade tracking equations against actual pixel locations.

## Problem Statement

Given two videos with motion, this assignment:

**Part A** — Computes dense optical flow (Farnebäck) and produces a side-by-side visualization video showing the original frames alongside the color-coded flow field. Explains what information optical flow reveals.

**Part B** — Derives the Lucas-Kanade motion tracking equations from the brightness constancy assumption, derives bilinear interpolation, and validates the theoretical tracking predictions against actual pixel locations on consecutive frame pairs.

## Setup

### Virtual Environment (Linux/macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Virtual Environment (Windows)

**PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Command Prompt:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Prerequisites

Place **two video files** (`.mp4` or similar) in the `week-7/` directory. The videos should contain visible motion (e.g., sports, traffic, people walking) with at least 30 seconds of motion each. You can name the files anything — paths are passed via command-line arguments.

## Usage

### Part A: Optical Flow

```bash
# Compute and visualize optical flow for both videos (30s each)
python optical_flow.py -v1 video1.mp4 -v2 video2.mp4

# Process a custom duration (e.g., 45 seconds)
python optical_flow.py -v1 video1.mp4 -v2 video2.mp4 --duration 45
```

**Output:** `output/flow_video1.mp4` and `output/flow_video2.mp4` — side-by-side videos of original frames and color-coded optical flow.

#### Part A Options

| Option | Default | Description |
|--------|---------|-------------|
| `-v1, --video1` | *(required)* | Path to first video |
| `-v2, --video2` | *(required)* | Path to second video |
| `-d, --duration` | `30` | Seconds of video to process |

### Part B: Motion Tracking

```bash
# Run tracking derivation and validation on both videos
python motion_tracking.py -v1 video1.mp4 -v2 video2.mp4

# Pick a specific frame index for the consecutive pair
python motion_tracking.py -v1 video1.mp4 -v2 video2.mp4 --frame 100
```

**Output:**
- Mathematical derivations printed to the console
- Tracking results table (predicted vs. actual positions, pixel error)
- Annotated images saved to `output/tracking_video1.png` and `output/tracking_video2.png`

#### Part B Options

| Option | Default | Description |
|--------|---------|-------------|
| `-v1, --video1` | *(required)* | Path to first video |
| `-v2, --video2` | *(required)* | Path to second video |
| `-f, --frame` | `60` | Frame index for the consecutive pair |

## Project Structure

```
week-7/
├── optical_flow.py      # Part A: Optical flow computation & visualization
├── motion_tracking.py   # Part B: Tracking derivation & validation
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── video1.mp4           # Your first input video (user-provided)
├── video2.mp4           # Your second input video (user-provided)
└── output/              # Generated outputs
    ├── flow_video1.mp4
    ├── flow_video2.mp4
    ├── tracking_video1.png
    └── tracking_video2.png
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Optical flow, feature detection, template matching |
| `numpy` | Array operations, mathematical computation |
| `matplotlib` | Visualization support |
| `imageio` | Video encoding with ffmpeg backend |
| `imageio-ffmpeg` | FFmpeg codec support for MP4 output |

## How Optical Flow Visualization Works

The HSV color coding used in the flow videos encodes:

| Channel | Encodes | Interpretation |
|---------|---------|----------------|
| **Hue** | Direction of motion | Red=right, Cyan=left, Green=down, Magenta=up |
| **Saturation** | Fixed (255) | Always full saturation |
| **Value** | Speed of motion | Bright=fast, Dark=static |

## Notes

- **Frame Selection:** For Part B, pick a frame index (via `--frame`) in a region with motion for better results. The default (frame 60) assumes the video has motion early on.
- **Feature Count:** Part B tracks up to 20 features per frame pair. Features are selected automatically using Shi-Tomasi corner detection.
- **Sub-pixel Accuracy:** The Lucas-Kanade method naturally produces sub-pixel flow estimates, which is why bilinear interpolation is relevant for reading image values at non-integer positions.
