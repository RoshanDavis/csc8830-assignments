# Week 5: Eye Blink Rate Detection

This project measures and compares eye blink rates during different activities using a webcam. It uses real-time eye tracking to detect blinks and calculates blinks per second (BPS) for comparison between watching movies versus reading documents.

## Theory

### Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is a real-time algorithm for detecting eye blinks based on facial landmark positions. It measures the ratio between the vertical and horizontal distances of the eye.

For each eye, 6 landmark points are used:
- **p1, p4**: Horizontal extremes (corners of the eye)
- **p2, p3**: Upper eyelid points
- **p5, p6**: Lower eyelid points

The EAR formula:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

**Behavior:**
- **Open eye**: EAR ≈ 0.25 - 0.30 (relatively constant)
- **Closed eye**: EAR < 0.21 (drops significantly)
- **Blink detection**: EAR falls below threshold for 2+ consecutive frames

### Blink Rate Research

Studies have shown that blink rates vary significantly based on activity:
- **Normal resting**: ~0.25-0.33 blinks/second (~15-20/min)
- **Reading/Focused work**: ~0.05-0.07 blinks/second (~3-4/min, reduced due to concentration)
- **Watching video**: ~0.17-0.25 blinks/second (~10-15/min)
- **Conversation**: ~0.33-0.43 blinks/second (~20-26/min)

This project allows you to measure your personal blink rates during different activities.

## Setup

### 1. Create Virtual Environment

```bash
# Navigate to week-5 directory
cd week-5

# Create virtual environment
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running a Session

```bash
# Movie watching session (1 minute)
python blink_detector.py --mode movie

# Document reading session (1 minute)
python blink_detector.py --mode reading

# Custom duration (30 seconds)
python blink_detector.py --mode movie --duration 30

# Use a specific camera
python blink_detector.py --mode reading --camera 1
```

### Comparing Results

After running multiple sessions, generate a comparison chart:

```bash
python blink_detector.py --compare
```

### Utility Commands

```bash
# List available cameras
python blink_detector.py --list-cameras
```

### Controls During Session

| Key | Action |
|-----|--------|
| `q` | Quit session early |
| `c` | Cycle through available cameras |
| `r` | Reset blink counter |

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--mode` | `-m` | Required | Session mode: `movie` or `reading` |
| `--duration` | `-d` | 60 | Session duration in seconds |
| `--camera` | `-c` | 0 | Camera index to use |
| `--compare` | | | Generate comparison chart |
| `--list-cameras` | | | List available cameras |

## Workflow

1. **Setup**: Position yourself in front of the webcam with good lighting
2. **Movie Session**: Run `--mode movie` while watching a video/movie on another screen
3. **Reading Session**: Run `--mode reading` while reading a document
4. **Compare**: Run `--compare` to see the difference in blink rates

**Tip**: Run multiple sessions of each type for more accurate averages.

## Project Structure

```
week-5/
├── blink_detector.py       # Main detection script
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── output/
    ├── sessions/
    │   └── blink_sessions.csv   # Session data log
    └── comparison_chart.png     # Generated comparison chart
```

## Output Files

### blink_sessions.csv

Records all session data with columns:
- `timestamp`: When the session was recorded
- `mode`: Session type (movie/reading)
- `duration_sec`: Actual session duration
- `blink_count`: Total blinks detected
- `blinks_per_second`: Calculated blink rate

### comparison_chart.png

A visualization showing:
- Bar chart comparing average blink rates between modes
- Scatter plot of individual session results
- Statistical summary (mean ± standard deviation)

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Webcam capture and image display |
| mediapipe | ≥0.10.0 | Face mesh landmark detection |
| numpy | ≥1.24.0 | Numerical computations |
| matplotlib | ≥3.7.0 | Chart generation |

## Troubleshooting

### No face detected
- Ensure adequate lighting on your face
- Position yourself closer to the camera
- Avoid backlighting (light should be in front of you)

### Inaccurate blink detection
- The default EAR threshold is 0.21; this may need adjustment for your eyes
- Ensure you're not wearing glasses that cause reflections
- Keep your head relatively still during the session

### Camera not found
- Run `python blink_detector.py --list-cameras` to see available cameras
- Try different camera indices with `--camera 1`, `--camera 2`, etc.

## References

- Soukupová, T., & Čech, J. (2016). "Real-Time Eye Blink Detection using Facial Landmarks"
- MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh.html
