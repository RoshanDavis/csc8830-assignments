# Week 6: Temporally Adaptive Video Compression for Video-LLMs

A lightweight computer vision prototype that performs motion-based frame compression on video files, producing a condensed video ready for upload to any Video-LLM service.

## Problem Statement

Video-Language Models (Video-LLMs) face significant computational constraints when processing long videos. Most videos—especially security footage, lectures, or static scenes—contain large amounts of redundant frames where nothing meaningful changes. Sending all frames to a Vision-LLM:

- Exceeds token/frame limits
- Wastes API costs on static content
- Increases latency dramatically

## Solution

This pipeline uses traditional computer vision techniques to perform **temporally adaptive compression** before LLM processing:

1. **Frame Differencing**: Compare sequential frames using grayscale `cv2.absdiff()`
2. **Motion Detection**: Only retain frames where pixel changes exceed a threshold
3. **Video Re-encoding**: Stitch keyframes into a condensed video at 1 FPS
4. **Manual LLM Upload**: Upload the compressed video to ChatGPT, Google AI Studio, or other Video-LLM services

This approach is inspired by the "Long View" paper's concept of temporal compression for efficient Video-LLM processing.

## Setup

### Virtual Environment (Linux/macOS)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
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

## Usage

### Basic Usage

```bash
# Process video with default settings
python main.py -i test_video.mp4

# Custom motion threshold (lower = more sensitive)
python main.py -i test_video.mp4 -t 5.0

# Save individual keyframes as JPEGs
python main.py -i test_video.mp4 --save-frames
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | `test_video.mp4` | Path to input video file |
| `-t, --threshold` | `10.0` | Motion threshold for frame difference detection |
| `--save-frames` | `False` | Save keyframes as individual JPEGs |

### Example Output

```
==================================================
Processing: test_video.mp4
Motion threshold: 10.0
==================================================

==================================================
COMPRESSION STATISTICS
==================================================
  Total Original Frames:  1800
  Saved Keyframes:        47
  Compression:            97.4% frame reduction
==================================================

Encoding compressed video at 1 FPS...
Compressed video saved: compressed_video.mp4

==================================================
NEXT STEPS: Upload to a Video-LLM
==================================================
Upload 'compressed_video.mp4' to one of these services:
  - ChatGPT (GPT-4o):    https://chatgpt.com
  - Google AI Studio:    https://aistudio.google.com
  - MiniGPT4-Video:      https://huggingface.co/spaces/Vision-CAIR/MiniGPT4-video
==================================================

Done!
```

## Project Structure

```
week-6/
├── main.py              # Main compression script
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── compressed_video.mp4 # Output (generated)
└── keyframes/           # Optional JPEG output (generated)
    ├── keyframe_0000.jpg
    ├── keyframe_0001.jpg
    └── ...
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video I/O, frame processing, grayscale conversion |
| `numpy` | Array operations, mean difference calculation |
| `imageio` | Video encoding with ffmpeg backend |
| `imageio-ffmpeg` | FFmpeg codec support for MP4 output |

## Next Steps: LLM Analysis

After running the compression, upload `compressed_video.mp4` to one of these Video-LLM services:

| Service | URL | Notes |
|---------|-----|-------|
| ChatGPT (GPT-4o) | https://chatgpt.com | Requires Plus subscription for video |
| Google AI Studio | https://aistudio.google.com | Free tier available |
| MiniGPT4-Video | https://huggingface.co/spaces/Vision-CAIR/MiniGPT4-video | Free, may have queue |

**Suggested prompt:**
> "This is a temporally compressed video where redundant static frames were removed. Summarize the sequence of events shown."

## Notes

- **Threshold Tuning**: Lower thresholds (5-10) capture more motion; higher thresholds (20-30) only capture significant scene changes.
- **Memory Usage**: Keyframes are held in memory during processing. For very long videos, consider using `--save-frames` and processing in batches.
