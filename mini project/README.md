# Mini Project: Blink Rate and Facial Dimension Estimation

This project processes your study videos under `videos/` and estimates:

1. Blink rate (blinks per second) across the full study duration.
2. Approximate dimensions of eyes, face, nose, and mouth using iris-based scale estimation.

## Method Summary

### A) Blink Rate Estimation

The program uses MediaPipe Face Mesh to extract eye landmarks and computes Eye Aspect Ratio (EAR):

`EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)`

A blink is counted when EAR stays below a threshold for a minimum number of consecutive frames.

### B) Dimension Estimation (Approximate Real Units)

For each detected face frame:

1. Pixel distances are measured between landmark pairs for each facial feature.
2. A pixel-to-mm scale is estimated from iris diameter landmarks.
3. Distances are converted to mm/cm and aggregated.

Assumption: average iris diameter is approximately `11.8 mm`, so outputs are approximate.

## Project Structure

- `main.py`: CLI entrypoint for batch processing.
- `analyzer.py`: Core frame-by-frame analysis pipeline.
- `metrics.py`: EAR and geometry helpers.
- `scaling.py`: Iris-based scale conversion.
- `plotting.py`: Plot generation.
- `io_utils.py`: CSV and output helpers.
- `config.py`: Landmark indices and defaults.
- `videos/`: Input video files (already provided).
- `output/`: Generated CSVs and plots.

## Setup (Windows PowerShell)

From `c:\Vault\Projects\csc8830-assignments\mini project`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script execution is blocked in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Default: process all videos in `videos/`.

```powershell
python main.py
```

Useful options:

```powershell
python main.py --ear-threshold 0.21 --consec-frames 2 --frame-step 1
python main.py --max-videos 3
python main.py --videos-dir .\videos --output-dir .\output
```

## Output Files

Generated under `output/`:

- `video_summary.csv`: one row per video with duration, blinks, blinks/sec, and average dimensions.
- `global_per_second.csv`: per-second blink and measurement aggregates across all videos.
- `per_video/*_per_second.csv`: per-second metrics for each video.
- `plots/blink_rate_timeline.png`: blink-rate trend over cumulative study time.
- `plots/dimension_summary.png`: average estimated feature dimensions.

## Notes and Limitations

- Missing-face frames are skipped and do not stop processing.
- Real-world dimension outputs are approximate due to iris-size scaling assumptions.
- Faster runtime can be achieved with `--frame-step` greater than 1, but blink sensitivity may drop.
