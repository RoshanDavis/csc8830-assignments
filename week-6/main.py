"""
Temporally Adaptive Video Compression for Video-LLMs

This script performs motion-based frame compression on video files.
It extracts keyframes based on motion detection and re-encodes them
into a condensed video for manual upload to Video-LLM services.

Inspired by the "Long View" paper on temporal compression for Vision-Language Models.

Usage:
    python main.py -i test_video.mp4 -t 10.0
    python main.py --input video.mp4 --threshold 15.0 --save-frames
"""

import os
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import imageio


# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_VIDEO = os.path.join(SCRIPT_DIR, "compressed_video.mp4")
KEYFRAMES_DIR = os.path.join(SCRIPT_DIR, "keyframes")


def compress_video(video_path: str, threshold: float) -> Tuple[List[np.ndarray], int, int]:
    """
    Perform temporally adaptive compression on a video file.
    
    Compares sequential frames using grayscale pixel differences.
    Frames with motion exceeding the threshold are kept as keyframes.
    
    Args:
        video_path: Path to the input video file.
        threshold: Mean pixel difference threshold for motion detection.
        
    Returns:
        Tuple of (keyframes list, total frame count, keyframe count).
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    keyframes: List[np.ndarray] = []
    total_frames = 0
    last_keyframe_gray = None
    
    print(f"\n{'='*50}")
    print(f"Processing: {video_path}")
    print(f"Motion threshold: {threshold}")
    print(f"{'='*50}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Always save the first frame
        if last_keyframe_gray is None:
            keyframes.append(frame.copy())
            last_keyframe_gray = gray.copy()
            continue
        
        # Calculate absolute difference from last keyframe
        diff = cv2.absdiff(gray, last_keyframe_gray)
        mean_diff = np.mean(diff)
        
        # If motion exceeds threshold, save as keyframe
        if mean_diff > threshold:
            keyframes.append(frame.copy())
            last_keyframe_gray = gray.copy()
    
    cap.release()
    
    return keyframes, total_frames, len(keyframes)


def save_keyframes_to_disk(keyframes: List[np.ndarray], output_dir: str) -> List[str]:
    """
    Save keyframes as JPEG images to disk.
    
    Args:
        keyframes: List of BGR frames.
        output_dir: Directory to save frames.
        
    Returns:
        List of saved file paths.
    """
    # Clear and recreate output directory
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, frame in enumerate(keyframes):
        filename = f"keyframe_{i:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        saved_paths.append(filepath)
    
    return saved_paths


def encode_compressed_video(keyframes: List[np.ndarray], output_path: str, fps: int = 1) -> str:
    """
    Re-encode keyframes into a compressed video file.
    
    Args:
        keyframes: List of BGR frames from OpenCV.
        output_path: Path for the output video file.
        fps: Frames per second for output video (default: 1 FPS).
        
    Returns:
        Path to the created video file.
    """
    if not keyframes:
        raise ValueError("No keyframes to encode")
    
    # imageio expects RGB, OpenCV uses BGR
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
    
    for frame in keyframes:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)
    
    writer.close()
    
    return output_path


def print_statistics(total_frames: int, keyframe_count: int) -> None:
    """Print compression statistics to terminal."""
    if total_frames == 0:
        print("No frames processed.")
        return
        
    reduction = (1 - keyframe_count / total_frames) * 100
    
    print(f"\n{'='*50}")
    print("COMPRESSION STATISTICS")
    print(f"{'='*50}")
    print(f"  Total Original Frames:  {total_frames}")
    print(f"  Saved Keyframes:        {keyframe_count}")
    print(f"  Compression:            {reduction:.1f}% frame reduction")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Temporally adaptive video compression for Video-LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py -i video.mp4
    python main.py -i video.mp4 -t 15.0
    python main.py -i video.mp4 --save-frames
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='test_video.mp4',
        help='Path to input video file (default: test_video.mp4)'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=10.0,
        help='Motion threshold for frame difference (default: 10.0)'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save individual keyframes as JPEGs to keyframes/ folder'
    )
    
    args = parser.parse_args()
    
    # Resolve input path
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(SCRIPT_DIR, input_path)
    
    if not os.path.exists(input_path):
        print(f"Error: Video file not found: {input_path}")
        return 1
    
    # Part A: Adaptive compression
    keyframes, total_frames, keyframe_count = compress_video(input_path, args.threshold)
    
    if keyframe_count == 0:
        print("Error: No keyframes extracted. Try lowering the threshold.")
        return 1
    
    # Print compression statistics
    print_statistics(total_frames, keyframe_count)
    
    # Optionally save individual frames
    if args.save_frames:
        saved_paths = save_keyframes_to_disk(keyframes, KEYFRAMES_DIR)
        print(f"\nSaved {len(saved_paths)} keyframes to: {KEYFRAMES_DIR}/")
    
    # Part B: Re-encode compressed video
    print(f"\nEncoding compressed video at 1 FPS...")
    output_path = encode_compressed_video(keyframes, OUTPUT_VIDEO, fps=1)
    print(f"Compressed video saved: {output_path}")
    
    # Next steps message
    print(f"\n{'='*50}")
    print("NEXT STEPS: Upload to a Video-LLM")
    print(f"{'='*50}")
    print("Upload 'compressed_video.mp4' to one of these services:")
    print("  - ChatGPT (GPT-4o):    https://chatgpt.com")
    print("  - Google AI Studio:    https://aistudio.google.com")
    print("  - MiniGPT4-Video:      https://huggingface.co/spaces/Vision-CAIR/MiniGPT4-video")
    print(f"{'='*50}")
    print("\nDone!")
    
    return 0


if __name__ == "__main__":
    exit(main())
