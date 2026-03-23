"""CLI entrypoint for blink-rate and facial dimension estimation from videos."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from typing import List

from analyzer import (
    AnalysisSettings,
    aggregate_global_summaries,
    analyze_video,
    ensure_model_downloaded,
    list_video_files,
)
from config import DEFAULT_OUTPUT_DIR, DEFAULT_VIDEOS_DIR, VIDEO_EXTENSIONS
from io_utils import ensure_output_dirs, write_csv
from plotting import plot_blink_rate_timeline, plot_dimension_summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate blink rate (blinks/sec) and approximate eye/face/nose/mouth dimensions "
            "from all videos in a folder."
        )
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=DEFAULT_VIDEOS_DIR,
        help=f"Directory containing video files (default: {DEFAULT_VIDEOS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSVs and plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.4,
        help="Blendshape blink score threshold used for blink detection (default: 0.4)",
    )
    parser.add_argument(
        "--consec-frames",
        type=int,
        default=2,
        help="Consecutive blink score frames required to count a blink (default: 2)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional cap on number of videos processed (0 = no cap)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel processes for video processing",
    )
    return parser.parse_args()


def process_single_video(video_path: str, settings: AnalysisSettings) -> dict:
    print(f"Starting: {os.path.basename(video_path)}")
    try:
        result = analyze_video(video_path, settings)
        print(f"Finished: {os.path.basename(video_path)}")
        return result
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {e}")
        return {}


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full analysis pipeline and export all artifacts."""
    output_paths = ensure_output_dirs(args.output_dir)

    videos = list_video_files(args.videos_dir, VIDEO_EXTENSIONS)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    if not videos:
        raise FileNotFoundError(f"No supported video files found in: {args.videos_dir}")

    ensure_model_downloaded()

    settings = AnalysisSettings(
        ear_threshold=args.ear_threshold,
        consec_frames=max(1, args.consec_frames),
        frame_step=max(1, args.frame_step),
    )

    video_summaries: List[dict] = []
    global_per_second_rows: List[dict] = []

    cumulative_second_offset = 0

    results = []
    print(f"Processing {len(videos)} videos with {args.num_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_video, v, settings): v for v in videos}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: x["summary"]["video_file"])

    for result in results:
        summary = result["summary"]
        per_second_rows = result["per_second_rows"]

        video_summaries.append(summary)

        per_video_csv_path = os.path.join(
            output_paths["per_video"],
            f"{os.path.splitext(summary['video_file'])[0]}_per_second.csv",
        )
        if per_second_rows:
            write_csv(per_video_csv_path, per_second_rows, fieldnames=list(per_second_rows[0].keys()))

        for row in per_second_rows:
            merged = dict(row)
            merged["cumulative_second"] = int(row["second"]) + cumulative_second_offset
            global_per_second_rows.append(merged)

        cumulative_second_offset += int(round(float(summary["duration_sec"])))

    if video_summaries:
        summary_csv_path = os.path.join(output_paths["base"], "video_summary.csv")
        write_csv(summary_csv_path, video_summaries, fieldnames=list(video_summaries[0].keys()))

    if global_per_second_rows:
        per_second_csv_path = os.path.join(output_paths["base"], "global_per_second.csv")
        write_csv(per_second_csv_path, global_per_second_rows, fieldnames=list(global_per_second_rows[0].keys()))

        blink_plot_path = os.path.join(output_paths["plots"], "blink_rate_timeline.png")
        plot_blink_rate_timeline(global_per_second_rows, blink_plot_path)

    global_dimension_means = aggregate_global_summaries(video_summaries)
    if global_dimension_means:
        dimension_plot_path = os.path.join(output_paths["plots"], "dimension_summary.png")
        plot_dimension_summary(global_dimension_means, dimension_plot_path)

    total_duration = sum(float(row.get("duration_sec", 0)) for row in video_summaries)
    total_blinks = sum(int(row.get("total_blinks", 0)) for row in video_summaries)
    overall_bps = (total_blinks / total_duration) if total_duration > 0 else 0.0

    print("\nRun complete")
    print(f"Videos processed: {len(video_summaries)}")
    print(f"Total duration (sec): {total_duration:.2f}")
    print(f"Total blinks: {total_blinks}")
    print(f"Overall blink rate (blinks/sec): {overall_bps:.4f}")
    print(f"Outputs: {output_paths['base']}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
