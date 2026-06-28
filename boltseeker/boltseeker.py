#!/usr/bin/env python3
"""boltseeker.py - Detect and extract frames containing lightning from a video."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def detect_lightning(
    video_path: Path,
    brightness_threshold: int,
    pixel_ratio: float,
) -> tuple[list[int], int, float]:
    """Detect frames with lightning using the bright pixel ratio method.

    A frame is flagged when the fraction of pixels exceeding brightness_threshold
    is greater than pixel_ratio. This requires a meaningful area of bright pixels,
    which distinguishes a lightning bolt (large illuminated sky region) from
    small persistent light sources such as house lights or street lamps.

    Args:
        video_path: Path to the input video file.
        brightness_threshold: Minimum luminance value (0-254) to count a pixel as bright.
        pixel_ratio: Minimum fraction of bright pixels required to flag a frame.

    Returns:
        Tuple of (hit_frames, total_frames, fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video file '{video_path}'", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    hit_frames: list[int] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = float(np.sum(gray > brightness_threshold) / gray.size)

        if ratio > pixel_ratio:
            hit_frames.append(frame_idx)
            timestamp = frame_idx / fps
            print(
                f"  Frame {frame_idx:6d} ({timestamp:7.2f}s) — "
                f"bright pixel ratio: {ratio:.4f}"
            )

    cap.release()
    return hit_frames, total_frames, fps


def save_frames(
    video_path: Path,
    hit_frames: list[int],
    output_dir: Path,
    padding: int,
    total_frames: int,
) -> None:
    """Save detected frames (plus padding) as JPEG images.

    Args:
        video_path: Path to the input video file.
        hit_frames: List of frame indices flagged as lightning.
        output_dir: Directory to save images.
        padding: Number of extra frames to save on each side of a hit.
        total_frames: Total number of frames in the video, used to clamp indices.
    """
    if not hit_frames:
        print("No frames to save.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    hit_set = set(hit_frames)

    # Build deduplicated, sorted set of all frames to extract, clamped to valid range
    frames_to_save: set[int] = set()
    for idx in hit_frames:
        for offset in range(-padding, padding + 1):
            clamped = max(0, min(idx + offset, total_frames - 1))
            frames_to_save.add(clamped)

    cap = cv2.VideoCapture(str(video_path))

    saved = 0
    current_pos = -1
    for idx in sorted(frames_to_save):
        # Only seek when the target isn't the very next frame; sequential reads
        # are faster and avoid keyframe-alignment issues with compressed codecs.
        if idx != current_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        current_pos = idx
        if not ret:
            print(f"  Warning: could not read frame {idx}", file=sys.stderr)
            continue
        label = "lightning" if idx in hit_set else "padding"
        out_path = output_dir / f"frame_{idx:06d}_{label}.jpg"
        if not cv2.imwrite(str(out_path), frame):
            print(f"  Warning: failed to write '{out_path}'", file=sys.stderr)
        else:
            saved += 1

    cap.release()
    print(f"\nSaved {saved} frames to '{output_dir}/'")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="boltseeker",
        description="Detect and extract lightning frames from a storm video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("lightning_frames"),
        help="Directory to save extracted frames.",
    )
    parser.add_argument(
        "-b",
        "--brightness-threshold",
        type=int,
        default=180,
        help="Pixel luminance value (0-254) above which a pixel counts as bright. "
             "Lower this to catch dimmer bolts.",
    )
    parser.add_argument(
        "-r",
        "--pixel-ratio",
        type=float,
        default=0.005,
        help="Minimum fraction of bright pixels (0.0-1.0) required to flag a frame. "
             "Raise this if persistent light sources cause false positives.",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=1,
        help="Extra frames to save before/after each hit (>= 0). Ignored with --no-save.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Detect only; do not write any image files.",
    )

    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"file not found: '{args.video}'")
    if not (0 <= args.brightness_threshold <= 254):
        parser.error("--brightness-threshold must be between 0 and 254")
    if not (0.0 < args.pixel_ratio <= 1.0):
        parser.error("--pixel-ratio must be between 0.0 (exclusive) and 1.0")
    if args.padding < 0:
        parser.error("--padding must be >= 0")

    print(f"Scanning '{args.video}' for lightning frames...")
    print(
        f"Settings: brightness_threshold={args.brightness_threshold}, "
        f"pixel_ratio={args.pixel_ratio}, padding={args.padding}\n"
    )

    hit_frames, total_frames, fps = detect_lightning(
        video_path=args.video,
        brightness_threshold=args.brightness_threshold,
        pixel_ratio=args.pixel_ratio,
    )

    print(f"\nVideo: {total_frames} frames @ {fps:.3f} fps")
    print(f"Detected {len(hit_frames)} lightning frame(s).")

    if not args.no_save:
        save_frames(
            video_path=args.video,
            hit_frames=hit_frames,
            output_dir=args.output_dir,
            padding=args.padding,
            total_frames=total_frames,
        )
    else:
        print("--no-save set; skipping image extraction.")


if __name__ == "__main__":
    main()
