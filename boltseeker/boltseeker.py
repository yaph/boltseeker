#!/usr/bin/env python3
"""boltseeker.py - Detect and extract frames containing lightning from a video."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Dilation kernel used to merge nearby changed pixels into coherent blobs.
_DILATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def _blob_qualifies(
    contour: np.ndarray,
    gray: np.ndarray,
    min_blob_area: int,
    luminance_threshold: int,
    mask_buffer: np.ndarray,
) -> bool:
    """Return True if a contour meets the area and luminance criteria.

    Args:
        contour: OpenCV contour array.
        gray: Grayscale frame the contour was extracted from.
        min_blob_area: Minimum contour area in pixels.
        luminance_threshold: Minimum mean luminance within the contour mask.
        mask_buffer: Pre-allocated single-channel uint8 array the same size as gray,
            cleared and reused on each call to avoid repeated heap allocations.

    Returns:
        True if the blob qualifies as a potential lightning region.
    """
    if cv2.contourArea(contour) < min_blob_area:
        return False
    mask_buffer[:] = 0
    cv2.drawContours(mask_buffer, [contour], -1, 255, cv2.FILLED)
    region_mean = float(cv2.mean(gray, mask=mask_buffer)[0])
    return region_mean >= luminance_threshold


def detect_lightning(
    video_path: Path,
    diff_threshold: int,
    min_blob_area: int,
    luminance_threshold: int,
) -> tuple[list[int], int, float]:
    """Detect lightning frames using frame differencing, blob detection, and luminance.

    For each consecutive frame pair, an absolute difference image is computed and
    thresholded. The resulting binary mask is cleaned up with morphological dilation
    to merge nearby changed pixels, then contours are extracted. A frame is flagged
    when at least one contour (blob) meets both of the following conditions:
    - Its area exceeds min_blob_area (large coherent change, not noise or raindrops).
    - The mean luminance of the current frame within the contour mask exceeds
      luminance_threshold (the changed region is bright, not just dark cloud movement).

    Args:
        video_path: Path to the input video file.
        diff_threshold: Minimum per-pixel absolute difference (0-254) to count as
            changed. Higher values ignore gradual changes like slow cloud movement.
        min_blob_area: Minimum area in pixels of a connected changed region to
            consider. Filters out small transient changes like raindrops or noise.
        luminance_threshold: Minimum mean luminance (0-254) within the contour mask
            in the current frame. Filters out dark blobs caused by cloud movement.

    Returns:
        Tuple of (hit_frames, actual_frame_count, fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video file '{video_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print('Warning: could not determine frame rate; timestamps will show n/a.', file=sys.stderr)
        fps = 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask_buffer = np.zeros((height, width), dtype=np.uint8)

    hit_frames: list[int] = []
    frame_count = 0
    prev_gray: np.ndarray | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = frame_count
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, _DILATION_KERNEL, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if any(_blob_qualifies(c, gray, min_blob_area, luminance_threshold, mask_buffer) for c in contours):
                hit_frames.append(frame_idx)
                timestamp = f'{frame_idx / fps:7.2f}s' if fps > 0 else '    n/a'
                print(f'  Frame {frame_idx:6d} ({timestamp})')

        prev_gray = gray

    cap.release()

    if frame_count == 0:
        print('Warning: no frames could be read from the video.', file=sys.stderr)

    return hit_frames, frame_count, fps


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
        print('No frames to save.')
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
            print(f'  Warning: could not read frame {idx}', file=sys.stderr)
            continue
        label = 'lightning' if idx in hit_set else 'padding'
        out_path = output_dir / f'frame_{idx:06d}_{label}.jpg'
        if not cv2.imwrite(str(out_path), frame):
            print(f"  Warning: failed to write '{out_path}'", file=sys.stderr)
        else:
            saved += 1

    cap.release()
    print(f"\nSaved {saved} frames to '{output_dir}/'")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='boltseeker',
        description='Detect and extract lightning frames from a storm video.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('video', type=Path, help='Path to the input video file.')
    parser.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        default=Path('lightning_frames'),
        help='Directory to save extracted frames.',
    )
    parser.add_argument(
        '-d',
        '--diff-threshold',
        type=int,
        choices=range(255),
        default=25,
        metavar='DIFF_THRESHOLD',
        help='Minimum per-pixel absolute difference (0-254) between consecutive frames '
        'to count a pixel as changed. Raise this to ignore gradual changes '
        'like slow cloud movement or rain.',
    )
    parser.add_argument(
        '-a',
        '--min-blob-area',
        type=int,
        default=500,
        help='Minimum area in pixels of a connected changed region to consider. '
        'Raise this to filter out small transient changes like raindrops or noise.',
    )
    parser.add_argument(
        '-l',
        '--luminance-threshold',
        type=int,
        choices=range(255),
        default=80,
        metavar='LUMINANCE_THRESHOLD',
        help="Minimum mean luminance (0-254) within a blob's contour mask in the "
        'current frame. Filters out dark blobs caused by cloud or rain movement.',
    )
    parser.add_argument(
        '-p',
        '--padding',
        type=int,
        default=1,
        help='Extra frames to save before/after each hit (>= 0). Ignored with --no-save.',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Detect only; do not write any image files.',
    )

    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"file not found: '{args.video}'")
    if args.min_blob_area < 1:
        parser.error('--min-blob-area must be >= 1')
    if args.padding < 0:
        parser.error('--padding must be >= 0')

    print(f"Scanning '{args.video}' for lightning frames...")
    print(
        f'Settings: diff_threshold={args.diff_threshold}, '
        f'min_blob_area={args.min_blob_area}, '
        f'luminance_threshold={args.luminance_threshold}, '
        f'padding={args.padding}\n'
    )

    hit_frames, total_frames, fps = detect_lightning(
        video_path=args.video,
        diff_threshold=args.diff_threshold,
        min_blob_area=args.min_blob_area,
        luminance_threshold=args.luminance_threshold,
    )

    print(f'\nVideo: {total_frames} frames @ {fps:.3f} fps')
    print(f'Detected {len(hit_frames)} lightning frame(s).')

    if not args.no_save:
        save_frames(
            video_path=args.video,
            hit_frames=hit_frames,
            output_dir=args.output_dir,
            padding=args.padding,
            total_frames=total_frames,
        )
    else:
        print('--no-save set; skipping image extraction.')


if __name__ == '__main__':
    main()
