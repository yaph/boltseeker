# boltseeker

boltseeker is a command-line tool that detects and extracts frames containing lightning bolts from storm videos. It uses frame differencing and blob detection to identify sudden, spatially coherent brightness changes while ignoring persistent light sources like house lights and street lamps that don't change between frames.

## How it works

For each consecutive frame pair, boltseeker computes an absolute difference image and thresholds it to isolate changed pixels. Morphological dilation merges nearby changes into coherent regions (blobs). A frame is flagged when at least one blob meets two conditions: its area exceeds a minimum size (ruling out raindrops and noise), and the mean luminance within the blob's contour in the current frame exceeds a minimum value (ruling out dark blobs from cloud or rain movement).

## Installation

```bash
pip install boltseeker
```

## Usage

<!-- START: DO NOT EDIT -->
```text
usage: boltseeker [-h] [-o OUTPUT_DIR] [-d DIFF_THRESHOLD] [-a MIN_BLOB_AREA] [-l LUMINANCE_THRESHOLD] [-p PADDING] [--no-save]
                  video

Detect and extract lightning frames from a storm video.

positional arguments:
  video                 Path to the input video file.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to save extracted frames. (default: lightning_frames)
  -d DIFF_THRESHOLD, --diff-threshold DIFF_THRESHOLD
                        Minimum per-pixel absolute difference (0-254) between consecutive frames to count a pixel as changed.
                        Raise this to ignore gradual changes like slow cloud movement or rain. (default: 25)
  -a MIN_BLOB_AREA, --min-blob-area MIN_BLOB_AREA
                        Minimum area in pixels of a connected changed region to consider. Raise this to filter out small
                        transient changes like raindrops or noise. (default: 500)
  -l LUMINANCE_THRESHOLD, --luminance-threshold LUMINANCE_THRESHOLD
                        Minimum mean luminance (0-254) within a blob's contour mask in the current frame. Filters out dark blobs
                        caused by cloud or rain movement. (default: 80)
  -p PADDING, --padding PADDING
                        Extra frames to save before/after each hit (>= 0). Ignored with --no-save. (default: 1)
  --no-save             Detect only; do not write any image files. (default: False)

```
<!-- END: DO NOT EDIT -->

### Examples

```bash
# Basic usage
boltseeker storm.mp4

# Save to a custom directory
boltseeker storm.mp4 -o ~/bolts

# Detect only, no output files
boltseeker storm.mp4 --no-save

# More sensitive detection
boltseeker storm.mp4 -d 15 -l 60

# Less sensitive detection
boltseeker storm.mp4 -d 40 -a 1000
```

## Output

Detected frames are saved as JPEG files named `frame_NNNNNN_lightning.jpg`. Padding frames saved around each hit are named `frame_NNNNNN_padding.jpg`. The terminal output lists each flagged frame with its index and timestamp as the video is scanned.

## Tuning

Run with `--no-save` first to see which frames are flagged and at what timestamps before committing to saving files. The three detection parameters are independent:

- If dark frames from cloud movement are being saved, raise `--luminance-threshold`.
- If small rain or noise artefacts trigger false positives, raise `--min-blob-area`.
- If slow-moving clouds cause false positives, raise `--diff-threshold`.
- If faint or distant bolts are being missed, lower `--diff-threshold` or `--luminance-threshold`.
