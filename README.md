# Fisheye Lens Correction Pipeline

Calibrate a fisheye/wide-angle camera from checkerboard videos and undistort
images or videos into a corrected linear (pinhole) view.

## Requirements

```
pip install -r requirements.txt
```

Only **Python 3.8+** and **OpenCV** (with the fisheye module) are needed.

## Quick Start

### 1. Calibrate

Run calibration on the wide-lens checkerboard videos:

```bash
python calibrate_camera.py \
    "data/input/Wide lens video.MP4" \
    "data/input/Wide lens angle 3.MP4" \
    "data/input/Wide lens angle 4.MP4" \
    "data/input/Wide lens angle 5.MP4" \
    "data/input/Wide lens angle 6.MP4" \
    "data/input/Wide lens angle 7.MP4" \
    "data/input/Wide lens angle 8.MP4" \
    "data/input/Wide lens angle 9.MP4" \
    --board-width 9 --board-height 6 \
    --square-size 25.0 \
    --frame-skip 15
```

This produces:
- `data/calibration/calibration.npz` — camera matrix K, distortion coefficients D, metadata
- `data/calibration/` — annotated corner-detection images and a sample undistorted frame

### 2. Undistort

Correct a single video:

```bash
python undistort_media.py data/calibration/calibration.npz \
    "data/input/Wide lens video.MP4" \
    -o "data/output/Wide lens video_corrected.mp4"
```

Correct multiple files into the output directory:

```bash
python undistort_media.py data/calibration/calibration.npz \
    "data/input/Wide lens angle 3.MP4" \
    "data/input/Wide lens angle 4.MP4" \
    -o data/output/
```

Correct a still image:

```bash
python undistort_media.py data/calibration/calibration.npz photo.jpg -o photo_corrected.jpg
```

### 3. Validate

Compare corrected output against the linear-lens ground truth:

```
data/output/Wide lens angle 5_corrected.MP4  ←→  data/ground_truth/Linear lens angle 5.MP4
```

## Calibration Options

| Flag | Default | Description |
|---|---|---|
| `--board-width` | 9 | Inner corners along checkerboard width |
| `--board-height` | 6 | Inner corners along checkerboard height |
| `--square-size` | 25.0 | Square side length in mm |
| `--frame-skip` | 15 | Process every Nth frame |
| `--max-frames` | all | Max frames per video |
| `--scale` | 1.0 | Resize factor before detection (e.g. 0.5) |
| `--distortion-params` | 2 | Active distortion coefficients: 2, 3, or 4 |
| `--save-samples` | 8 | Annotated sample frames to save |
| `--output` | data/calibration/calibration.npz | Output calibration file |

## Undistortion Options

| Flag | Default | Description |
|---|---|---|
| `--balance` | 0.0 | 0 = crop black borders, 1 = keep full FOV |
| `--fov-scale` | 1.0 | New-camera focal length scale (>1 widens) |
| `--codec` | mp4v | FourCC codec for output videos |

## How It Works

1. **Corner detection** — `cv2.findChessboardCorners` locates the 9x6
   inner-corner grid, then `cv2.cornerSubPix` refines to sub-pixel accuracy.

2. **Fisheye calibration** — `cv2.fisheye.calibrate` fits the Kannala-Brandt
   equidistant projection model. By default, only k1 and k2 are active
   (k3 = k4 = 0) for a stable, well-conditioned solution.

3. **Undistortion** — `cv2.fisheye.initUndistortRectifyMap` builds a pixel
   remap table, then `cv2.remap` warps each frame into a corrected pinhole
   view. The `--balance` flag controls how much of the original FOV is
   retained vs. cropped.

### Why fisheye calibration instead of standard?

The standard OpenCV distortion model (`cv2.calibrateCamera`) uses a polynomial
approximation that breaks down at wide angles. The fisheye module implements
the equidistant projection model, which correctly handles the geometric
properties of wide-angle and fisheye lenses.

## Resolution Handling

The calibration was computed at 1920x1080. When undistorting files at different
resolutions (e.g. the 3840x2160 videos), the scripts automatically rescale
the camera matrix to match.

## Calibration Results (current dataset)

```
Valid frames        : 159 / 159  (100%)
Image size          : 1920 x 1080
RMS reprojection    : 1.23 pixels
Camera matrix K:
  fx = 1067.66   fy = 1081.60
  cx =  959.04   cy =  541.82
Distortion D:
  k1 = +0.0221   k2 = +0.1668
  k3 =  0.0000   k4 =  0.0000
```

## File Structure

```
├── calibrate_camera.py           # Calibration script
├── undistort_media.py             # Undistortion script
├── requirements.txt               # Python dependencies
├── README.md
└── data/
    ├── input/                     # Raw wide-lens videos (to be corrected)
    │   ├── Wide lens video.MP4
    │   └── Wide lens angle 2-9.MP4
    ├── ground_truth/              # Linear-lens videos (reference)
    │   ├── Linear Lens video.MP4
    │   └── Linear lens angle 2-9.MP4
    ├── output/                    # Corrected wide-lens videos
    │   └── *_corrected.MP4
    └── calibration/               # Calibration artifacts
        ├── calibration.npz        # Camera matrix + distortion coefficients
        ├── undistorted_sample.jpg # Quick visual check
        ├── corner_detection/      # Proof corners were detected correctly
        │   └── corners_*.jpg
        └── comparison/            # Side-by-side validation images
            └── compare_*.jpg
```
