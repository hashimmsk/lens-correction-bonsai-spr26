#!/usr/bin/env python3
"""
Undistort images or videos using a saved fisheye calibration.

Loads K and D from the .npz produced by calibrate_camera.py, then
corrects barrel/fisheye distortion with cv2.fisheye remap.

Usage examples:

    # Single video
    python undistort_media.py calibration.npz "data/Wide lens video.MP4" \\
        -o output/corrected.mp4

    # Multiple videos into an output directory
    python undistort_media.py calibration.npz "data/Wide lens angle 3.MP4" \\
        "data/Wide lens angle 4.MP4" -o output/

    # Adjust balance (0=tight crop, 1=keep full FOV with black borders)
    python undistort_media.py calibration.npz input.mp4 -o out.mp4 --balance 0.3
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def load_calibration(path):
    """Load K, D, image_size from a .npz calibration file."""
    data = np.load(path)
    K = data["K"]
    D = data["D"]
    image_size = (int(data["image_size"][0]), int(data["image_size"][1]))
    rms = float(data["rms"])
    print(f"Loaded calibration from {path}")
    print(f"  Calibration image size : {image_size[0]}x{image_size[1]}")
    print(f"  RMS reprojection error : {rms:.4f}")
    print(f"  D = {D.flatten()}")
    return K, D, image_size


# ---------------------------------------------------------------------------
# Undistortion map builder
# ---------------------------------------------------------------------------

def build_undistort_maps(K, D, image_size, balance=0.0, fov_scale=1.0):
    """
    Compute undistortion + rectification remap arrays.

    balance: 0.0 crops out all black borders, 1.0 retains full FOV.
    fov_scale: focal-length multiplier on the new camera matrix (>1 widens).
    """
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, np.eye(3), balance=balance, fov_scale=fov_scale,
    )
    # Guard against degenerate results from estimateNew (known OpenCV issue
    # when high-order distortion terms are active)
    if new_K[0, 0] < 1.0 or new_K[1, 1] < 1.0:
        new_K = K.copy()
        crop = 1.0 - 0.3 * (1.0 - balance)
        new_K[0, 0] *= crop
        new_K[1, 1] *= crop

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2,
    )
    return map1, map2, new_K


def remap_frame(frame, map1, map2):
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------
# File-type helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


def is_image(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _maps_for_resolution(K, D, cal_size, actual_size, balance, fov_scale):
    """Return (map1, map2) scaled to actual_size if it differs from cal_size."""
    if actual_size == cal_size:
        return build_undistort_maps(K, D, cal_size, balance, fov_scale)[:2]

    sx = actual_size[0] / cal_size[0]
    sy = actual_size[1] / cal_size[1]
    K_s = K.copy()
    K_s[0, :] *= sx
    K_s[1, :] *= sy
    m1, m2, _ = build_undistort_maps(K_s, D, actual_size, balance, fov_scale)
    print(f"  [INFO] Rescaled calibration from {cal_size[0]}x{cal_size[1]} "
          f"to {actual_size[0]}x{actual_size[1]}")
    return m1, m2


def process_image(src, dst, K, D, cal_size, balance, fov_scale):
    img = cv2.imread(src)
    if img is None:
        print(f"  [ERROR] Cannot read {src}")
        return
    h, w = img.shape[:2]
    m1, m2 = _maps_for_resolution(K, D, cal_size, (w, h), balance, fov_scale)
    corrected = remap_frame(img, m1, m2)
    cv2.imwrite(dst, corrected)
    print(f"  {src} -> {dst}")


def process_video(src, dst, K, D, cal_size, balance, fov_scale, codec="mp4v"):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {src}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    m1, m2 = _maps_for_resolution(K, D, cal_size, (w, h), balance, fov_scale)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(dst, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"  [ERROR] Cannot create output {dst}")
        cap.release()
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(remap_frame(frame, m1, m2))
        count += 1
        if count % 50 == 0 or count == total:
            pct = count / total * 100 if total else 0
            print(f"  [{count}/{total}] {pct:.0f}%", end="\r")

    cap.release()
    writer.release()
    print(f"  {src} -> {dst}  ({count} frames)         ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Undistort images/videos using a fisheye calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "calibration",
        help="Path to .npz calibration file from calibrate_camera.py.",
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Input image(s) or video(s) to undistort.",
    )
    parser.add_argument(
        "-o", "--output", default="output",
        help="Output file (single input) or directory (multiple inputs).",
    )
    parser.add_argument(
        "--balance", type=float, default=0.0,
        help="0.0 = crop black borders, 1.0 = keep full FOV.",
    )
    parser.add_argument(
        "--fov-scale", type=float, default=1.0,
        help="Focal-length scale for the new camera (>1 widens FOV).",
    )
    parser.add_argument(
        "--codec", type=str, default="mp4v",
        help="FourCC codec for output videos.",
    )
    args = parser.parse_args()

    K, D, cal_size = load_calibration(args.calibration)

    multi = len(args.inputs) > 1
    if multi or os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    for src in args.inputs:
        if not os.path.isfile(src):
            print(f"  [WARN] Skipping {src} (not found)")
            continue

        # Build output path
        if multi or os.path.isdir(args.output):
            base = os.path.splitext(os.path.basename(src))[0]
            ext = os.path.splitext(src)[1]
            dst = os.path.join(args.output, f"{base}_corrected{ext}")
        else:
            dst = args.output

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

        if is_image(src):
            process_image(src, dst, K, D, cal_size, args.balance, args.fov_scale)
        elif is_video(src):
            process_video(src, dst, K, D, cal_size, args.balance, args.fov_scale,
                          args.codec)
        else:
            print(f"  [WARN] Unknown file type: {src}")

    print("\nDone.")


if __name__ == "__main__":
    main()
