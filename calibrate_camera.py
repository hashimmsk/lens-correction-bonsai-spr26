#!/usr/bin/env python3
"""
Fisheye Camera Calibration from Checkerboard Videos

Extracts frames from one or more checkerboard calibration videos,
detects inner corners, and computes fisheye camera intrinsics and
distortion coefficients using the Kannala-Brandt model
(cv2.fisheye.calibrate). Saves results in a reusable .npz file.

Usage examples:

    # Calibrate from wide-lens checkerboard videos
    python calibrate_camera.py "data/input/Wide lens video.MP4" \\
        "data/input/Wide lens angle 3.MP4" "data/input/Wide lens angle 4.MP4" \\
        --board-width 9 --board-height 6 --square-size 25.0 \\
        --output data/calibration/calibration.npz --sample-dir data/calibration

    # Use all 4 distortion params (less stable, sometimes needed)
    python calibrate_camera.py "data/input/Wide lens video.MP4" --distortion-params 4
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path, frame_skip=10, max_frames=None, scale=1.0):
    """Yield (frame_index, frame) tuples from a video at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return

    idx = 0
    yielded = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            if scale != 1.0:
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            yield idx, frame
            yielded += 1
            if max_frames and yielded >= max_frames:
                break
        idx += 1

    cap.release()


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------

CHECKER_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SUBPIX_WINSIZE = (5, 5)


def detect_corners(frame, board_size):
    """
    Detect checkerboard corners with sub-pixel refinement.
    Returns (success, corners) where corners is Nx1x2 float32 or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size, CHECKER_FLAGS)
    if not found:
        return False, None

    corners = cv2.cornerSubPix(gray, corners, SUBPIX_WINSIZE, (-1, -1), SUBPIX_CRITERIA)
    return True, corners


# ---------------------------------------------------------------------------
# Fisheye calibration
# ---------------------------------------------------------------------------

def calibrate_fisheye(obj_points_list, img_points_list, image_size,
                      n_distortion_params=2):
    """
    Run cv2.fisheye.calibrate.

    n_distortion_params: 2 fixes k3=k4=0 (more stable, recommended default).
                         4 uses all four Kannala-Brandt coefficients.
    """
    N = len(obj_points_list)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    if n_distortion_params <= 2:
        calibration_flags |= cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
    elif n_distortion_params == 3:
        calibration_flags |= cv2.fisheye.CALIB_FIX_K4

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    obj_pts = [p.reshape(-1, 1, 3).astype(np.float64) for p in obj_points_list]
    img_pts = [p.reshape(-1, 1, 2).astype(np.float64) for p in img_points_list]

    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]

    try:
        calibration_flags_try = calibration_flags | cv2.fisheye.CALIB_CHECK_COND
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_pts, img_pts, image_size, K, D, rvecs, tvecs,
            calibration_flags_try, criteria,
        )
    except cv2.error as e:
        print(f"  [WARN] CHECK_COND failed ({e}); retrying without it ...")
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_pts, img_pts, image_size, K, D, rvecs, tvecs,
            calibration_flags, criteria,
        )

    return K, D, rvecs, tvecs, rms


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(K, D, rms, n_valid, n_total, image_size):
    active_d = [i for i in range(4) if D[i, 0] != 0.0]
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Valid / examined frames       : {n_valid} / {n_total}")
    print(f"  Image size (w x h)            : {image_size[0]} x {image_size[1]}")
    print(f"  RMS reprojection error         : {rms:.6f} pixels")
    print(f"\n  Camera matrix K:")
    print(f"    fx = {K[0,0]:.2f}")
    print(f"    fy = {K[1,1]:.2f}")
    print(f"    cx = {K[0,2]:.2f}")
    print(f"    cy = {K[1,2]:.2f}")
    print(f"\n  Distortion coefficients D (Kannala-Brandt):")
    for i in range(4):
        marker = "" if i in active_d else "  (fixed)"
        print(f"    k{i+1} = {D[i,0]:+.6f}{marker}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_calibration(path, K, D, rms, image_size, board_size, square_size):
    np.savez(
        path,
        K=K, D=D, rms=rms,
        image_size=np.array(image_size),
        board_size=np.array(board_size),
        square_size=np.array(square_size),
    )
    print(f"  Calibration saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fisheye camera calibration from checkerboard videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "videos", nargs="+",
        help="Path(s) to checkerboard calibration video(s).",
    )
    parser.add_argument(
        "--board-width", type=int, default=9,
        help="Inner corner count along the checkerboard width.",
    )
    parser.add_argument(
        "--board-height", type=int, default=6,
        help="Inner corner count along the checkerboard height.",
    )
    parser.add_argument(
        "--square-size", type=float, default=25.0,
        help="Side length of one checkerboard square in mm.",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=15,
        help="Process every Nth frame from each video.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to extract per video (default: all).",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Resize factor applied before detection (e.g. 0.5 = half).",
    )
    parser.add_argument(
        "--distortion-params", type=int, default=2, choices=[2, 3, 4],
        help="Number of active distortion coefficients (2 recommended).",
    )
    parser.add_argument(
        "--output", type=str, default="data/calibration/calibration.npz",
        help="Output path for the calibration file.",
    )
    parser.add_argument(
        "--save-samples", type=int, default=8,
        help="Number of annotated sample frames to save (0 to disable).",
    )
    parser.add_argument(
        "--sample-dir", type=str, default="data/calibration",
        help="Directory to save annotated sample frames.",
    )
    args = parser.parse_args()

    board_size = (args.board_width, args.board_height)
    n_corners = board_size[0] * board_size[1]

    objp = np.zeros((n_corners, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= args.square_size

    obj_points_list = []
    img_points_list = []
    sample_frames = []      # (video_path, frame_idx, original_frame, corners)
    image_size = None
    n_total = 0
    skipped_res = 0

    for vpath in args.videos:
        print(f"\nProcessing: {vpath}")
        if not os.path.isfile(vpath):
            print(f"  [ERROR] File not found: {vpath}")
            continue

        for fidx, frame in extract_frames(vpath, args.frame_skip,
                                           args.max_frames, args.scale):
            n_total += 1
            h, w = frame.shape[:2]

            if image_size is None:
                image_size = (w, h)
            elif (w, h) != image_size:
                skipped_res += 1
                continue

            found, corners = detect_corners(frame, board_size)
            if found:
                obj_points_list.append(objp)
                img_points_list.append(corners)
                if len(sample_frames) < max(args.save_samples, 1):
                    sample_frames.append((vpath, fidx, frame.copy(), corners))
                status = "OK"
            else:
                status = "no corners"
            print(f"  frame {fidx:>5d}  [{status}]  (valid: {len(obj_points_list)})")

    n_valid = len(obj_points_list)
    print(f"\nCorner detection complete: {n_valid} valid / {n_total} examined"
          + (f" ({skipped_res} skipped due to resolution mismatch)" if skipped_res else ""))

    if n_valid < 6:
        print("[ERROR] At least 6 valid frames required for calibration. Aborting.")
        sys.exit(1)

    # Save annotated corner images
    if args.save_samples > 0 and sample_frames:
        os.makedirs(args.sample_dir, exist_ok=True)
        for i, (vp, fidx, img, corners) in enumerate(sample_frames[:args.save_samples]):
            annotated = img.copy()
            cv2.drawChessboardCorners(annotated, board_size, corners, True)
            out = os.path.join(args.sample_dir, f"corners_{i:03d}_frame{fidx}.jpg")
            cv2.imwrite(out, annotated)
        print(f"  Saved {min(len(sample_frames), args.save_samples)} annotated frames "
              f"to {args.sample_dir}/")

    # Run calibration
    print(f"\nRunning fisheye calibration ({args.distortion_params}-param model) ...")
    t0 = time.time()
    K, D, rvecs, tvecs, rms = calibrate_fisheye(
        obj_points_list, img_points_list, image_size, args.distortion_params)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    print_diagnostics(K, D, rms, n_valid, n_total, image_size)
    save_calibration(args.output, K, D, rms, image_size, board_size, args.square_size)

    # Quick visual validation: undistort one sample frame
    if sample_frames:
        os.makedirs(args.sample_dir, exist_ok=True)
        sample_img = sample_frames[0][2]

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, image_size, np.eye(3), balance=0.0)
        # Fallback if estimateNew returns degenerate matrix
        if new_K[0, 0] < 1.0 or new_K[1, 1] < 1.0:
            new_K = K.copy()
            new_K[0, 0] *= 0.7
            new_K[1, 1] *= 0.7

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2)
        undistorted = cv2.remap(sample_img, map1, map2, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        out_path = os.path.join(args.sample_dir, "undistorted_sample.jpg")
        cv2.imwrite(out_path, undistorted)
        print(f"  Sample undistorted frame -> {out_path}")

    print("\nDone. Use undistort_media.py to correct images or videos.\n")


if __name__ == "__main__":
    main()
