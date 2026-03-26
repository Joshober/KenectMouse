import argparse
import time

import cv2
import freenect
import numpy as np
import pyautogui


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def get_depth():
    frame = freenect.sync_get_depth()
    if frame is None:
        return None
    depth, _ = frame
    if depth is None:
        return None
    return depth.astype(np.uint16)


def get_hand_position(depth, min_area, depth_band):
    # Ignore invalid values and very far points.
    valid = np.where((depth > 0) & (depth < 2047), depth, 0).astype(np.uint16)
    if np.count_nonzero(valid) == 0:
        return None

    # Use a near-depth band so background doesn't dominate the "closest point".
    nearest = np.percentile(valid[valid > 0], 5)
    near_band_max = min(nearest + depth_band, 2047)
    mask = np.where((valid >= nearest) & (valid <= near_band_max), 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None

    y = int(moments["m01"] / moments["m00"])
    x = int(moments["m10"] / moments["m00"])
    return x, y


def run(alpha, click_threshold, min_area, enable_click, depth_band, flip_y, edge_margin):
    screen_w, screen_h = pyautogui.size()
    prev_x, prev_y = 0, 0
    prev_depth = None

    # Map Kinect 640x480 to a slightly inset rectangle (calibration: avoid screen edges / failsafe corner).
    margin = max(0.0, min(0.45, edge_margin))
    x0, x1 = 639.0 * margin, 639.0 * (1.0 - margin)
    y0, y1 = 479.0 * margin, 479.0 * (1.0 - margin)
    span_x = max(x1 - x0, 1.0)
    span_y = max(y1 - y0, 1.0)

    print("Kinect v1 mouse control started.")
    print("Press ESC in the depth window to quit.")
    print(f"Calibration: depth_band={depth_band}, flip_y={flip_y}, edge_margin={margin}, alpha={alpha}, min_area={min_area}")

    while True:
        depth = get_depth()
        if depth is None:
            time.sleep(0.01)
            continue

        hand = get_hand_position(depth, min_area=min_area, depth_band=depth_band)
        if hand is not None:
            x, y = hand

            nx = (float(x) - x0) / span_x
            ny = (float(y) - y0) / span_y
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            if flip_y:
                ny = 1.0 - ny

            mouse_x = int(nx * (screen_w - 1))
            mouse_y = int(ny * (screen_h - 1))

            smooth_x = int(prev_x * (1 - alpha) + mouse_x * alpha)
            smooth_y = int(prev_y * (1 - alpha) + mouse_y * alpha)

            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

            if enable_click:
                current_depth = int(depth[y, x])
                if prev_depth is not None and current_depth < (prev_depth - click_threshold):
                    pyautogui.click()
                prev_depth = current_depth
        else:
            prev_depth = None

        depth_display = (np.clip(depth, 0, 2048) / 2048 * 255).astype(np.uint8)
        cv2.imshow("Depth", depth_display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kinect v1 depth-based mouse control for Raspberry Pi desktop."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Smoothing factor from 0.0 to 1.0 (default: 0.2).",
    )
    parser.add_argument(
        "--click-threshold",
        type=int,
        default=50,
        help="Depth drop in raw Kinect units to trigger click (default: 50).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1500,
        help="Minimum contour area for hand blob tracking (default: 1500).",
    )
    parser.add_argument(
        "--disable-click",
        action="store_true",
        help="Disable push-forward click gesture.",
    )
    parser.add_argument(
        "--depth-band",
        type=int,
        default=120,
        help="Kinect raw-depth window around nearest objects (default: 120). "
        "Smaller = stricter 'closest blob' (less background), larger = more forgiving.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Invert vertical mapping (hand up moves cursor up — often feels more natural).",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.0,
        help="Ignore outer fraction of Kinect frame when mapping to screen (0–0.45). "
        "E.g. 0.08 keeps pointer away from screen edges / PyAutoGUI failsafe corner.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        alpha=max(0.0, min(1.0, args.alpha)),
        click_threshold=max(1, args.click_threshold),
        min_area=max(100, args.min_area),
        enable_click=not args.disable_click,
        depth_band=max(20, min(800, args.depth_band)),
        flip_y=args.flip_y,
        edge_margin=args.edge_margin,
    )
