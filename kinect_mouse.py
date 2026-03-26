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


def get_rgb():
    frame = freenect.sync_get_video()
    if frame is None:
        return None
    rgb, _ = frame
    if rgb is None:
        return None
    # libfreenect returns RGB (not BGR)
    return rgb.astype(np.uint8)


def depth_sample(depth, x, y, radius):
    h, w = depth.shape[:2]
    r = max(0, int(radius))
    x0 = max(0, int(x) - r)
    x1 = min(w - 1, int(x) + r)
    y0 = max(0, int(y) - r)
    y1 = min(h - 1, int(y) + r)
    patch = depth[y0 : y1 + 1, x0 : x1 + 1]
    vals = patch[(patch > 0) & (patch < 2047)]
    if vals.size == 0:
        return None
    return int(np.median(vals))


def get_marker_position_rgb(
    rgb,
    min_area,
    max_area,
    hsv_lower,
    hsv_upper,
    morph_kernel,
    morph_open_iters,
    morph_close_iters,
):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = max(1, int(morph_kernel))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    oi = max(0, int(morph_open_iters))
    ci = max(0, int(morph_close_iters))
    if oi > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=oi)
    if ci > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=ci)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    candidates = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < float(min_area):
            continue
        if max_area is not None and area > float(max_area):
            continue
        candidates.append((area, c))

    if not candidates:
        return None, mask

    chosen = max(candidates, key=lambda t: t[0])[1]
    m = cv2.moments(chosen)
    if m["m00"] == 0:
        return None, mask
    y = int(m["m01"] / m["m00"])
    x = int(m["m10"] / m["m00"])
    return (x, y), mask


def refine_rgb_position_with_depth(
    pos,
    rgb_mask,
    depth,
    strategy,
    depth_percentile,
):
    if pos is None or rgb_mask is None or depth is None:
        return pos

    if strategy == "none":
        return pos

    mask_coords = np.where(rgb_mask > 0)
    if mask_coords[0].size == 0:
        return pos

    ys = mask_coords[0]
    xs = mask_coords[1]
    dvals = depth[ys, xs].astype(np.int32)
    valid = (dvals > 0) & (dvals < 2047)
    if not np.any(valid):
        return pos

    xs = xs[valid]
    ys = ys[valid]
    dvals = dvals[valid]

    if strategy == "closest":
        idx = int(np.argmin(dvals))
        return int(xs[idx]), int(ys[idx])

    if strategy == "percentile":
        p = float(depth_percentile)
        p = max(0.0, min(100.0, p))
        target = float(np.percentile(dvals, p))
        idx = int(np.argmin(np.abs(dvals.astype(np.float32) - target)))
        return int(xs[idx]), int(ys[idx])

    return pos


def get_hand_position(
    depth,
    min_area,
    max_area,
    depth_band,
    target_mode,
    nearest_percentile,
    morph_kernel,
    morph_open_iters,
    morph_close_iters,
):
    # Ignore invalid values and very far points.
    valid = np.where((depth > 0) & (depth < 2047), depth, 0).astype(np.uint16)
    if np.count_nonzero(valid) == 0:
        return None

    # Use a near-depth band so background doesn't dominate the "closest point".
    # Smaller percentile = more "lock to nearest" (often hand/band), but can be noisier.
    nearest = np.percentile(valid[valid > 0], nearest_percentile)
    near_band_max = min(nearest + depth_band, 2047)
    mask = np.where((valid >= nearest) & (valid <= near_band_max), 255, 0).astype(np.uint8)

    k = max(1, int(morph_kernel))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    oi = max(0, int(morph_open_iters))
    ci = max(0, int(morph_close_iters))
    if oi > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=oi)
    if ci > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=ci)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < float(min_area):
            continue
        if max_area is not None and area > float(max_area):
            continue
        candidates.append((area, c))

    if not candidates:
        return None

    if target_mode == "largest":
        chosen = max(candidates, key=lambda t: t[0])[1]
    elif target_mode == "closest":
        # Pick the blob with the smallest median depth within its contour.
        # This tends to lock onto a hand/band closer to the sensor than the torso.
        best = None
        best_med = None
        for _, c in candidates:
            cmask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(cmask, [c], -1, 255, thickness=-1)
            dvals = valid[cmask > 0]
            dvals = dvals[dvals > 0]
            if dvals.size == 0:
                continue
            med = float(np.median(dvals))
            if best is None or med < best_med:
                best = c
                best_med = med
        chosen = best if best is not None else max(candidates, key=lambda t: t[0])[1]
    else:
        chosen = max(candidates, key=lambda t: t[0])[1]

    moments = cv2.moments(chosen)
    if moments["m00"] == 0:
        return None

    y = int(moments["m01"] / moments["m00"])
    x = int(moments["m10"] / moments["m00"])
    return x, y


def run(
    alpha,
    click_threshold,
    min_area,
    max_area,
    enable_click,
    depth_band,
    flip_y,
    edge_margin,
    target_mode,
    nearest_percentile,
    morph_kernel,
    morph_open_iters,
    morph_close_iters,
    use_rgb,
    hsv_lower,
    hsv_upper,
    rgb_depth_radius,
    show_rgb,
    rgb_refine_depth,
    rgb_depth_percentile,
):
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
    print(
        f"Calibration: depth_band={depth_band}, flip_y={flip_y}, edge_margin={margin}, alpha={alpha}, "
        f"min_area={min_area}, max_area={max_area}, target={target_mode}"
    )
    if use_rgb:
        print(
            f"RGB mode: hsv_lower={tuple(hsv_lower)}, hsv_upper={tuple(hsv_upper)}, "
            f"rgb_depth_radius={rgb_depth_radius}, show_rgb={show_rgb}"
        )

    while True:
        depth = get_depth()
        if depth is None:
            time.sleep(0.01)
            continue

        rgb = None
        rgb_mask = None
        if use_rgb:
            rgb = get_rgb()
            if rgb is not None:
                hand, rgb_mask = get_marker_position_rgb(
                    rgb,
                    min_area=min_area,
                    max_area=max_area,
                    hsv_lower=hsv_lower,
                    hsv_upper=hsv_upper,
                    morph_kernel=morph_kernel,
                    morph_open_iters=morph_open_iters,
                    morph_close_iters=morph_close_iters,
                )
                if hand is not None and rgb_refine_depth != "none":
                    hand = refine_rgb_position_with_depth(
                        hand,
                        rgb_mask=rgb_mask,
                        depth=depth,
                        strategy=rgb_refine_depth,
                        depth_percentile=rgb_depth_percentile,
                    )
        else:
            hand = get_hand_position(
                depth,
                min_area=min_area,
                max_area=max_area,
                depth_band=depth_band,
                target_mode=target_mode,
                nearest_percentile=nearest_percentile,
                morph_kernel=morph_kernel,
                morph_open_iters=morph_open_iters,
                morph_close_iters=morph_close_iters,
            )
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
                d = depth_sample(depth, x, y, radius=rgb_depth_radius if use_rgb else 0)
                current_depth = d if d is not None else int(depth[y, x])
                if prev_depth is not None and current_depth < (prev_depth - click_threshold):
                    pyautogui.click()
                prev_depth = current_depth
        else:
            prev_depth = None

        depth_display = (np.clip(depth, 0, 2048) / 2048 * 255).astype(np.uint8)
        cv2.imshow("Depth", depth_display)
        if use_rgb and show_rgb and rgb is not None and rgb_mask is not None:
            cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imshow("RGB Mask", rgb_mask)

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
        "--max-area",
        type=int,
        default=0,
        help="Ignore blobs larger than this area (0 disables). Useful to ignore torso (example: 35000).",
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
        "--nearest-percentile",
        type=float,
        default=5.0,
        help="Which percentile defines 'nearest' depth (default: 5). Lower = more 'nearest lock' (1–3 often works for a band).",
    )
    parser.add_argument(
        "--target",
        choices=["largest", "closest"],
        default="largest",
        help="Which blob to track: 'largest' (default) or 'closest' (better for hand/band).",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=5,
        help="Morphology kernel size (odd int; default: 5). Smaller = faster/more responsive but noisier.",
    )
    parser.add_argument(
        "--morph-open-iters",
        type=int,
        default=1,
        help="How many MORPH_OPEN iterations (default: 1). Set 0 to disable.",
    )
    parser.add_argument(
        "--morph-close-iters",
        type=int,
        default=1,
        help="How many MORPH_CLOSE iterations (default: 1). Set 0 to disable.",
    )
    parser.add_argument(
        "--use-rgb",
        action="store_true",
        help="Track a colored marker using Kinect RGB (recommended for a bright orange box).",
    )
    parser.add_argument(
        "--hsv-lower",
        type=int,
        nargs=3,
        default=[5, 120, 120],
        metavar=("H", "S", "V"),
        help="HSV lower bound for marker threshold (default tuned for orange).",
    )
    parser.add_argument(
        "--hsv-upper",
        type=int,
        nargs=3,
        default=[25, 255, 255],
        metavar=("H", "S", "V"),
        help="HSV upper bound for marker threshold (default tuned for orange).",
    )
    parser.add_argument(
        "--rgb-depth-radius",
        type=int,
        default=8,
        help="When using RGB, sample depth in a +/-radius neighborhood for click depth (default: 8).",
    )
    parser.add_argument(
        "--show-rgb",
        action="store_true",
        help="Show RGB + mask windows (helpful for tuning HSV).",
    )
    parser.add_argument(
        "--rgb-refine-depth",
        choices=["none", "closest", "percentile"],
        default="closest",
        help="When using RGB, refine the marker position using depth inside the RGB mask (default: closest).",
    )
    parser.add_argument(
        "--rgb-depth-percentile",
        type=float,
        default=10.0,
        help="If --rgb-refine-depth percentile: which depth percentile inside the RGB mask to target (default: 10).",
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
        max_area=None if args.max_area <= 0 else max(1, args.max_area),
        enable_click=not args.disable_click,
        depth_band=max(20, min(800, args.depth_band)),
        flip_y=args.flip_y,
        edge_margin=args.edge_margin,
        target_mode=args.target,
        nearest_percentile=max(0.1, min(20.0, float(args.nearest_percentile))),
        morph_kernel=max(1, args.morph_kernel),
        morph_open_iters=max(0, args.morph_open_iters),
        morph_close_iters=max(0, args.morph_close_iters),
        use_rgb=bool(args.use_rgb),
        hsv_lower=[max(0, min(179, int(v))) for v in args.hsv_lower],
        hsv_upper=[max(0, min(255, int(v))) for v in args.hsv_upper],
        rgb_depth_radius=max(0, int(args.rgb_depth_radius)),
        show_rgb=bool(args.show_rgb),
        rgb_refine_depth=str(args.rgb_refine_depth),
        rgb_depth_percentile=max(0.0, min(100.0, float(args.rgb_depth_percentile))),
    )
