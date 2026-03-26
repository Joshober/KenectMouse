import argparse
import collections
import time

import cv2
import freenect
import numpy as np
from evdev import UInput, ecodes


class WaveDetector:
    def __init__(self, min_span=0.18, min_direction_changes=3, history_size=20):
        self.min_span = min_span
        self.min_direction_changes = min_direction_changes
        self.history = collections.deque(maxlen=history_size)

    def update(self, x_value):
        self.history.append(x_value)
        if len(self.history) < 8:
            return False

        values = list(self.history)
        span = max(values) - min(values)
        if span < self.min_span:
            return False

        changes = 0
        last_dir = 0
        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            if abs(delta) < 0.004:
                continue
            cur_dir = 1 if delta > 0 else -1
            if last_dir != 0 and cur_dir != last_dir:
                changes += 1
            last_dir = cur_dir
        return changes >= self.min_direction_changes


class CursorMover:
    def __init__(self):
        capabilities = {ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y]}
        self.ui = UInput(capabilities, name="kinect-v1-mouse", version=0x3)

    def move(self, dx, dy):
        self.ui.write(ecodes.EV_REL, ecodes.REL_X, int(dx))
        self.ui.write(ecodes.EV_REL, ecodes.REL_Y, int(dy))
        self.ui.syn()

    def close(self):
        self.ui.close()


def get_depth_frame():
    frame, _ = freenect.sync_get_depth()
    if frame is None:
        return None
    return frame.astype(np.uint16)


def extract_hand_centroid(depth_frame, near_mm, far_mm, min_area):
    mask = np.where((depth_frame > near_mm) & (depth_frame < far_mm), 255, 0).astype(
        np.uint8
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def run(args):
    wave = WaveDetector()
    mover = CursorMover()
    armed = False
    last_seen = time.time()
    last_x = None
    last_y = None

    print("Kinect v1 cursor control started.")
    print("Wave your hand side-to-side to arm tracking.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            depth = get_depth_frame()
            if depth is None:
                time.sleep(0.01)
                continue

            point = extract_hand_centroid(
                depth, near_mm=args.near_mm, far_mm=args.far_mm, min_area=args.min_area
            )
            if point is None:
                if armed and (time.time() - last_seen) > 2.0:
                    armed = False
                    wave.history.clear()
                    print("Tracking lost. Wave again to re-arm.")
                time.sleep(0.005)
                continue

            cx, cy = point
            last_seen = time.time()

            x_norm = cx / 639.0
            y_norm = cy / 479.0

            if wave.update(x_norm) and not armed:
                armed = True
                last_x = x_norm
                last_y = y_norm
                print("Wave detected. Cursor control enabled.")
                continue

            if not armed:
                continue

            # Relative motion keeps this independent from desktop APIs in Docker.
            if last_x is None or last_y is None:
                last_x = x_norm
                last_y = y_norm
                continue

            dx_norm = x_norm - last_x
            dy_norm = y_norm - last_y

            move_x = clamp(dx_norm * args.speed_x, -args.max_step, args.max_step)
            move_y = clamp(dy_norm * args.speed_y, -args.max_step, args.max_step)

            if abs(move_x) > args.deadzone or abs(move_y) > args.deadzone:
                mover.move(move_x, move_y)

            last_x = x_norm
            last_y = y_norm
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopped Kinect v1 cursor control.")
    finally:
        mover.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kinect v1 hand-wave to cursor controller (Linux/Raspberry Pi)."
    )
    parser.add_argument("--near-mm", type=int, default=350, help="Near depth threshold.")
    parser.add_argument("--far-mm", type=int, default=1200, help="Far depth threshold.")
    parser.add_argument(
        "--min-area",
        type=int,
        default=1800,
        help="Minimum contour area for tracked hand blob.",
    )
    parser.add_argument(
        "--speed-x",
        type=float,
        default=260.0,
        help="Horizontal cursor speed multiplier.",
    )
    parser.add_argument(
        "--speed-y",
        type=float,
        default=220.0,
        help="Vertical cursor speed multiplier.",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=28.0,
        help="Max pixels per update to avoid jumps.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=1.0,
        help="Ignore small movements under this many pixels.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
