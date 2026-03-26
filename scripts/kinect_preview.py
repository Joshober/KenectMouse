#!/usr/bin/env python3
"""
Live Kinect v1 preview (depth + optional RGB). No mouse control.
Uses freenect + OpenCV for display. For the same data without OpenCV, see
kinect_preview_matplotlib.py (freenect + matplotlib).

Run in WSL/Linux with Kinect attached:  python3 scripts/kinect_preview.py
Quit: ESC or Q in the OpenCV window, or Ctrl+C in the terminal.
"""
import argparse
import sys

import cv2
import freenect
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Preview Kinect v1 depth (and optional RGB).")
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Also show RGB camera (second window).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Flip images horizontally (self-view style).",
    )
    args = parser.parse_args()

    print("Kinect preview — ESC or Q in a window to quit.")
    if args.rgb:
        print("Showing: Depth + RGB")
    else:
        print("Showing: Depth only (use --rgb for color too)")

    try:
        while True:
            depth_raw, _ = freenect.sync_get_depth()
            if depth_raw is None:
                continue

            depth_vis = (np.clip(depth_raw, 0, 2048) / 2048.0 * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            if args.mirror:
                depth_color = cv2.flip(depth_color, 1)
            cv2.imshow("Kinect depth (what tracking uses)", depth_color)

            if args.rgb:
                rgb, _ = freenect.sync_get_video()
                if rgb is not None:
                    # freenect returns RGB; OpenCV expects BGR for imshow
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if args.mirror:
                        bgr = cv2.flip(bgr, 1)
                    cv2.imshow("Kinect RGB", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
