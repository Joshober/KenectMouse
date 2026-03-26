#!/usr/bin/env python3
"""
Grab one Kinect depth (and optional RGB) frame and save PNG files — **no GUI**.
Use this when matplotlib/OpenCV windows do not appear in WSL.

  python3 scripts/kinect_snapshot.py
  python3 scripts/kinect_snapshot.py --out-dir /tmp/k

Opens in Windows:  explorer.exe $(wslpath -w /tmp/k)   or open paths from File Explorer.
"""
import argparse
import os
import sys

import freenect
import numpy as np

# Non-interactive save only
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=".", help="Directory for PNG files")
    p.add_argument("--rgb", action="store_true", help="Also save RGB frame")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    depth, _ = freenect.sync_get_depth()
    if depth is None:
        print("FAIL: no depth frame", file=sys.stderr)
        return 1

    d = np.asarray(depth)
    depth_path = os.path.join(args.out_dir, "kinect_depth.png")
    plt.imsave(depth_path, d, cmap="inferno", vmin=0, vmax=2048)
    print("Saved:", os.path.abspath(depth_path))

    if args.rgb:
        rgb, _ = freenect.sync_get_video()
        if rgb is not None:
            rgb_path = os.path.join(args.out_dir, "kinect_rgb.png")
            plt.imsave(rgb_path, np.asarray(rgb))
            print("Saved:", os.path.abspath(rgb_path))
        else:
            print("WARN: no RGB frame", file=sys.stderr)

    print("\nFrom WSL, copy path above or run:")
    print('  explorer.exe "$(wslpath -w ' + os.path.abspath(args.out_dir) + ')"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
