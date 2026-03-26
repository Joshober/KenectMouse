#!/usr/bin/env python3
"""
Kinect v1 live preview: **freenect** + **matplotlib** (no OpenCV).

WSL often defaults matplotlib to non-interactive **Agg** → no windows.
This script forces **TkAgg** (needs `sudo apt install python3-tk` on Ubuntu).

Quit: close the window or Ctrl+C.

No-GUI fallback:  python3 scripts/kinect_snapshot.py
"""
import argparse
import os
import sys

import matplotlib

# Force a GUI backend; WSL often sets MPLBACKEND=Agg which never opens a window.
_backend = os.environ.get("MPLBACKEND", "TkAgg")
if _backend.lower() == "agg":
    _backend = "TkAgg"
matplotlib.use(_backend, force=True)
import freenect
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Preview Kinect v1 via freenect + matplotlib (no OpenCV)."
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Show RGB next to depth (two subplots).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Flip images horizontally.",
    )
    args = parser.parse_args()

    backend = matplotlib.get_backend()
    print("Matplotlib backend:", backend)

    if backend.lower() == "agg":
        print(
            "\nNo GUI backend — matplotlib will not open a window.\n"
            "Fix on Ubuntu/WSL:\n"
            "  sudo apt install -y python3-tk\n"
            "  export MPLBACKEND=TkAgg\n"
            "  export DISPLAY=:0\n"
            "Or save PNGs without any window:\n"
            "  python3 scripts/kinect_snapshot.py\n",
            file=sys.stderr,
        )
        return 1

    try:
        import tkinter  # noqa: F401 — TkAgg needs this
    except ImportError:
        print(
            "Tkinter missing (required for TkAgg). Run:\n"
            "  sudo apt install -y python3-tk",
            file=sys.stderr,
        )
        return 1

    print("Close the plot window or press Ctrl+C to quit.")

    plt.ion()
    if args.rgb:
        fig, (ax_d, ax_c) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Kinect v1 — depth | RGB (freenect)")
    else:
        fig, ax_d = plt.subplots(1, 1, figsize=(7, 5))
        fig.suptitle("Kinect v1 — depth (freenect)")
        ax_c = None

    im_d = ax_d.imshow(np.zeros((480, 640)), cmap="inferno", vmin=0, vmax=2048)
    ax_d.set_title("Depth")
    if ax_c is not None:
        im_c = ax_c.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        ax_c.set_title("RGB")
    plt.tight_layout()
    try:
        fig.canvas.manager.set_window_title("Kinect preview (matplotlib)")
    except (AttributeError, ValueError):
        pass
    plt.show(block=False)
    plt.pause(0.2)

    first_stats = True
    frame_n = 0

    try:
        while plt.fignum_exists(fig.number):
            depth, _ = freenect.sync_get_depth()
            if depth is None:
                plt.pause(0.05)
                continue

            d = np.asarray(depth, dtype=np.float32)
            if args.mirror:
                d = np.fliplr(d)

            if first_stats:
                nz = d[d > 0]
                print(
                    f"First depth frame: shape={d.shape} min={d.min():.0f} max={d.max():.0f} "
                    f"nonzero={nz.size}"
                )
                first_stats = False

            im_d.set_data(d)
            # Fixed 0..2048 often looks “all black” if most pixels are 0; stretch contrast on valid depth.
            valid = d[d > 0]
            if valid.size > 100:
                lo, hi = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
                hi = max(hi, lo + 1)
                im_d.set_clim(lo, hi)
            else:
                im_d.set_clim(0, 2048)

            if ax_c is not None:
                rgb, _ = freenect.sync_get_video()
                if rgb is not None:
                    vis = np.asarray(rgb)
                    if vis.ndim == 2:
                        vis = np.stack([vis] * 3, axis=-1)
                    if vis.dtype != np.uint8:
                        vis = np.clip(vis, 0, 255).astype(np.uint8)
                    if args.mirror:
                        vis = np.fliplr(vis)
                    im_c.set_data(vis)
                    if frame_n == 0:
                        print(f"First RGB frame: shape={vis.shape} dtype={vis.dtype}")

            # TkAgg often needs a full draw, not only draw_idle, or the canvas stays black.
            fig.canvas.draw()
            plt.pause(0.001)
            frame_n += 1
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
