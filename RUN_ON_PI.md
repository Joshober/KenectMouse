# Run on Raspberry Pi 4 (Kinect v1)

This repo includes a few scripts:

- `kinect_mouse.py`: depth-hand cursor control (desktop/X11)
- `kinect_v1_mouse.py`: Linux virtual mouse via `/dev/uinput` (works well in Docker on Linux)
- `scripts/kinect_preview.py`: OpenCV preview (what the tracker sees)
- `scripts/kinect_preview_matplotlib.py`: Matplotlib preview (no OpenCV)
- `scripts/kinect_snapshot.py`: save a PNG snapshot (no GUI)

## Hardware checklist

- Kinect v1 **power + USB adapter**
- Plug Kinect into a **USB 2.0 port** if possible (Pi 4 USB3 can work but USB2 is often more stable)

## 1) Install OS packages

On Raspberry Pi OS (64-bit recommended):

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y \
  python3-pip python3-venv \
  libfreenect-dev freenect libfreenect-bin \
  python3-opencv
```

## 2) Test Kinect first (don’t skip)

```bash
freenect-glview
```

You should see depth/RGB. Quit it.

If this doesn’t work, fix power/USB first before running Python.

## 3) Install Python deps

From the repo:

```bash
cd KenectMouse
# If you cloned earlier, pull the latest docs/requirements:
# git pull
#
# Use system-site-packages so the venv can see the apt-installed `cv2` (python3-opencv).
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
```

### If you see “can’t find a compatible version of cv2/opencv”

On Raspberry Pi, `cv2` usually comes from **apt**, not `pip`.

Fix:

```bash
sudo apt update
sudo apt install -y python3-opencv

cd KenectMouse
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
python3 -c "import cv2; print(cv2.__version__)"
```

## 4) Preview what the code sees

OpenCV preview:

```bash
source .venv/bin/activate
python3 scripts/kinect_preview.py --rgb --mirror
```

No-window snapshot (saves PNGs):

```bash
source .venv/bin/activate
python3 scripts/kinect_snapshot.py --rgb --out-dir /tmp/kinect
ls -la /tmp/kinect
```

## 5) Run the mouse controller (desktop)

Run (ESC quits in the window):

```bash
source .venv/bin/activate
python3 kinect_mouse.py --flip-y --edge-margin 0.08
```

Suggested tuning knobs:

- `--alpha 0.12` (smoother) … `--alpha 0.35` (faster)
- `--min-area 1200` (more sensitive) … `--min-area 3500` (less jitter)
- `--depth-band 80` (stricter) … `--depth-band 200` (more forgiving)
- `--disable-click` while you tune pointing
- `--click-threshold 60` if it clicks too easily

## If you get permission errors

```bash
sudo usermod -aG video,plugdev $USER
sudo reboot
```

