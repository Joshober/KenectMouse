un (recommended)
python3 kinect_mouse.py --use-rgb --show-rgb --rgb-refine-depth closest
If “closest” is too jittery (depth noise), use percentile
This targets a stable “near” depth within the orange blob:

python3 kinect_mouse.py --use-rgb --show-rgb --rgb-refine-depth percentile --rgb-depth-percentile 15
Turn off the depth refine (RGB-only) if you want
python3 kinect_mouse.py --use-rgb --rgb-refine-depth none

python3 kinect_mouse.py ^
  --target closest ^
  --alpha 0.45 ^
  --nearest-percentile 2 ^
  --depth-band 60 ^
  --min-area 900 ^
  --max-area 30000 ^
  --morph-kernel 3 --morph-open-iters 1 --morph-close-iters 0




More responsive (less lag): increase --alpha (try 0.55), reduce morphology (--morph-kernel 3, set --morph-close-iters 0)
More accurate “only the band”: lower --nearest-percentile (try 1.0–2.0) and lower --depth-band (try 40–70)
If it jumps to noise: raise --min-area (e.g. 1200–2000) and/or turn --morph-close-iters 1 back on
If it still follows your body: lower --max-area (e.g. 20000–25000) and keep --target closest
Physical setup that matters a lot
Stand so your band/hand is closer to the Kinect than your torso (even 10–20 cm closer helps), and keep the band clearly in front when controlling.











# Run on Raspberry Pi 4 (Kinect v1)
python3 kinect_mouse.py --target closest --depth-band 80 --max-area 35000 --min-area 1200

## TL;DR

```bash
git clone https://github.com/Joshober/KenectMouse.git
cd KenectMouse
```

Then follow **sections 1 → 2 → 3** in order (system packages → test Kinect → venv + `pip install`).  
If anything fails, jump to the troubleshooting blocks under each section.

**Pi-specific:** On ARM, `requirements.txt` does **not** pip-install **`opencv-python`** or **`freenect`** (they are unreliable on Pi). Install **`python3-opencv`** and **`python3-freenect`** with **apt**, and use a venv with **`--system-site-packages`** so `import cv2` and `import freenect` work.

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

# Python binding for libfreenect (avoids broken `pip install freenect` wheels on Pi):
sudo apt install -y python3-freenect
```

### If `python3-freenect` is not found

Your OS image may be missing that package. Then install build deps and use pip **on the same Python you run**

(the `freenect` PyPI package compiles Cython against `libfreenect`):

```bash
sudo apt install -y python3-dev build-essential cython3 libfreenect-dev
cd KenectMouse
source .venv/bin/activate
pip install -r requirements-pi-freenect-pip.txt
```

### If pip still fails to build `freenect` (common on very new Python, e.g. 3.12 / 3.13)

Use an **older Python** for the venv (often **3.10** or **3.11**), whichever your OS ships.

List what’s available:

```bash
apt-cache search --names-only '^python3\.1[0-9]$'
```

Example with **Python 3.10** (replace `3.10` with another version from the list if needed):

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev \
  libfreenect-dev build-essential cython3

cd KenectMouse
deactivate 2>/dev/null || true
rm -rf .venv
python3.10 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-pi-freenect-pip.txt   # only if python3-freenect apt pkg missing
python3 -c "import freenect; print('freenect OK')"
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

Quick check that **freenect** is importable (should print `freenect OK`):

```bash
python3 -c "import freenect; print('freenect OK')"
```

If this fails before you even run preview/mouse, fix `python3-freenect` / pip build / Python version (sections above) first.

### If pip says it “can’t find a version” of `freenect`

On Raspberry Pi, the PyPI `freenect` package often fails (or has no compatible wheels).
Use the **apt** Python binding instead.

```bash
sudo apt update
sudo apt install -y python3-freenect libfreenect-dev freenect libfreenect-bin
python3 -c "import freenect; print('freenect OK')"
```

Then rebuild your venv so it can see apt packages:

```bash
cd KenectMouse
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -r requirements.txt
```

If you *still* see pip trying to install `freenect` from PyPI, verify you’re using the repo you just updated and that you didn’t run `pip` from a different Python:

```bash
cd KenectMouse
python3 -c "import platform; print('machine=', platform.machine())"
python3 -m pip --version
python3 -m pip install -r requirements.txt -vvv
```

Paste the line where it attempts to install `freenect` and I’ll tell you why the marker wasn’t applied.

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

