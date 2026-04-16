# Kinect v1 Hand Mouse (Raspberry Pi + Docker)

Control your cursor with Kinect v1 depth tracking on Linux (including Raspberry Pi).

## Raspberry Pi (native install, no Docker)

Step-by-step guide (Kinect drivers, OpenCV, freenect, venv, run): see **`RUN_ON_PI.md`**.

## What this uses

- Kinect v1 (`libfreenect`)
- Depth blob tracking for hand position
- Wave gesture detection to arm control
- Virtual mouse events through `/dev/uinput`

## Prerequisites on Raspberry Pi

- Raspberry Pi OS (64-bit recommended)
- Docker + Docker Compose plugin installed
- Kinect v1 connected by USB
- Kernel module available: `uinput`

Enable uinput if needed:

```bash
sudo modprobe uinput
echo uinput | sudo tee -a /etc/modules
```

## Run with Docker

From this project directory:

```bash
docker compose build
docker compose up
```

The service runs:

```bash
python kinect_v1_mouse.py
```

Wave your hand side-to-side to arm tracking, then move your hand to move the cursor.

## Optional tuning

You can pass parameters in `docker-compose.yml` command, for example:

```yaml
command: python kinect_v1_mouse.py --near-mm 350 --far-mm 1200 --speed-x 260 --speed-y 220
```

Useful flags:

- `--near-mm`, `--far-mm`: depth range used to isolate the hand
- `--min-area`: minimum blob area
- `--speed-x`, `--speed-y`: cursor sensitivity
- `--max-step`: max movement per frame
- `--deadzone`: ignore tiny jitter

### Fruit Ninja mode (white poster "sword")

You can track a folded white poster/paper prop as a sword and slice via drag gestures:

```bash
python3 kinect_mouse.py --fruit-ninja-mode --white-sword-preset --show-rgb --flip-y --edge-margin 0.05
```

How it works:

- `--white-sword-preset` uses HSV thresholds tuned for bright white objects (low saturation, high value)
- `--fruit-ninja-mode` holds left mouse down while motion is fast, then releases on slow motion
- `--slash-speed-px` tunes slash sensitivity (lower = easier to trigger)
- `--slash-release-frames` tunes how quickly slash drag releases

If your prop is brown (cardboard/wood-like), use:

```bash
python3 kinect_mouse.py --fruit-ninja-mode --brown-sword-preset --show-rgb --flip-y --edge-margin 0.05
```

Or run everything in one shot on Raspberry Pi (installs deps, creates venv, installs pip packages, then launches):

```bash
chmod +x scripts/run_fruit_ninja_pi.sh
./scripts/run_fruit_ninja_pi.sh --preset brown
```

Linux executable launcher (small RAM-friendly "exe" for Pi):

```bash
chmod +x fruit_ninja_pi scripts/run_fruit_ninja_pi.sh
./fruit_ninja_pi --preset brown
```

Useful options:

- `--preset white` to track a white poster sword
- `--test-glview` to verify Kinect feed before launch
- `--skip-apt` for faster reruns after first setup
- `--no-rgb-windows` to hide RGB/mask windows

---

## WSL2 on Windows (Kinect USB must be forwarded)

WSL does **not** see USB devices by default. If `freenect` prints **“Can’t open device”** or `lsusb` shows no Kinect, attach the Kinect from Windows with **usbipd-win**.

### 1) Install usbipd-win (one-time, Administrator PowerShell)

```powershell
winget install --id dorssel.usbipd-win -e
```

Close and reopen **Administrator** PowerShell.

### 2) List USB devices and note the Kinect **BUSID**

```powershell
usbipd list
```

Look for something like **Xbox NUI Camera** / **Kinect** / **Microsoft**.

### 3) Bind and attach to Ubuntu

From this repo (Administrator PowerShell):

```powershell
cd "C:\Users\IT Lab VR\Desktop\KenectMouse"
.\scripts\attach-kinect-wsl.ps1
.\scripts\attach-kinect-wsl.ps1 -BusId 2-4
```

(Replace `2-4` with your real BUSID from `usbipd list`.)

Or manually:

```powershell
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

**`usbipd attach` needs your target distro actually running.** If you see *“There is no WSL 2 distribution running”*, open an **Ubuntu** terminal first, or run **`.\scripts\Run-KinectWsl-Admin.ps1`** (it starts Ubuntu in the background before attaching).

**Kinect v1 needs two USB devices in WSL for depth:** the **motor** (e.g. `045e:02b0`, “Xbox NUI Motor”) and the **camera** (`045e:02ae`, “Xbox NUI Camera”). Forward both BUSIDs:

```powershell
.\scripts\Run-KinectWsl-Admin.ps1 -BusId 7-2 -CameraBusId 7-1
```

(Use your real BUSIDs from `usbipd list`.) If the camera never appears in `usbipd list`, fix power/USB/drivers on Windows first.

If you have several distros, `attach --wsl` uses the **default** WSL distro. Set it with:

```powershell
wsl --set-default Ubuntu
```

Newer **usbipd-win** may support `--distribution Ubuntu` on `attach`; omit it if you get “Unrecognized command … `--distribution`”.

### 4) Verify inside Ubuntu

```bash
lsusb | grep -i -E 'microsoft|kinect|xbox'
freenect-glview
```

Then run the desktop mouse script:

```bash
cd "/mnt/c/Users/IT Lab VR/Desktop/KenectMouse"
python3 kinect_mouse.py
```

**Note:** After unplugging the Kinect or rebooting Windows, run **bind + attach** again. For udev rules in WSL (optional): `sudo apt install libfreenect-dev` and ensure your user can access the device; you may need `sudo usermod -aG plugdev $USER` and replug.
