#!/usr/bin/env bash
# Run in WSL: bash scripts/wsl-check-kinect.sh
echo "=== USB (Microsoft / Kinect VID 045e) ==="
if command -v lsusb >/dev/null 2>&1; then
  lsusb | grep -iE '045e|microsoft|kinect|xbox|nui' || echo "(no match - try: sudo apt install usbutils)"
else
  echo "lsusb missing. Run: sudo apt install usbutils"
  found=0
  for d in /sys/bus/usb/devices/*; do
    [ -f "$d/idVendor" ] || continue
    v=$(cat "$d/idVendor" 2>/dev/null)
    p=$(cat "$d/idProduct" 2>/dev/null)
    if [ "$v" = "045e" ]; then
      echo "  $d  $v:$p"
      found=1
    fi
  done
  [ "$found" = 1 ] || echo "(no 045e in sysfs - use usbipd on Windows Admin PowerShell)"
fi
# Depth (libfreenect) needs the NUI camera (PID 02ae). Motor-only (02b0) is not enough.
if [ "$found" = 1 ]; then
  cam=0
  for d in /sys/bus/usb/devices/*; do
    [ -f "$d/idVendor" ] || continue
    [ "$(cat "$d/idVendor" 2>/dev/null)" = "045e" ] || continue
    p=$(cat "$d/idProduct" 2>/dev/null | tr 'A-Z' 'a-z')
    case "$p" in 02ae|02a9) cam=1 ;; esac
  done
  if [ "$cam" = 0 ]; then
    echo ""
    echo ">>> No Xbox NUI Camera (045e:02ae) in WSL — depth will fail."
    echo ">>> On Windows (Admin): usbipd list  ->  bind+attach the CAMERA BUSID too."
    echo ">>> Example: .\\scripts\\Run-KinectWsl-Admin.ps1 -BusId 7-2 -CameraBusId <BUSID>"
  fi
fi
echo ""
echo "=== Python freenect (5s max) ==="
# Run-KinectWsl-Admin.ps1 copies this to %TEMP% with LF only; set repo explicitly:
cd "${KINECT_REPO:-$(dirname "$0")/..}" || exit 1
timeout 5 python3 -c "
import sys
try:
    import freenect
    r = freenect.sync_get_depth()
    if r is None:
        print('FAIL: sync_get_depth returned None')
        sys.exit(1)
    d, _ = r
    if d is not None:
        print('OK: depth shape', d.shape)
        sys.exit(0)
    print('FAIL: depth is None')
    sys.exit(1)
except Exception as e:
    print('FAIL:', e)
    sys.exit(1)
" || true
echo "(check finished)"
