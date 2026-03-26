"""One-shot depth grab for WSL/CLI (no OpenCV window)."""
import sys

try:
    import freenect
except ImportError as e:
    print("FAIL: import freenect:", e)
    sys.exit(1)


def _timeout(_signum, _frame):
    print(
        "FAIL: timed out waiting for Kinect (12s).\n"
        "  Try: sudo python3 scripts/quick_freenect_test.py\n"
        "  If sudo works: sudo usermod -aG video,plugdev $USER  then close and reopen Ubuntu.\n"
        "  Also: sudo apt install -y libfreenect-dev freenect libfreenect-bin"
    )
    sys.exit(2)


# SIGALRM only on Unix (WSL/Linux)
if sys.platform != "win32":
    import signal

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(12)

try:
    r = freenect.sync_get_depth()
finally:
    if sys.platform != "win32":
        signal.alarm(0)

if r is None:
    print("FAIL: sync_get_depth returned None")
    sys.exit(1)
d, _ = r
if d is None:
    print("FAIL: depth buffer is None")
    sys.exit(1)
print("OK: depth shape", d.shape, "dtype", d.dtype)
sys.exit(0)
