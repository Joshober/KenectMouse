#!/usr/bin/env bash
set -euo pipefail

# One-shot Raspberry Pi bootstrap + run script for Kinect Fruit Ninja mode.
# - Installs apt dependencies (first run)
# - Creates/updates .venv with system-site-packages
# - Installs pip requirements
# - Optionally sanity-checks Kinect
# - Launches kinect_mouse.py with Fruit Ninja defaults

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

RUN_GLVIEW=0
SKIP_APT=0
PRESET="brown"
SHOW_RGB=1

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_fruit_ninja_pi.sh [options]

Options:
  --preset brown|white    Knife color preset (default: brown)
  --no-rgb-windows        Do not show RGB/mask windows while running
  --test-glview           Run freenect-glview check before launching Python
  --skip-apt              Skip apt install/upgrade step
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="${2:-}"
      shift 2
      ;;
    --no-rgb-windows)
      SHOW_RGB=0
      shift
      ;;
    --test-glview)
      RUN_GLVIEW=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${PRESET}" != "brown" && "${PRESET}" != "white" ]]; then
  echo "Invalid --preset value: ${PRESET}. Use brown or white." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/kinect_mouse.py" ]]; then
  echo "Could not find kinect_mouse.py at ${REPO_ROOT}" >&2
  exit 1
fi

if [[ "${SKIP_APT}" -eq 0 ]]; then
  echo "==> Installing OS dependencies with apt..."
  sudo apt update
  sudo apt install -y \
    python3-pip python3-venv \
    libfreenect-dev freenect libfreenect-bin \
    python3-opencv python3-freenect
fi

if [[ "${RUN_GLVIEW}" -eq 1 ]]; then
  echo "==> Running Kinect sanity check (close glview to continue)..."
  freenect-glview
fi

echo "==> Creating/updating virtual environment..."
python3 -m venv "${VENV_DIR}" --system-site-packages

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -r "${REPO_ROOT}/requirements.txt"

PRESET_FLAG="--brown-sword-preset"
if [[ "${PRESET}" == "white" ]]; then
  PRESET_FLAG="--white-sword-preset"
fi

SHOW_RGB_FLAG=""
if [[ "${SHOW_RGB}" -eq 1 ]]; then
  SHOW_RGB_FLAG="--show-rgb"
fi

echo "==> Launching Fruit Ninja mode (preset=${PRESET})..."
set -x
python3 "${REPO_ROOT}/kinect_mouse.py" \
  --fruit-ninja-mode \
  "${PRESET_FLAG}" \
  --flip-y \
  --edge-margin 0.05 \
  --slash-speed-px 42 \
  --slash-release-frames 3 \
  ${SHOW_RGB_FLAG}
