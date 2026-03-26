# Attach Microsoft Kinect (or any USB device) from Windows to WSL2.
# Run PowerShell as Administrator.
#
# One-time install of usbipd-win:
#   winget install --id dorssel.usbipd-win -e
#
# Usage:
#   .\scripts\attach-kinect-wsl.ps1              # lists devices only
#   .\scripts\attach-kinect-wsl.ps1 -BusId 2-2   # bind + attach that port

param(
    [string]$BusId
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    Write-Host "usbipd is not installed or not on PATH."
    Write-Host "Install: winget install --id dorssel.usbipd-win -e"
    Write-Host "Then open a new Administrator PowerShell and run this script again."
    exit 1
}

Write-Host "=== USB devices (find Kinect / Xbox NUI) ===" -ForegroundColor Cyan
usbipd list

if (-not $BusId) {
    Write-Host ""
    Write-Host "Copy the BUSID for your Kinect (e.g. 2-4), then run:" -ForegroundColor Yellow
    Write-Host "  .\scripts\attach-kinect-wsl.ps1 -BusId <BUSID>" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Binding and attaching $BusId to default WSL distro ..." -ForegroundColor Cyan
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & usbipd bind --busid $BusId 2>&1 | ForEach-Object { Write-Host $_ }
    & usbipd attach --wsl --busid $BusId 2>&1 | ForEach-Object { Write-Host $_ }
}
finally {
    $ErrorActionPreference = $prevEap
}

Write-Host ""
Write-Host "In Ubuntu, verify:" -ForegroundColor Green
Write-Host "  lsusb | grep -i -E 'microsoft|kinect|xbox'"
Write-Host "  freenect-glview"
Write-Host "  cd '/mnt/c/Users/IT Lab VR/Desktop/KenectMouse' && python3 kinect_mouse.py"
