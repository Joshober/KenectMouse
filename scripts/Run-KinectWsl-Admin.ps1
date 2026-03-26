# Run this file: Right-click -> Run with PowerShell -> Run as Administrator
# Forwards a Kinect USB device then runs checks inside Ubuntu WSL.
# Motor alone (e.g. 7-2) is NOT enough for depth — also pass -CameraBusId from "usbipd list"
# (look for Xbox NUI Camera / 045e:02ae).

param(
    [string]$BusId = "7-2",
    [string]$CameraBusId = ""
)

function Invoke-UsbipdBindAttach {
    param([Parameter(Mandatory)][string]$Bid)
    Write-Host "  bind/attach $Bid ..." -ForegroundColor Cyan
    & usbipd bind --busid $Bid 2>&1 | ForEach-Object {
        if ($_ -match 'already shared') { Write-Host "  $_" -ForegroundColor DarkGray }
        else { Write-Host "  $_" }
    }
    & usbipd attach --wsl --busid $Bid 2>&1 | ForEach-Object {
        if ($_ -match 'already attached to a client') {
            Write-Host "  (device already attached to WSL — OK)" -ForegroundColor Green
        }
        else { Write-Host "  $_" }
    }
}

function ConvertTo-WslPath {
    param([Parameter(Mandatory)][string]$WindowsPath)
    # Avoid wslpath here: PowerShell -> wsl.exe quoting can strip backslashes (breaks paths with spaces).
    $full = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($full -notmatch '^([a-zA-Z]):\\(.*)$') {
        throw "Cannot convert to WSL path: $WindowsPath"
    }
    $drive = $Matches[1].ToLower()
    $tail = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$tail"
}

$ErrorActionPreference = "Stop"
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    Write-Host "Install usbipd: winget install --id dorssel.usbipd-win -e" -ForegroundColor Yellow
    exit 1
}

# usbipd attach requires a running WSL2 distro (Ubuntu), not only docker-desktop.
Write-Host "Starting Ubuntu WSL (background) so attach can succeed ..." -ForegroundColor Cyan
$null = Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", "Ubuntu", "-e", "sleep", "120") -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 4
wsl -d Ubuntu -e /bin/true 2>&1 | Out-Null

Write-Host "Binding and attaching USB device(s) ..." -ForegroundColor Cyan
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    Invoke-UsbipdBindAttach -Bid $BusId
    if ($CameraBusId) {
        Invoke-UsbipdBindAttach -Bid $CameraBusId
    }
    elseif ($BusId) {
        Write-Host "`nTip: If freenect still cannot open camera, run usbipd list on Windows and attach" -ForegroundColor Yellow
        Write-Host "     the 'Xbox NUI Camera' (045e:02ae) BUSID too, e.g.:" -ForegroundColor Yellow
        Write-Host "     .\scripts\Run-KinectWsl-Admin.ps1 -BusId $BusId -CameraBusId 7-1" -ForegroundColor Yellow
    }
}
finally {
    $ErrorActionPreference = $prevEap
}

Write-Host "`nRunning WSL checks ..." -ForegroundColor Cyan
$checkSh = Join-Path $PSScriptRoot "wsl-check-kinect.sh"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
# sed -i on /mnt/c (DrvFs) often fails to strip CRLF — build LF-only copy in %TEMP%.
Write-Host "Writing LF-only copy to %TEMP% (DrvFs-safe) ..." -ForegroundColor DarkGray
$raw = [System.IO.File]::ReadAllText($checkSh)
$unix = $raw -replace "`r`n", "`n" -replace "`r", "`n"
$tempSh = Join-Path $env:TEMP "kinect-wsl-check-run.sh"
$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($tempSh, $unix, $utf8)
$tmpWsl = ConvertTo-WslPath $tempSh
$repoWsl = ConvertTo-WslPath $repoRoot
function EscapeBashSingleQuote([string]$s) {
    "'" + ($s -replace "'", "'\''") + "'"
}
$inner = "export KINECT_REPO=" + (EscapeBashSingleQuote $repoWsl) + "; exec bash " + (EscapeBashSingleQuote $tmpWsl)
$prevEap2 = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & wsl -d Ubuntu -- bash -c $inner 2>&1 | ForEach-Object { Write-Host $_ }
}
finally {
    $ErrorActionPreference = $prevEap2
}

Write-Host "`nDepth needs the NUI Camera USB device (045e:02ae), not only the motor (045e:02b0)." -ForegroundColor Yellow
Read-Host "Press Enter to close"
