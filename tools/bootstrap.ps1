<#
  Bootstrap a Windows dev environment for this repo.

  - Creates/refreshes a local venv at .venv using the Python launcher (py -3.11)
  - Upgrades pip
  - Installs requirements from requirements.txt

  Usage:
    powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1

  Notes:
  - Idempotent: safe to re-run; does not modify global PATH.
  - Requires Python Launcher with Python 3.11: `py -3.11`.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[bootstrap] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[bootstrap] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[bootstrap] ERROR: $msg" -ForegroundColor Red }

try {
  $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
  $repoRoot  = Split-Path -Parent $scriptDir
  Push-Location $repoRoot

  Write-Info "Repo root: $repoRoot"

  # Verify Python launcher and 3.11 availability
  if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher 'py' not found. Install Python 3.11 from python.org or Microsoft Store and ensure 'py' is available."
  }
  try {
    $pyVersion = (& py -3.11 -c "import sys; print(sys.version)" 2>$null)
    if (-not $pyVersion) { throw "no output" }
    Write-Info "Detected Python 3.11: $pyVersion"
  } catch {
    throw "Python 3.11 not found via 'py -3.11'. Install Python 3.11 and try again."
  }

  $venvPath    = Join-Path $repoRoot ".venv"
  $venvPy      = Join-Path $venvPath "Scripts\\python.exe"

  if (-not (Test-Path $venvPath)) {
    Write-Info "Creating virtual environment at .venv ..."
    & py -3.11 -m venv $venvPath
  } else {
    Write-Info "Using existing virtual environment at .venv"
  }

  if (-not (Test-Path $venvPy)) {
    throw "Virtual environment seems incomplete. Expected '$venvPy'. Try deleting .venv and re-running."
  }

  Write-Info "Upgrading pip ..."
  & $venvPy -m pip install --upgrade pip

  $reqFile = Join-Path $repoRoot "requirements.txt"
  if (-not (Test-Path $reqFile)) { throw "requirements.txt not found at $reqFile" }
  Write-Info "Installing dependencies from requirements.txt ..."
  & $venvPy -m pip install -r $reqFile

  Write-Host "\n[bootstrap] Environment ready. Activate with: `\n  .\\.venv\\Scripts\\Activate.ps1`" -ForegroundColor Green
} catch {
  Write-Err $_
  exit 1
} finally {
  Pop-Location | Out-Null
}

