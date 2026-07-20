<#
.SYNOPSIS
  One-command setup for FluxInfer on Windows: fetches FluxInfer and a CUDA
  build of llama.cpp, puts them somewhere sensible, and opens the menu.

.DESCRIPTION
  FluxInfer needs two things it does not ship: llama.cpp binaries, and a
  model. This script handles the first, checks for the second, and hands
  over to `fluxinfer menu`.

  It downloads published release archives from GitHub -- it does not
  compile anything, and it does not require Visual Studio, CMake or git.

  What it deliberately does NOT do: install a CUDA toolkit or an NVIDIA
  driver. Those are prerequisites; the script checks for them and stops
  with an explanation rather than pretending it can fix them.

.EXAMPLE
  irm https://raw.githubusercontent.com/federicobarrosgiuffrida/fluxinfer/master/scripts/install.ps1 | iex

.EXAMPLE
  .\install.ps1 -InstallDir "D:\tools\fluxinfer"
#>

[CmdletBinding()]
param(
  [string]$InstallDir = "$env:LOCALAPPDATA\FluxInfer",
  [switch]$SkipLlama,   # already have llama.cpp binaries elsewhere
  [switch]$NoLaunch     # install only, do not open the menu
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"  # makes Invoke-WebRequest downloads far faster

function Info($msg)  { Write-Host $msg -ForegroundColor Cyan }
function Ok($msg)    { Write-Host "  $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "  $msg" -ForegroundColor Yellow }
function Die($msg)   { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

Info "FluxInfer setup"
Info "==============="

# --- Prerequisites we can check but not provide --------------------------
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
  Warn "nvidia-smi not found. FluxInfer tunes NVIDIA GPUs; without a driver it will only see the CPU."
  Warn "Install the NVIDIA driver first if you have an NVIDIA card. Continuing anyway."
} else {
  $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
  Ok "GPU detected: $gpuName"
}

if ($PSVersionTable.PSVersion.Major -lt 5) { Die "PowerShell 5 or newer is required." }

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$binDir = Join-Path $InstallDir "bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

# --- Helper: fetch a release asset by name pattern -----------------------
function Get-ReleaseAsset {
  param([string]$Repo, [string]$Pattern, [string]$Destination)

  $api = "https://api.github.com/repos/$Repo/releases/latest"
  try {
    $release = Invoke-RestMethod -Uri $api -Headers @{ "User-Agent" = "fluxinfer-installer" }
  } catch {
    Die "could not query GitHub for $Repo releases: $($_.Exception.Message)"
  }
  $asset = $release.assets | Where-Object { $_.name -match $Pattern } | Select-Object -First 1
  if (-not $asset) {
    return $null
  }
  $target = Join-Path $env:TEMP $asset.name
  Info "Downloading $($asset.name) ($([math]::Round($asset.size / 1MB, 1)) MB)..."
  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $target -UseBasicParsing
  Expand-Archive -Path $target -DestinationPath $Destination -Force
  Remove-Item $target -ErrorAction SilentlyContinue
  return $asset.name
}

# --- FluxInfer itself ----------------------------------------------------
Info "Installing FluxInfer..."
$fluxAsset = Get-ReleaseAsset -Repo "federicobarrosgiuffrida/fluxinfer" -Pattern "windows-x64.*\.zip$" -Destination $InstallDir
if (-not $fluxAsset) {
  Die @"
no published Windows release found for FluxInfer.
Until one is tagged, build from source instead:
  git clone https://github.com/federicobarrosgiuffrida/fluxinfer
  cd fluxinfer
  cmake -S . -B build ; cmake --build build --config Release
"@
}
$fluxExe = Get-ChildItem -Path $InstallDir -Filter "fluxinfer.exe" -Recurse | Select-Object -First 1
if (-not $fluxExe) { Die "the downloaded archive did not contain fluxinfer.exe" }
Ok "FluxInfer: $($fluxExe.FullName)"

# --- llama.cpp binaries --------------------------------------------------
if (-not $SkipLlama) {
  Info "Installing llama.cpp (CUDA build)..."
  # llama.cpp publishes CUDA builds as llama-<tag>-bin-win-cuda-<ver>-x64.zip,
  # with the CUDA runtime DLLs in a separate cudart archive.
  $llamaDir = Join-Path $InstallDir "llama.cpp"
  New-Item -ItemType Directory -Force -Path $llamaDir | Out-Null

  $llama = Get-ReleaseAsset -Repo "ggml-org/llama.cpp" -Pattern "bin-win-cuda.*x64\.zip$" -Destination $llamaDir
  if (-not $llama) {
    Warn "no CUDA build found in the latest llama.cpp release; trying a CPU build instead."
    $llama = Get-ReleaseAsset -Repo "ggml-org/llama.cpp" -Pattern "bin-win-(cpu|avx2).*x64\.zip$" -Destination $llamaDir
  }
  if (-not $llama) {
    Die "could not find any suitable llama.cpp Windows build. Download one manually from
https://github.com/ggml-org/llama.cpp/releases and re-run with -SkipLlama."
  }

  if ($llama -match "cuda") {
    # Without these DLLs the CUDA binaries fail to start at all.
    $cudart = Get-ReleaseAsset -Repo "ggml-org/llama.cpp" -Pattern "cudart-llama.*x64\.zip$" -Destination $llamaDir
    if ($cudart) { Ok "CUDA runtime installed" }
    else { Warn "CUDA runtime archive not found; if llama-bench fails to start, download cudart-llama-*.zip manually." }
  }

  $llamaBench = Get-ChildItem -Path $llamaDir -Filter "llama-bench.exe" -Recurse | Select-Object -First 1
  if (-not $llamaBench) { Die "llama-bench.exe not found after extraction" }
  $env:FLUXINFER_LLAMA_DIR = $llamaBench.Directory.FullName
  [Environment]::SetEnvironmentVariable("FLUXINFER_LLAMA_DIR", $llamaBench.Directory.FullName, "User")
  Ok "llama.cpp: $($llamaBench.Directory.FullName)"
  Ok "FLUXINFER_LLAMA_DIR set for future sessions"
}

# --- Put fluxinfer on PATH ----------------------------------------------
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$($fluxExe.Directory.FullName)*") {
  [Environment]::SetEnvironmentVariable("Path", "$userPath;$($fluxExe.Directory.FullName)", "User")
  Ok "added to PATH (new terminals will find `fluxinfer` directly)"
}

# --- A model is the one thing this script cannot fetch for you -----------
Info "Checking for models..."
$modelDirs = @("$env:USERPROFILE\Documents\ai-models", "$env:USERPROFILE\models", "$env:USERPROFILE\.lmstudio\models")
$found = @()
foreach ($dir in $modelDirs) {
  if (Test-Path $dir) {
    $found += Get-ChildItem -Path $dir -Filter "*.gguf" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 5
  }
}
if ($found.Count -gt 0) {
  Ok "$($found.Count)+ model file(s) found"
} else {
  Warn "no .gguf model found. Download one (e.g. from huggingface.co) into $env:USERPROFILE\Documents\ai-models"
  Warn "and re-run 'fluxinfer menu'."
}

Write-Host ""
Info "Done. FluxInfer is installed in $InstallDir"
Write-Host ""

if (-not $NoLaunch -and $found.Count -gt 0) {
  Info "Opening the menu..."
  & $fluxExe.FullName menu
} else {
  Write-Host "Start with:  fluxinfer menu" -ForegroundColor White
}
