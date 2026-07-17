#Requires -Version 5.1
param(
    [string]$CudaArchitectures = ""
)
$ErrorActionPreference = "Stop"

# Verify 64-bit Windows
if (-not [Environment]::Is64BitOperatingSystem) {
    Write-Error "This script requires 64-bit Windows."
    exit 1
}
Write-Host "64-bit Windows: verified."

# Verify Python 3.11+
$pyVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found. Install Python 3.11+ from https://python.org."
    exit 1
}
$pyVersionStr = $pyVersion.ToString()
if ($pyVersionStr -notmatch '3\.(1[1-9]|[2-9]\d)') {
    Write-Error "Python 3.11+ required, found: $pyVersionStr"
    exit 1
}
Write-Host "Python: $pyVersionStr"

# Verify Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git not found. Install Git from https://git-scm.com or use: winget install Git.Git"
    exit 1
}
Write-Host "Git: found."

# Verify CMake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "CMake not found. Install from https://cmake.org or use: winget install Kitware.CMake"
    exit 1
}
Write-Host "CMake: found."

# Verify Ninja
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    Write-Error "Ninja not found. Install from https://ninja-build.org or use: winget install Ninja-build.Ninja"
    exit 1
}
Write-Host "Ninja: found."

# Verify Visual Studio 2022 C++ Build Tools
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "Visual Studio 2022 not found. Install Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload."
    exit 1
}
$vsInstallPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Workload.NativeDesktop -property installationPath
if (-not $vsInstallPath) {
    Write-Error "Visual Studio 2022 C++ Build Tools with 'Desktop development with C++' workload required."
    exit 1
}
Write-Host "Visual Studio 2022 found at: $vsInstallPath"

# Verify CUDA nvcc
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Error "CUDA nvcc not found. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads."
    exit 1
}
$nvccVersion = & nvcc --version 2>&1 | Select-Object -Last 1
Write-Host "CUDA: $nvccVersion"

# Resolve script directory
$ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$ProjectDir = Split-Path -Path $ScriptDir -Parent
Set-Location $ProjectDir

# Create venv
$venvPath = Join-Path $ProjectDir ".venv"
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}
# Activate venv (PowerShell)
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
. $activateScript

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install Almost ARCADIA deps
pip install -e ".[all]"

# Clone/update llama.cpp
$commitFile = Join-Path $ScriptDir "llama_cpp_commit.txt"
$commit = Get-Content $commitFile -Raw | ForEach-Object { $_.Trim() }
if ($commit.Length -ne 40) {
    Write-Error "llama_cpp_commit.txt must contain a full 40-character commit SHA, got: '$commit'"
    exit 1
}

$vendorDir = Join-Path $ProjectDir "vendor"
$llamaDir = Join-Path $vendorDir "llama.cpp"
if (Test-Path $llamaDir) {
    Set-Location $llamaDir
    git fetch origin
} else {
    New-Item -ItemType Directory -Path $vendorDir -Force | Out-Null
    git clone https://github.com/ggerganov/llama.cpp.git $llamaDir
    Set-Location $llamaDir
}
git checkout --detach $commit

# Build with CUDA
$buildDir = Join-Path $llamaDir "build"
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
Set-Location $buildDir

# Use VS developer prompt for cmake
$vsDevCmd = Join-Path $vsInstallPath "Common7\Tools\VsDevCmd.bat"
$buildArgs = @(
    "..", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON"
)
if ($CudaArchitectures) {
    $buildArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}
$cmakeCmd = "`"$vsDevCmd`" -arch=amd64 -host_arch=amd64 && cmake $($buildArgs -join ' ')"
Invoke-Expression "cmd /c $cmakeCmd"
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed. Verify CUDA and Visual Studio installations."
    exit 1
}

cmake --build . --config Release --target llama-server
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed."
    exit 1
}

# Verify
$binary = Join-Path $buildDir "bin\Release\llama-server.exe"
if (-not (Test-Path $binary)) {
    $binary = Join-Path $buildDir "bin\llama-server.exe"
}
if (-not (Test-Path $binary)) {
    Write-Error "Build succeeded but llama-server.exe not found."
    exit 1
}
& $binary --version

# Django checks
Set-Location $ProjectDir
python manage.py migrate
python manage.py check

Write-Host "Installation complete."
Write-Host "Binary: $binary"
Write-Host "Run: python manage.py runserver 127.0.0.1:8000"