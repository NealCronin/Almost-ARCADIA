$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Commit = (Get-Content (Join-Path $PSScriptRoot "llama_cpp_commit.txt") -Raw).Trim()

if (-not [Environment]::Is64BitOperatingSystem) { throw "64-bit Windows is required." }
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python 3.11+ is required." }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw "CMake is required." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is required." }
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) { throw "CUDA Toolkit (nvcc) is required." }

Set-Location $Root
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

if (-not (Test-Path .\vendor\llama.cpp\.git)) {
    New-Item -ItemType Directory -Force .\vendor | Out-Null
    git clone https://github.com/ggml-org/llama.cpp.git .\vendor\llama.cpp
}

git -C .\vendor\llama.cpp fetch --tags --prune
git -C .\vendor\llama.cpp checkout --detach $Commit
cmake -S .\vendor\llama.cpp -B .\vendor\llama.cpp\build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build .\vendor\llama.cpp\build --config Release --parallel

& .\.venv\Scripts\python.exe manage.py check
& .\.venv\Scripts\python.exe -m pytest -q
Write-Host "Installed: $Root\vendor\llama.cpp\build\bin\Release\llama-server.exe"
