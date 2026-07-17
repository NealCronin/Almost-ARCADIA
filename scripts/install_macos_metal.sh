#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script is for macOS only."
    exit 1
fi

# Verify Python 3.11+
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
echo "Python $(python3 --version) found."

# Verify Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    echo "Xcode Command Line Tools not found."
    echo "Run: xcode-select --install"
    echo "Then re-run this script."
    exit 1
fi
echo "Xcode Command Line Tools found."

# Verify Clang
if ! command -v clang &>/dev/null; then
    echo "Clang not found. Verify Xcode Command Line Tools are installed."
    exit 1
fi
echo "Clang $(clang --version | head -1) found."

# Check Homebrew availability before using it
_have_brew() { command -v brew &>/dev/null; }
_missing_tools=()
for cmd in git cmake ninja; do
    if ! command -v "$cmd" &>/dev/null; then
        _missing_tools+=("$cmd")
    fi
done

if [[ ${#_missing_tools[@]} -gt 0 ]]; then
    if _have_brew; then
        echo "Installing missing tools: ${_missing_tools[*]}"
        brew install "${_missing_tools[@]}"
    else
        echo "Missing tools: ${_missing_tools[*]}"
        echo "Homebrew is not available. Install Homebrew from https://brew.sh, then re-run this script."
        exit 1
    fi
fi
echo "Git $(git --version | cut -d' ' -f3), CMake $(cmake --version | head -1 | cut -d' ' -f3), Ninja $(ninja --version) found."

# Create venv
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install Almost ARCADIA deps
pip install -e ".[all]"

# Clone/update llama.cpp
COMMIT="$(cat scripts/llama_cpp_commit.txt)"
COMMIT="${COMMIT//[$'\r\n\t ']/}"
if [[ ${#COMMIT} -ne 40 ]]; then
    echo "llama_cpp_commit.txt must contain a full 40-character commit SHA, got: '$COMMIT'"
    exit 1
fi

if [[ -d vendor/llama.cpp ]]; then
    cd vendor/llama.cpp
    git fetch origin
else
    mkdir -p vendor
    git clone https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp
    cd vendor/llama.cpp
fi
git checkout --detach "$COMMIT"

# Build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DCMAKE_OSX_ARCHITECTURES="arm64"
cmake --build . --config Release --target llama-server -j "$(sysctl -n hw.ncpu)"

# Verify
BINARY="bin/llama-server"
if [[ ! -f "$BINARY" ]]; then
    echo "Build failed: $BINARY not found"
    exit 1
fi
"$BINARY" --version

# Django checks
cd "$SCRIPT_DIR"
python manage.py migrate
python manage.py check

echo "Installation complete."
echo "Binary: $(pwd)/vendor/llama.cpp/build/$BINARY"
echo "Run: python manage.py runserver 127.0.0.1:8000"