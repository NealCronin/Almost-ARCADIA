#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="$(tr -d '[:space:]' < "$ROOT/scripts/llama_cpp_commit.txt")"

command -v python3 >/dev/null || { echo "Python 3.11+ is required."; exit 1; }
command -v xcode-select >/dev/null || { echo "Install Xcode Command Line Tools."; exit 1; }
command -v cmake >/dev/null || { echo "Install CMake (for example: brew install cmake)."; exit 1; }
command -v git >/dev/null || { echo "Git is required."; exit 1; }

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required.")
PY

cd "$ROOT"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

if [[ ! -d vendor/llama.cpp/.git ]]; then
    mkdir -p vendor
    git clone https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
fi

git -C vendor/llama.cpp fetch --tags --prune
git -C vendor/llama.cpp checkout --detach "$COMMIT"
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build --config Release -j "$(sysctl -n hw.logicalcpu)"

python manage.py check
python -m pytest -q
echo "Installed: $ROOT/vendor/llama.cpp/build/bin/llama-server"
