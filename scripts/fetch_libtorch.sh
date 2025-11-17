#!/bin/bash
# Download the correct libtorch package (CPU / CUDA / ROCm, Linux / macOS / Windows).
# Usage: scripts/fetch_libtorch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY_DIR="$PROJECT_ROOT/third_party"
LIBTORCH_DIR="$THIRD_PARTY_DIR/libtorch"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: No python interpreter found for extraction step." >&2
  exit 1
fi

LIBTORCH_VERSION="${CHESSHACKS_LIBTORCH_VERSION:-2.3.0}"
VARIANT_OVERRIDE="${CHESSHACKS_LIBTORCH_VARIANT:-}"
OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH_NAME="$(uname -m | tr '[:upper:]' '[:lower:]')"

case "$OS_NAME" in
  linux*) PLATFORM="linux" ;;
  darwin*) PLATFORM="macos" ;;
  msys*|mingw*|cygwin*) PLATFORM="windows" ;;
  *) echo "ERROR: Unsupported platform '$OS_NAME'." >&2; exit 1 ;;
esac

detect_rocm_variant() {
  if [[ -n "${CHESSHACKS_LIBTORCH_ROCM_VERSION:-}" ]]; then
    echo "rocm${CHESSHACKS_LIBTORCH_ROCM_VERSION}"
    return
  fi
  local info_file="/opt/rocm/.info/version"
  if [[ -f "$info_file" ]]; then
    local ver
    ver=$(tr -d '\r' <"$info_file")
    ver="${ver// /}"
    if [[ -n "$ver" ]]; then
      echo "rocm${ver}"
      return
    fi
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    local ver
    ver=$(rocminfo | awk '/ROCm version/ {print $3; exit}')
    if [[ -n "$ver" ]]; then
      ver="${ver// /}"
      echo "rocm${ver}"
      return
    fi
  fi
  echo "rocm6.0"
}

detect_variant() {
  if [[ -n "$VARIANT_OVERRIDE" ]]; then
    echo "$VARIANT_OVERRIDE"
    return
  fi

  case "$PLATFORM" in
    macos)
      if [[ "$ARCH_NAME" == "arm64" || "$ARCH_NAME" == "aarch64" ]]; then
        echo "macos-arm64"
      else
        echo "macos-x86_64"
      fi
      ;;
    windows)
      if [[ "${CHESSHACKS_ENABLE_CUDA:-0}" != "0" ]]; then
        echo "cu121"
      else
        echo "cpu"
      fi
      ;;
    linux)
      if [[ "${CHESSHACKS_ENABLE_ROCM:-0}" != "0" ]]; then
        echo "$(detect_rocm_variant)"
      elif [[ "${CHESSHACKS_ENABLE_CUDA:-0}" != "0" ]]; then
        echo "cu121"
      elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "cu121"
      else
        echo "cpu"
      fi
      ;;
    *)
      echo "cpu"
      ;;
  esac
}

VARIANT="$(detect_variant)"
CHANNEL=""
ARCHIVE_NAME=""

case "$PLATFORM" in
  linux)
    case "$VARIANT" in
      cpu)
        CHANNEL="cpu"
        ARCHIVE_NAME="libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
        ;;
      cu11*|cu12*)
        CHANNEL="$VARIANT"
        ARCHIVE_NAME="libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${VARIANT}.zip"
        ;;
      rocm*)
        CHANNEL="rocm6.4"
        ARCHIVE_NAME="libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${VARIANT}.zip"
        ;;
      *)
        echo "ERROR: Unsupported linux variant '$VARIANT'." >&2
        exit 1
        ;;
    esac
    ;;
  macos)
    CHANNEL="macos"
    if [[ "$VARIANT" == "macos-arm64" ]]; then
      ARCHIVE_NAME="libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
    else
      ARCHIVE_NAME="libtorch-macos-${LIBTORCH_VERSION}.zip"
    fi
    ;;
  windows)
    case "$VARIANT" in
      cpu)
        CHANNEL="cpu"
        ARCHIVE_NAME="libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
        ;;
      cu11*|cu12*)
        CHANNEL="$VARIANT"
        ARCHIVE_NAME="libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2B${VARIANT}.zip"
        ;;
      *)
        echo "ERROR: Unsupported Windows variant '$VARIANT'." >&2
        exit 1
        ;;
    esac
    ;;
esac

DOWNLOAD_URL="https://download.pytorch.org/libtorch/${CHANNEL}/${ARCHIVE_NAME}"

echo "==============================================="
echo "Fetching libtorch headers"
echo "  Platform: ${PLATFORM} (${ARCH_NAME})"
echo "  Version : ${LIBTORCH_VERSION}"
echo "  Variant : ${VARIANT}"
echo "  URL     : ${DOWNLOAD_URL}"
echo "  Target  : ${LIBTORCH_DIR}"
echo "==============================================="

mkdir -p "$THIRD_PARTY_DIR"
TMP_ARCHIVE="$THIRD_PARTY_DIR/libtorch_download.zip"

if command -v curl >/dev/null 2>&1; then
  curl -L "$DOWNLOAD_URL" -o "$TMP_ARCHIVE"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TMP_ARCHIVE" "$DOWNLOAD_URL"
else
  echo "ERROR: Install curl or wget to download libtorch." >&2
  exit 1
fi

rm -rf "$LIBTORCH_DIR"

"$PYTHON_BIN" - <<PY
import zipfile, pathlib
zip_path = pathlib.Path("$TMP_ARCHIVE")
target_dir = pathlib.Path("$THIRD_PARTY_DIR")
with zipfile.ZipFile(zip_path, "r") as archive:
    archive.extractall(target_dir)
print("✓ Extracted libtorch to", target_dir / "libtorch")
PY

rm -f "$TMP_ARCHIVE"

EXPECTED_HEADER="$LIBTORCH_DIR/include/torch/script.h"
if [[ ! -f "$EXPECTED_HEADER" ]]; then
  echo "ERROR: libtorch extraction failed; missing $EXPECTED_HEADER" >&2
  exit 1
fi

echo "✓ libtorch ready at $LIBTORCH_DIR"

