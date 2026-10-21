#!/usr/bin/env bash
# Extract headless GL/GLES runtime libs into a user prefix (no sudo needed).
# MediaPipe (CPU delegate) requires libGLESv2.so.2 even on headless servers.
set -euo pipefail

PREFIX="${GL_PREFIX:-$HOME/.local/gl-libs}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

mkdir -p "$PREFIX"
cd "$WORKDIR"

PKGS=(libgles2 libegl1 libgl1 libglvnd0)
if apt-get download "${PKGS[@]}" 2>/dev/null; then
  for deb in ./*.deb; do
    dpkg -x "$deb" "$PREFIX"
  done
  echo "GL libs extracted to $PREFIX"
else
  echo "apt-get download failed; trying pip PyOpenGL is NOT sufficient for mediapipe." >&2
  echo "Install via sudo: sudo apt-get install -y libgles2 libegl1 libgl1" >&2
  exit 1
fi

LIBDIR="$PREFIX/usr/lib/x86_64-linux-gnu"
if [ ! -e "$LIBDIR/libGLESv2.so.2" ]; then
  echo "libGLESv2.so.2 still missing after extraction" >&2
  exit 1
fi
echo "Add to your environment before running the server/benchmark:"
echo "  export LD_LIBRARY_PATH=$LIBDIR:\$LD_LIBRARY_PATH"
