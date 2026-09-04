#!/usr/bin/env bash
# build_apps.sh — Build ISPASS-2009 apps natively for sm_89 (CUDA 12.x)
# using the minimal cutil shim in port/cutil/. Phase 0 scope: LPS, LIB, NQU.
#
# Usage: bash ispass2009-benchmarks/port/build_apps.sh [lps|lib|nqu ...]
set -euo pipefail

PORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISPASS_DIR="$(cd "$PORT_DIR/.." && pwd)"
BIN_DIR="$PORT_DIR/bin"
CUBIN_DIR="$PORT_DIR/cubin"

mkdir -p "$BIN_DIR" "$CUBIN_DIR"

NVCC_FLAGS=(-O2 -arch=sm_89 -I "$PORT_DIR/cutil" -Xcompiler -Wno-deprecated-declarations)

build_one() {
    local app="$1"; shift
    local srcdir="$1"; shift
    echo "== Building $app =="
    nvcc "${NVCC_FLAGS[@]}" -I "$srcdir" -o "$BIN_DIR/$app" "$@" 2>&1
    # also emit a standalone cubin for the static-patch / SASS analysis path
    local cu_sources=()
    for src in "$@"; do
        [[ "$src" == *.cu ]] && cu_sources+=("$src")
    done
    nvcc -O2 -arch=sm_89 -I "$PORT_DIR/cutil" -I "$srcdir" -cubin \
        -o "$CUBIN_DIR/$app.cubin" "${cu_sources[@]}" 2>&1
    echo "   bin : $BIN_DIR/$app"
    echo "   cubin: $CUBIN_DIR/$app.cubin"
}

APPS=("$@")
[ ${#APPS[@]} -eq 0 ] && APPS=(lps lib nqu)

for app in "${APPS[@]}"; do
    case "$app" in
        lps)
            build_one LPS "$ISPASS_DIR/LPS" "$ISPASS_DIR/LPS/laplace3d.cu" "$ISPASS_DIR/LPS/laplace3d_gold.cpp"
            ;;
        lib)
            build_one LIB "$ISPASS_DIR/LIB" "$ISPASS_DIR/LIB/libor.cu"
            ;;
        nqu)
            build_one NQU "$ISPASS_DIR/NQU" "$ISPASS_DIR/NQU/nqueen.cu"
            ;;
        *)
            echo "Unknown app: $app (Phase 0 scope: lps lib nqu)" >&2
            exit 1
            ;;
    esac
done

echo "All requested apps built."
