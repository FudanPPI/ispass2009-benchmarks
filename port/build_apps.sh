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

# Emit only the standalone cubin (for apps whose host side needs legacy
# toolchains, e.g. WP's Fortran driver). -cubin is a non-link phase: one
# input file per invocation, so multi-.cu apps get one cubin per source.
build_cubin_only() {
    local app="$1"; shift
    local srcdir="$1"; shift
    echo "== Building $app (cubin only) =="
    local cu_sources=() extra_flags=() pending_inc=0
    for arg in "$@"; do
        if [ "$pending_inc" -eq 1 ]; then
            extra_flags+=("-I$arg")   # value of a preceding -I (keep attached)
            pending_inc=0
        elif [[ "$arg" == *.cu ]]; then
            cu_sources+=("$arg")
        elif [[ "$arg" == "-I" ]]; then
            pending_inc=1
        elif [[ "$arg" == -* ]]; then
            extra_flags+=("$arg")
        fi  # non-flag non-.cu args (host-only sources) skipped for cubin
    done
    local rc=0
    for src in "${cu_sources[@]}"; do
        local base out
        base="$(basename "$src" .cu)"
        if [ "${#cu_sources[@]}" -eq 1 ]; then
            out="$CUBIN_DIR/$app.cubin"
        else
            out="$CUBIN_DIR/${app}_${base}.cubin"
        fi
        nvcc -O2 -arch=sm_89 -I "$PORT_DIR/cutil" -I "$srcdir" \
            "${extra_flags[@]}" -cubin -o "$out" "$src" 2>&1 || rc=1
        echo "   cubin: $out"
    done
    return $rc
}

build_one() {
    local app="$1"; shift
    local srcdir="$1"; shift
    echo "== Building $app =="
    # tolerate link failure (e.g. DG's MPI/ParMetis deps) so the cubin
    # step below still runs — cubin is the essential simulator artifact
    if ! nvcc "${NVCC_FLAGS[@]}" -I "$srcdir" -o "$BIN_DIR/$app" "$@" 2>&1; then
        echo "   WARNING: $app native link failed (bin skipped, cubin still built)"
    else
        echo "   bin : $BIN_DIR/$app"
    fi
    # also emit standalone cubin(s) for the static-patch / SASS analysis path
    build_cubin_only "$app" "$srcdir" "$@"
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
        aes)
            build_one AES "$ISPASS_DIR/AES" "$ISPASS_DIR/AES/aesHost.cu" \
                "$ISPASS_DIR/AES/aescuda.cpp" "$ISPASS_DIR/AES/aesCudaUtils.cpp"
            ;;
        bfs)
            # bfs.cu includes kernel.cu — compile the top-level only
            build_one BFS "$ISPASS_DIR/BFS" "$ISPASS_DIR/BFS/bfs.cu"
            ;;
        cp)
            build_one CP "$ISPASS_DIR/CP/benchmarks/cp/src/cuda" \
                -I "$ISPASS_DIR/CP/common/include" \
                "$ISPASS_DIR/CP/benchmarks/cp/src/cuda/main.cu" \
                "$ISPASS_DIR/CP/benchmarks/cp/src/cuda/cuenergy_pre8_coalesce.cu" \
                "$ISPASS_DIR/CP/common/src/parboil.c"
            ;;
        dg)
            build_one DG "$ISPASS_DIR/DG/src" \
                -I "$ISPASS_DIR/DG/include" -DNDG3d \
                "$ISPASS_DIR/DG/src/MaxwellsKernel3d.cu" \
                "$ISPASS_DIR/DG/src/tictoc.cu"
            ;;
        mum)
            # mummergpu.cu includes common.cu + mummergpu_kernel.cu
            build_one MUM "$ISPASS_DIR/MUM" \
                "$ISPASS_DIR/MUM/mummergpu.cu" \
                "$ISPASS_DIR/MUM/mummergpu_main.cpp" \
                "$ISPASS_DIR/MUM/mummergpu_gold.cpp" \
                "$ISPASS_DIR/MUM/suffix-tree.cpp" \
                "$ISPASS_DIR/MUM/PoolMalloc.cpp"
            ;;
        nn)
            # NN.cu includes NN_kernel.cu
            build_one NN "$ISPASS_DIR/NN" "$ISPASS_DIR/NN/NN.cu"
            ;;
        ray)
            # rayTracing.cu includes rayTracing_kernel.cu; needs cutil_math.h shim
            build_one RAY "$ISPASS_DIR/RAY" "$ISPASS_DIR/RAY/rayTracing.cu" \
                "$ISPASS_DIR/RAY/makebmp.cpp" "$ISPASS_DIR/RAY/EasyBMP.cpp"
            ;;
        sto)
            # original makefile builds main.cu + storeGPU.cu (kernels are
            # pulled in via headers); md5/sha1 kernel .cu are only deps
            build_one STO "$ISPASS_DIR/STO" "$ISPASS_DIR/STO/main.cu" \
                "$ISPASS_DIR/STO/storeGPU.cu" "$ISPASS_DIR/STO/storeCPU.cpp" \
                "$ISPASS_DIR/STO/md5_cpu.cpp" "$ISPASS_DIR/STO/sha1_cpu.cpp"
            ;;
        wp)
            # WP is an M4-macro pipeline (m4 | spt.pl | sed) with a Fortran
            # driver — build the generated .cu to cubin only. m4 needs cwd
            # at WP_DIR for include(debug.m4).
            WP_DIR="$ISPASS_DIR/WP"
            (cd "$WP_DIR" && m4 wsm5.cu | sed 's/float/float/g' > wsm5.f.cu)
            (cd "$WP_DIR" && m4 wsm5_gpu.cu | perl spt.pl | sed 's/float/float/g' > wsm5_gpu.f.cu)
            build_cubin_only WP "$WP_DIR" -DCUDA -DXXX=8 -DYYY=8 -DMKX=28 \
                -DDEBUG_I=59 -DDEBUG_J=45 -DDEBUG_K=1 \
                "$WP_DIR/wsm5.f.cu" "$WP_DIR/wsm5_gpu.f.cu"
            ;;
        *)
            echo "Unknown app: $app (available: lps lib nqu aes bfs cp dg mum nn ray sto wp)" >&2
            exit 1
            ;;
    esac
done

echo "All requested apps built."
