#!/usr/bin/env bash
# Build the TensorFlow Lite C runtime, plus the host shim wxscan talks to,
# as a WebAssembly module for browsers.
#
#   ./build.sh [output directory]
#   ./build.sh --ops             build dump_ops instead, to list a model's
#                                operators when the weights change
#
# Needs emscripten and cmake. The version comes from depversion.toml at the top
# of the repository, and has to match the desktop build pinned in wxscan's
# tool/tflite.lock, so that the browser runs the same runtime as every other
# platform. TENSORFLOW_VERSION overrides it, for trying one without committing
# to it.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

DEPVERSION="$HERE/../../depversion.toml"

# One key out of one section, read with sed rather than a TOML parser: a build
# script should not need one, and the file is kept small enough that this is
# honest. It does mean the section header is not checked, so keep it that way.
tflite_config_version() {
  sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' \
    "$DEPVERSION" | head -1
}

TENSORFLOW_VERSION="${TENSORFLOW_VERSION:-$(tflite_config_version)}"
if [ -z "$TENSORFLOW_VERSION" ]; then
  echo "no [tensorflow] version in $DEPVERSION, and TENSORFLOW_VERSION is not set" >&2
  exit 1
fi
MODE=run
if [ "${1:-}" = "--ops" ]; then MODE=ops; shift; fi
OUT="${1:-$HERE/out}"
WORK="${WXSCAN_WASM_WORKDIR:-$HERE/.work}"
EMSDK="${EMSDK:-}"

if [ -z "$EMSDK" ] && ! command -v emcc >/dev/null; then
  echo "emscripten not found: install emsdk and 'source emsdk_env.sh', or set EMSDK" >&2
  exit 1
fi
EMCC="${EMSDK:+$EMSDK/upstream/emscripten/}"

# Where the build happened must not end up in what it produced. TFLite's
# TF_LITE_KERNEL_LOG and TFLITE_DCHECK put __FILE__ in the binary, so without
# this the module carries the absolute path of whoever built it — 29 of them,
# about 2.6 KB — and the same sources at the same version give different bytes
# from a different directory. Under clang -ffile-prefix-map covers __FILE__ as
# well as debug info, so one flag does it.
#
# The more specific mapping comes first, because $WORK is normally inside
# $HERE. Either way round both land somewhere fixed: with first-match $WORK
# wins, and with last-match $HERE rewrites its own subtree to the same place.
#
# This makes a rebuild independent of the directory, not reproducible outright.
# The emscripten version still decides the rest, and it is recorded in
# wxscan_tflite.build beside the module.
PREFIX_MAP="-ffile-prefix-map=$WORK=/wxscan/work -ffile-prefix-map=$HERE=/wxscan/src"

mkdir -p "$WORK" "$OUT"

# ---- sources ---------------------------------------------------------------
if [ ! -d "$WORK/tensorflow" ]; then
  echo "== cloning tensorflow $TENSORFLOW_VERSION"
  git clone -q --depth 1 --branch "$TENSORFLOW_VERSION" \
    https://github.com/tensorflow/tensorflow.git "$WORK/tensorflow"
  # `std::abs<float>` is not a template in libc++ 19 and later; the emscripten
  # toolchain is well past that, while TF 2.17 is not.
  git -C "$WORK/tensorflow" apply "$HERE/patches/0001-tensorflow-std-abs.patch"
fi

# ---- configure -------------------------------------------------------------
# The thread overrides are for abseil, which insists on find_package(Threads)
# even where there are none. XNNPACK needs SIMD to be worth having.
configure() {
  "${EMCC}emcmake" cmake -S "$WORK/tensorflow/tensorflow/lite/c" -B "$WORK/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DTFLITE_ENABLE_MMAP=OFF \
    -DTFLITE_C_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_HAVE_LIBC_PTHREAD=1 \
    -DTHREADS_PREFER_PTHREAD_FLAG=OFF \
    -DCMAKE_THREAD_LIBS_INIT= \
    -DCMAKE_USE_PTHREADS_INIT=1 \
    -DCMAKE_C_FLAGS="-msimd128 $PREFIX_MAP" \
    -DCMAKE_CXX_FLAGS="-msimd128 $PREFIX_MAP" >/dev/null
}

# XNNPACK's CMake build has no wasm support at all upstream: it generates the
# wasm microkernel lists and never wires them up, rejects `Emscripten` as a
# system name, and `src/xnnpack/math.h` calls rint() without including math.h.
# Its wasm builds go through Bazel instead.
#
# So the first configure is expected to fail — it stops at exactly that
# rejection — and its job is only to fetch XNNPACK so the patch has something
# to apply to. `git apply --check` first is what makes a second run harmless,
# and re-patching every time is necessary rather than tidy: FetchContent checks
# the dependency out again on each configure and takes local changes with it.
echo "== configuring"
configure || true

repo="$WORK/build/xnnpack"
if [ -d "$repo" ] && git -C "$repo" apply --check "$HERE/patches/0002-xnnpack-wasm.patch" 2>/dev/null; then
  echo "== patching xnnpack for wasm"
  git -C "$repo" apply "$HERE/patches/0002-xnnpack-wasm.patch"
fi
configure

# ---- build -----------------------------------------------------------------
# The C++ library, not the C one: host.cc registers operators by hand, and
# going through the C API would link every kernel TensorFlow Lite has. See the
# note at the top of host.cc.
echo "== building tensorflow-lite (this takes a while)"
cmake --build "$WORK/build" -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" --target tensorflow-lite

# Two objects the dependency builds get wrong on this target: flatbuffers
# decides on locale independence per translation unit, and cpuinfo never
# compiles its emscripten backend.
"${EMCC}em++" -c "$WORK/build/flatbuffers/src/util.cpp" \
  -I"$WORK/build/flatbuffers/include" -DFLATBUFFERS_LOCALE_INDEPENDENT=1 \
  -O2 -msimd128 $PREFIX_MAP -o "$WORK/fb_util.o"
"${EMCC}emcc" -c "$WORK/build/cpuinfo/src/emscripten/init.c" \
  -I"$WORK/build/cpuinfo/include" -I"$WORK/build/cpuinfo/src" \
  -DCPUINFO_LOG_LEVEL=2 -O2 $PREFIX_MAP -o "$WORK/cpuinfo_em.o"

# `-Oz` buys little on its own, the archives having been compiled at -O3
# already; the size is in what gets linked. `ENVIRONMENT=worker` is what the
# module runs in, and emmalloc is the smaller allocator. The name matters:
# emscripten compiles the .wasm file name into its loader, so the pair cannot
# be renamed afterwards.
LIBS=$(find "$WORK/build/_deps" "$WORK/build/xnnpack" "$WORK/build/pthreadpool" -name "*.a" 2>/dev/null | tr '\n' ' ')
COMMON="-I$WORK/tensorflow -I$WORK/build/flatbuffers/include
  $WORK/fb_util.o $WORK/cpuinfo_em.o
  $WORK/build/tensorflow-lite/libtensorflow-lite.a $LIBS
  -Oz -msimd128 $PREFIX_MAP -s MODULARIZE=1 -s EXPORT_ES6=1
  -s ALLOW_MEMORY_GROWTH=1 -s STACK_SIZE=8MB -s INITIAL_MEMORY=64MB
  -s MALLOC=emmalloc -s FILESYSTEM=0"

if [ "$MODE" = ops ]; then
  echo "== linking dump_ops"
  # No ENVIRONMENT=worker here: this one is meant to be run under node.
  "${EMCC}em++" "$HERE/dump_ops.cc" $COMMON -o "$OUT/dump_ops.js" \
    -s EXPORTED_RUNTIME_METHODS=HEAPU8 \
    -s EXPORTED_FUNCTIONS=_dump_ops,_malloc,_free
else
  echo "== linking the host module"
  "${EMCC}em++" "$HERE/host.cc" $COMMON -o "$OUT/wxscan_tflite.js" \
    -s ENVIRONMENT=worker \
    -s EXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 \
    -s EXPORTED_FUNCTIONS=_tf_load,_tf_prepare,_tf_invoke,_tf_out_ptr,_tf_out_floats,_tf_out_rank,_tf_out_dim,_malloc,_free
fi

ls -l "$OUT"
