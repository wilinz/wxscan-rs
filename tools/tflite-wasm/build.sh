#!/usr/bin/env bash
# Build the TensorFlow Lite C runtime, plus the host shim wxscan talks to,
# as a WebAssembly module for browsers.
#
#   ./build.sh [output directory]
#
# Needs emscripten and cmake. TENSORFLOW_VERSION matches the desktop build
# pinned in wxscan's tool/tflite.lock, so the browser runs the same runtime as
# every other platform.
set -euo pipefail

TENSORFLOW_VERSION="${TENSORFLOW_VERSION:-v2.17.1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/out}"
WORK="${WXSCAN_WASM_WORKDIR:-$HERE/.work}"
EMSDK="${EMSDK:-}"

if [ -z "$EMSDK" ] && ! command -v emcc >/dev/null; then
  echo "emscripten not found: install emsdk and 'source emsdk_env.sh', or set EMSDK" >&2
  exit 1
fi
EMCC="${EMSDK:+$EMSDK/upstream/emscripten/}"

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
echo "== configuring"
"${EMCC}emcmake" cmake -S "$WORK/tensorflow/tensorflow/lite/c" -B "$WORK/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_ENABLE_XNNPACK=ON \
  -DTFLITE_ENABLE_MMAP=OFF \
  -DTFLITE_C_BUILD_SHARED_LIBS=OFF \
  -DCMAKE_HAVE_LIBC_PTHREAD=1 \
  -DTHREADS_PREFER_PTHREAD_FLAG=OFF \
  -DCMAKE_THREAD_LIBS_INIT= \
  -DCMAKE_USE_PTHREADS_INIT=1 \
  -DCMAKE_C_FLAGS="-msimd128" \
  -DCMAKE_CXX_FLAGS="-msimd128" >/dev/null

# XNNPACK is fetched by that configure step, and its CMake build has no wasm
# support at all upstream: the microkernel lists are generated but never wired
# up, and one header calls rint() without including math.h. Patch, then let the
# configure re-run pick it up.
if [ ! -f "$WORK/.xnnpack-patched" ]; then
  echo "== patching xnnpack for wasm"
  git -C "$WORK/build/xnnpack" apply "$HERE/patches/0002-xnnpack-wasm.patch"
  touch "$WORK/.xnnpack-patched"
  "${EMCC}emcmake" cmake -S "$WORK/tensorflow/tensorflow/lite/c" -B "$WORK/build" >/dev/null
fi

# ---- build -----------------------------------------------------------------
echo "== building tensorflowlite_c (this takes a while)"
cmake --build "$WORK/build" -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" --target tensorflowlite_c

# Two objects the dependency builds get wrong on this target: flatbuffers
# decides on locale independence per translation unit, and cpuinfo never
# compiles its emscripten backend.
"${EMCC}em++" -c "$WORK/build/flatbuffers/src/util.cpp" \
  -I"$WORK/build/flatbuffers/include" -DFLATBUFFERS_LOCALE_INDEPENDENT=1 \
  -O2 -msimd128 -o "$WORK/fb_util.o"
"${EMCC}emcc" -c "$WORK/build/cpuinfo/src/emscripten/init.c" \
  -I"$WORK/build/cpuinfo/include" -I"$WORK/build/cpuinfo/src" \
  -DCPUINFO_LOG_LEVEL=2 -O2 -o "$WORK/cpuinfo_em.o"

echo "== linking the host module"
"${EMCC}em++" "$HERE/host.cc" "$WORK/fb_util.o" "$WORK/cpuinfo_em.o" \
  -I"$WORK/tensorflow" -I"$WORK/build/flatbuffers/include" \
  "$WORK/build/libtensorflowlite_c.a" \
  "$WORK/build/tensorflow-lite/libtensorflow-lite.a" \
  $(find "$WORK/build/_deps" "$WORK/build/xnnpack" "$WORK/build/pthreadpool" -name "*.a" 2>/dev/null | tr '\n' ' ') \
  -O2 -msimd128 -o "$OUT/wxscan_tflite.js" \
  -s MODULARIZE=1 -s EXPORT_ES6=1 \
  -s EXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 \
  -s EXPORTED_FUNCTIONS=_tf_load,_tf_prepare,_tf_invoke,_tf_out_ptr,_tf_out_floats,_tf_out_rank,_tf_out_dim,_malloc,_free \
  -s ALLOW_MEMORY_GROWTH=1 -s STACK_SIZE=8MB -s INITIAL_MEMORY=64MB

ls -l "$OUT"
