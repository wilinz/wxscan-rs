# TFLite for the browser

`build.sh` produces `wxscan_tflite.js` and `wxscan_tflite.wasm`: the TensorFlow
Lite C runtime with the XNNPACK delegate, compiled to WebAssembly, plus the
small shim in [`host.cc`](host.cc) that
[`wxscan-ffi`'s host backend](../../crates/wxscan-ffi/src/host_net.rs) talks to.

The version matches the desktop build pinned in wxscan's `tool/tflite.lock`, so
a browser runs the same runtime, and the same `.tflite` weights, as every other
platform.

```sh
source /path/to/emsdk/emsdk_env.sh
./build.sh                       # writes ./out
```

Expect a quarter of an hour the first time: it clones TensorFlow, fetches a
dozen dependencies and compiles about a thousand XNNPACK microkernels.

## Why patches are needed

Upstream does not build this configuration, so `build.sh` applies two patches
after fetching. Both are in [`patches/`](patches).

| | |
|---|---|
| `0001-tensorflow-std-abs` | `std::abs<float>` stopped being a template in libc++ 19. TF 2.17 predates that; the emscripten toolchain does not. Two lines become lambdas. |
| `0002-xnnpack-wasm` | XNNPACK's CMake build has no wasm support: it generates the wasm microkernel lists and never wires them up, rejects `Emscripten` as a system name, and `src/xnnpack/math.h` calls `rint` without including `math.h`. Its wasm builds go through Bazel instead. |

Two more things the dependency builds get wrong on this target are handled in
the script rather than by patch, because they are single objects: flatbuffers
decides locale independence per translation unit and ends up not defining a
symbol TFLite references, and cpuinfo never compiles its emscripten backend.

## What it costs, and why it is worth it

Measured on a 1920x1080 scene image, one QR code, Node 22 on an M-series Mac:

| Detector input | Reference kernels | With XNNPACK + SIMD |
|---|---|---|
| 224x320 | 10.2 ms | 2.0 ms |
| 384x384 | 20.8 ms | 4.1 ms |
| 480x640 | 43.6 ms | 8.4 ms |

XNNPACK takes 137 of the detector's 139 nodes, leaving an execution plan of
three. The module grows from 2.1 MB to 3.0 MB, or 608 KB to 838 KB gzipped.

**The C API applies no delegate on its own** — `c_api.cc` never mentions
XNNPACK, unlike the C++ `InterpreterBuilder`, which applies it by default. It
has to be created and added explicitly, which is what `host.cc` does and what
the numbers above depend on.

## The alternative, and why not

`wxscan-wasm` can also compile tract into the module and run the ONNX weights
with no host at all. That module is 11.9 MB against this one's 3.0 MB, and the
same frame takes 347 ms rather than 332 ms. It needs no JavaScript, which is
its one real advantage.

LiteRT.js is not usable here: it refuses models with symbolic dimensions and
exposes no resize, while `detect.tflite` deliberately keeps height and width
symbolic. The TFLite C API has `TfLiteInterpreterResizeInputTensor`, which is
what makes this path work at all.
