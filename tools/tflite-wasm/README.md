# TFLite for the browser

`build.sh` produces `wxscan_tflite.js` and `wxscan_tflite.wasm`: the TensorFlow
Lite C runtime with the XNNPACK delegate, compiled to WebAssembly, plus the
small shim in [`host.cc`](host.cc) that
[`wxscan-ffi`'s host backend](../../crates/wxscan-ffi/src/host_net.rs) talks to.

The version lives in [`depversion.toml`](../../depversion.toml) at the top of
the repository, which both `build.sh` and the CI workflow read — one place,
because it used to be two and nothing made them agree, and at the root because
it is a fact about the library rather than about this build. It has to match
the desktop build pinned in wxscan's `tool/tflite.lock`, so that a browser runs
the same runtime, and the same `.tflite` weights, as every other platform.

Beside it is `patch`, which is ours rather than TensorFlow's: the revision of
the patches below and of the script that applies them. Upstream does not build
this configuration, so what comes out of a given version is decided as much by
those as by the version, and they change while it stays put. Raise it when they
do.

Changing either is the whole of a version bump. Tagging `tflite-<version>-p<patch>`
builds it and publishes it as that tag's release; a tag naming anything else
fails rather than putting bytes under a name that does not describe them.

This gets a release of its own rather than riding along in the scanner's,
because it is a dependency on its own rhythm — it changes a few times a year
where the scanner changes daily, and a copy inside every `v*` release would be
1.3 MB of the same bytes over and over. Downstream pins the two separately;
wxscan's `tool/web.lock` has a line for each.

```sh
source /path/to/emsdk/emsdk_env.sh
./build.sh                       # writes ./out
```

Expect a quarter of an hour the first time: it clones TensorFlow, fetches a
dozen dependencies and compiles about a thousand XNNPACK microkernels. The
first `cmake` of a fresh tree is *expected to fail* — see the patches below —
and the script carries on past it deliberately.

    ./build.sh --ops        # builds dump_ops instead, for when weights change

## Why patches are needed

Upstream does not build this configuration, so `build.sh` applies two patches
after fetching. Both are in [`patches/`](patches).

| | |
|---|---|
| `0001-tensorflow-std-abs` | `std::abs<float>` stopped being a template in libc++ 19. TF 2.17 predates that; the emscripten toolchain does not. Two lines become lambdas. |
| `0002-xnnpack-wasm` | XNNPACK's CMake build has no wasm support: it generates the wasm microkernel lists and never wires them up, rejects `Emscripten` as a system name, and `src/xnnpack/math.h` calls `rint` without including `math.h`. Its wasm builds go through Bazel instead. |

That second one is why the script configures twice. XNNPACK arrives during a
configure and rejects the target while it is still unpatched, so the first pass
is allowed to fail once it has fetched what the patch needs. It is also applied
on every run rather than once: FetchContent checks the dependency out again
each time, taking local changes with it.

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
three.

**Nothing applies the delegate on its own here.** The C API never mentions
XNNPACK at all — `c_api.cc` has no reference to it — and the C++ builder only
applies it through `BuiltinOpResolver`, which is the one thing `host.cc` avoids
using. So it is created and attached by name. Miss that and the module still
works, still decodes correctly, and quietly runs the reference kernels: the
only outward sign is the missing `Created TensorFlow Lite XNNPACK delegate`
line, since inference is a small enough part of a frame to hide four times its
own cost.

### Size

| | Module | Gzipped |
|---|---|---|
| Through the C API, every operator registered | 2.90 MB | 838 KB |
| Registering the sixteen these models use | **1.34 MB** | **418 KB** |

`-Oz`, LTO and a smaller allocator are worth about 70 KB between them: the
archives are compiled at `-O3` before the link ever sees them, so the size is
in what gets linked, not how. `BuiltinOpResolver` registers about 150
operators, and registering one links its kernel. See the note at the top of
`host.cc`, and `--ops` for regenerating the list when the weights change.

## The alternative, and why not

`wxscan-wasm` can also compile tract into the module and run the ONNX weights
with no host at all. That module is 11.9 MB against this one's 3.0 MB, and the
same frame takes 347 ms rather than 332 ms. It needs no JavaScript, which is
its one real advantage.

LiteRT.js is not usable here: it refuses models with symbolic dimensions and
exposes no resize, while `detect.tflite` deliberately keeps height and width
symbolic. The TFLite C API has `TfLiteInterpreterResizeInputTensor`, which is
what makes this path work at all.
