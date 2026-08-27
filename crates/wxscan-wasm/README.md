# wxscan-wasm

The [wxscan](../wxscan) C ABI as a WebAssembly module, for browsers. The
artifact is a `.wasm` file; this crate is not published.

```sh
# The small build: the host runs inference (see below)
cargo build -p wxscan-wasm --target wasm32-unknown-unknown --profile wasm

# The self-contained build: tract runs the ONNX weights inside the module
cargo build -p wxscan-wasm --target wasm32-unknown-unknown --profile wasm \
  --no-default-features --features tract
```

`RUSTFLAGS="-C target-feature=+simd128"` is worth setting for either: it cost
nothing in correctness and took about 28% off scanning time when measured.

## Two backends, and what they cost

Measured on a 1920x1080 scene image with one QR code in it, in Node 22 on an
M-series Mac, with simd128 on:

| | Module | Gzipped | Scan | Needs |
|---|---|---|---|---|
| host (default) | 433 KB | 221 KB | — | an engine in the host, reached through two imports |
| `tract` | 12.5 MB | 2.9 MB | 347 ms | nothing |
| no models at all | 242 KB | 155 KB | 20 ms | nothing, and finds far fewer symbols |

The gap between 20 ms and 347 ms is the two CNN stages, which is to say that
almost all of the time is inference, and the point of the host backend is to
send that part somewhere it can go faster — LiteRT.js on WebGPU runs the same
`.tflite` weights the native build uses.

The 12 MB is an ONNX runtime, not this algorithm: the scanner, the decoder and
the imgproc functions together are the 242 KB row.

## What the module expects from its host

Every build exports `malloc` and `free`, because a wasm module has no other way
to be handed an image, and its `memory`. The rest of the exports are the C ABI
in [`include/wxscan.h`](../wxscan-ffi/include/wxscan.h), unchanged.

The default build additionally **imports** two functions in the module
`wxscan`, and will not instantiate without them:

| Import | |
|---|---|
| `wxscan_host_forward(net, input, len, shape, rank) -> bytes` | Run network `net` (0 detector, 1 super resolution) over an NCHW f32 input, and return the byte size of the result it prepared, or 0 |
| `wxscan_host_fetch(dst, len) -> ok` | Write that result into the module's memory |

Two calls rather than one so that the module allocates and frees everything;
the host never holds memory the module has to release. The block written back
is little-endian 32-bit words: the number of outputs, then for each output its
rank followed by its dimensions, then all the f32 data one output after
another. [`host_net.rs`](../wxscan-ffi/src/host_net.rs) is the other half of
this contract, and the readable version of it.

Scanners for this backend come from `wxscan_scanner_new_host(has_detector,
has_sr)` rather than `wxscan_scanner_new`: the weights stay in the host, and
the module only needs to know which networks it may ask for.

**The imports are synchronous, and browser inference APIs are not.** LiteRT.js
returns a promise on every backend, so a host has to bridge that gap — JSPI,
an Asyncify pass over this module, or running the module in a worker that
blocks on `Atomics.wait` — and which of those to use is not settled here.

`--features debug-log` adds a third import, `wxscan_host_log`, and an exported
`wxscan_install_panic_hook`. Without it a Rust panic reaches the host as
`RuntimeError: unreachable` with the message dropped, since the module has
nowhere to print; with it the message arrives. Development only.
