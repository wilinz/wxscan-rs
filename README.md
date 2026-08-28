# wxscan-rs

**English** · [简体中文](README.zh-CN.md)

The `wechat_qrcode` algorithm from OpenCV contrib, in Rust: a CNN locates the
symbols in a frame, a second network upscales each crop, and a fork of ZXing
decodes them. No OpenCV, no C++, and with the `tract` backend no C at all.

<img src="https://raw.githubusercontent.com/wilinz/wxscan/main/docs/demo.webp" width="300"
     alt="Two QR codes in one camera frame, each marked; tapping one opens its decoded
     text, a Chinese payload read as UTF-8.">

*This algorithm at work, driven from the Flutter packages in
[wxscan](https://github.com/wilinz/wxscan): two codes in one frame, one of
them turned, read across a desk from a laptop screen.*

Decoding a QR code is the easy part. Finding a small, distant or badly lit one
in a 1080p frame is what the two CNN stages are for, and why the WeChat scanner
reads a code from across a room while a plain decoder asks you to hold it up to
the lens.

**It is checked against the original, not just tested.** On a fixed corpus the
decoded text matches OpenCV's C++ implementation on 159 of 160 images without
models and 24 of 24 scene images with them, and the corner coordinates are
bit-identical on all but two, which differ at sub-pixel level. What that costs
in freedom is deliberate: the port follows the [upstream sources][upstream] line
by line, including the parts that look wrong, because a port that quietly
improves on the original cannot be compared to it. See [Parity](#parity).

**It builds anywhere cargo does.** The default backend needs the TFLite C
library; the `tract` backend needs nothing outside Rust, so cross-compiling is
`cargo build --target`. Both run the same weights.

[upstream]: https://github.com/opencv/opencv_contrib/tree/4.x/modules/wechat_qrcode

## Usage

```sh
cargo add wxscan
```

Or from git, which follows the default branch until a `tag`, `branch` or `rev`
pins it:

```toml
[dependencies]
wxscan = { git = "https://github.com/wilinz/wxscan-rs" }
```

**What you need**

| | Version |
|---|---|
| Rust | 1.75 or newer to depend on the crates; a checkout builds with the 1.95.0 that `rust-toolchain.toml` pins, which rustup installs on first use |
| libtensorflowlite_c | only for the default `tflite` backend, which links it — see [TFLite library](#the-tflite-library). The `tract` backend needs nothing outside Rust |

Nothing is vendored and no build script reaches the network.

The git route has one more requirement. `cvlite` and `wxing` are named as
ordinary dependencies, so a build that takes this crate from git — rather than
from crates.io, where those two sit beside it — has to say where they are:

```toml
[patch.crates-io]
cvlite = { git = "https://github.com/wilinz/cvlite" }
wxing = { git = "https://github.com/wilinz/wxing" }
```

The weights are in no crate. Download `detect.tflite` and `sr.tflite` from
[wxscan-weights](https://github.com/wilinz/wxscan-weights) — or bring your own —
and read them as bytes:

```rust
use wxscan::WeChatQRCode;

let detect_bytes = std::fs::read("detect.tflite")?;
let sr_bytes = std::fs::read("sr.tflite")?;

// Both models may be None, which decodes without the CNN stages.
let detect = wxscan::tflite::TfliteNet::from_bytes(&detect_bytes)?;
let sr = wxscan::tflite::TfliteNet::from_bytes(&sr_bytes)?;
let scanner = WeChatQRCode::new(Some(detect), Some(sr));

for result in scanner.detect_and_decode_gray(&gray, width, height) {
    // The payload is raw bytes; `charset` says how to interpret it.
    println!("{} ({})", result.text_lossy(), result.charset);
}
```

`gray` is 8-bit grayscale, one byte per pixel, row after row.

## Crates

The split follows one boundary: whether a piece is specific to this algorithm.
The two pieces that are not live in their own repositories, because they are
useful without any of this:

| Crate | Repository | Contents |
|---|---|---|
| [`cvlite`](https://github.com/wilinz/cvlite) | own | The OpenCV `imgproc` functions used here: resize, adaptive threshold, colour conversion, blob. Not specific to QR codes; NEON paths on aarch64. |
| [`wxing`](https://github.com/wilinz/wxing) | own | The ZXing fork used by WeChat: binarizers, finder patterns, decoder. Independent of the CNN stages, so it decodes on its own. |

Everything below is specific to the WeChat algorithm, and the three move as one
version: the tflite binding is the backend the detector runs on, and the C ABI
is its surface.

| Crate | Contents |
|---|---|
| [`wxscan`](crates/wxscan) | CNN detection, super resolution, and the orchestration around them. This is the complete algorithm. |
| [`wxscan-tflite`](crates/wxscan-tflite) | The tflite binding, used as the default inference backend. Separate so the only C dependency is confined to it. |
| [`wxscan-ffi`](crates/wxscan-ffi) | The C ABI, for callers outside Rust. Generates `include/wxscan.h` with cbindgen. |

## Inference backends

CNN inference sits behind the `net::Net` trait, and no part of the algorithm
knows which library runs it. Two backends ship, both in `wxscan::backend`:

| Feature | Engine | Weights | C dependency |
|---|---|---|---|
| `tflite` (default) | `wxscan-tflite` | `detect.tflite`, `sr.tflite` | libtensorflowlite_c |
| `tract` | [tract](https://crates.io/crates/tract-onnx) | `detect.onnx`, `sr.onnx` | none |

The tflite adapter is also where layout conversion lives, since tflite is NHWC
while the trait contract is NCHW. tract needs none: ONNX is NCHW, like the Caffe
models both formats are converted from.

```sh
cargo add wxscan --no-default-features --features tract
```

Or, from git:

```toml
wxscan = { git = "https://github.com/wilinz/wxscan-rs", default-features = false, features = ["tract"] }
```

tract builds and runs anywhere cargo does, at some cost in speed against
tflite's XNNPACK kernels. Turning both features off leaves a core with no
inference at all, which still compiles and tests with plain `cargo build`.

One thing is needed on `wasm32-unknown-unknown` whatever the features: the host
must supply a `wxscan_host_now_us() -> f64` import in the module `wxscan`.
`std::time::Instant::now()` panics on that target, so the stage timers read the
host's clock instead — a browser answers `performance.now() * 1000`. A host with
no clock to lend can return a constant, and every stage then reports zero.

A backend means implementing one method. The trait lives in `wxscan`, so an
out-of-tree crate can implement it for its own type:

```rust
use wxscan::net::{Net, NetOutput};

struct MyNet(/* a CoreML, NNAPI or any other engine */);

impl Net for MyNet {
    fn forward(&self, input: &[f32], shape: &[usize]) -> Result<Vec<NetOutput>, String> {
        // input is NCHW; return NCHW
    }
}
```

Without models the pipeline degrades to a plain decoder rather than failing. It
still reads ordinary codes; what it loses is the detection rate on small or
distant ones, which is what the CNN stages contribute.

## The TFLite library

This repository vendors no binaries. Point `TFLITE_LIB_DIR` at a directory
containing libtensorflowlite_c, or let the final link step resolve the symbols,
which is what Apple platforms normally do. The library name differs by
distribution: desktop builds of the C API are `libtensorflowlite_c`, while
Google's LiteRT distribution for Android names the same API `libLiteRt`.

```sh
TFLITE_LIB_DIR=/path/to/libs cargo test --workspace
```

## Parity

`tools/parity` runs the same images through OpenCV's `wechat_qrcode` and through
this port and compares the decoded text and the corner coordinates. Current
results are in [`tools/parity/README.md`](tools/parity/README.md): text matches
on 159/160 images without models and 24/24 scene images with them, and corner
coordinates are bit-identical on all but two, which differ at sub-pixel level.

The remaining differences trace to `cv::adaptiveThreshold`, where OpenCV uses a
fixed-point separable filter for 8U images while this port accumulates in f32.

The [wxscan-weights](https://github.com/wilinz/wxscan-weights) repository holds
the prebuilt weights and the scripts that rebuild them from the published Caffe
models.

## Performance

[`docs/performance.md`](docs/performance.md) records what was optimized, how it
was measured, and what was tried and reverted. Every change there was verified
to leave the output byte-identical on the parity corpus.

## Bindings

The C ABI is consumed by the Flutter packages in
[`wxscan`](https://github.com/wilinz/wxscan), which is the reference for how to
drive it from a platform binding.

## Licence

Apache-2.0, as is the upstream implementation this is ported from.
