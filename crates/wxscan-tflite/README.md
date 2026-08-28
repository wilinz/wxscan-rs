# wxscan-tflite

A minimal FFI binding to libtensorflowlite_c, the LiteRT C API. It is the
default inference backend of [`wxscan`](../wxscan), and is
usable on its own as a small tflite binding.

```rust
let net = wxscan_tflite::TfliteNet::from_bytes(model_bytes)?;
// dims are in the model's own layout, NHWC for these models
let outputs = net.run(&input, &[1, height, width, 1])?;
```

The interpreter is cached by input shape and rebuilt when the shape changes,
because XNNPACK builds a static graph and resizing after the delegate has taken
effect fails.

This crate speaks tflite's conventions: tensor layouts are NHWC, and shapes are
whatever the model declares. Nothing converts them, which is deliberate —
adapting to an inference abstraction belongs in the layer that defines one.
`wxscan` does that behind its `net::Net` trait, so no part of the algorithm
depends on this crate.

## The TFLite library

No binaries are vendored. Point `TFLITE_LIB_DIR` at a directory containing the
library, or let the final link step resolve the symbols, which is what Apple
platforms normally do. The name differs by distribution: desktop builds of the C
API are `libtensorflowlite_c`, while Google's LiteRT distribution for Android
names the same API `libLiteRt`.

Part of [wxscan-rs](https://github.com/wilinz/wxscan-rs). Apache-2.0.
