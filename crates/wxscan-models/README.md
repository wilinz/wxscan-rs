# wxscan-models

Prebuilt weights for [`wxscan`](https://crates.io/crates/wxscan): an SSD
detector that locates candidate symbols, and a super resolution network that
upscales small crops before decoding.

Weights are grouped by file format, because the format a caller needs follows
the inference backend it runs. Each format sits behind the feature of the same
name, so a build embeds only what it uses:

```rust
let detect = wxscan_models::tflite::DETECT;
let sr = wxscan_models::tflite::SR;
```

`tflite` matches the default backend; `onnx` matches the pure Rust tract
backend. A new format is a new module and a new feature; the weights themselves
are the same, converted differently.

Both were converted from the Caffe models published at
[WeChatCV/opencv_3rdparty](https://github.com/WeChatCV/opencv_3rdparty), at the
revision `opencv_contrib/modules/wechat_qrcode/CMakeLists.txt` references,
without retraining or changing the weights. The conversion scripts are in
`tools/model_conversion` of the parent repository.

This is a separate crate so that callers supplying their own weights do not have
to download about a megabyte per format they will not use. `wxscan` pulls it in through its `bundled-models`
feature.

Part of [wxscan-rs](https://github.com/wilinz/wxscan-rs). Apache-2.0, as are the
upstream models; see `NOTICE`.
