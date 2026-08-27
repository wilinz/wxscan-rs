//! Prebuilt weights for the [`wxscan`](https://crates.io/crates/wxscan)
//! pipeline: an SSD detector that locates candidate symbols, and a super
//! resolution network that upscales small crops before decoding.
//!
//! Weights are grouped by file format, because the format a caller needs
//! follows the inference backend it runs. Each format sits behind the feature
//! of the same name, so a build embeds only what it uses:
//!
//! ```no_run
//! let detect = wxscan_models::tflite::DETECT;
//! let sr = wxscan_models::tflite::SR;
//! # let _ = (detect, sr);
//! ```
//!
//! `tflite` matches the default backend; `onnx` matches the pure Rust tract
//! backend. A new format is a new module and a new feature; the weights
//! themselves are the same, converted differently.
//!
//! Both formats derive from the Caffe models published at
//! <https://github.com/WeChatCV/opencv_3rdparty> (commit
//! `a8b69ccc738421293254aec5ddb38bd523503252`, the revision referenced by
//! `opencv_contrib/modules/wechat_qrcode/CMakeLists.txt`), converted by the
//! scripts in `tools/model_conversion` without retraining.
//!
//! This crate is separate from `wxscan` so that callers supplying their own
//! weights, or running a different backend, do not download data they will not
//! use.
//!
//! Licensed under Apache-2.0, as are the upstream models. See `NOTICE`.

/// Weights in TFLite format, for the backend `wxscan` enables by default.
#[cfg(feature = "tflite")]
pub mod tflite {
    /// SSD detector weights.
    pub const DETECT: &[u8] = include_bytes!("../models/detect.tflite");

    /// Super resolution weights.
    pub const SR: &[u8] = include_bytes!("../models/sr.tflite");
}

/// Weights in ONNX format, for the pure Rust tract backend.
#[cfg(feature = "onnx")]
pub mod onnx {
    /// SSD detector weights.
    pub const DETECT: &[u8] = include_bytes!("../models/detect.onnx");

    /// Super resolution weights.
    pub const SR: &[u8] = include_bytes!("../models/sr.onnx");
}
