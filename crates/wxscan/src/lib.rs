//! Rust port of the wechat_qrcode algorithm from OpenCV contrib: CNN detector,
//! super resolution upscaling and decoding.
//!
//! The module layout follows upstream `opencv_contrib/modules/wechat_qrcode/src`:
//!   - [`detector`] ← `src/detector/**` (SSD detection and affine alignment)
//!   - [`scale`]    ← `src/scale/**` (super resolution upscaling)
//!   - [`wechat_qrcode`] ← `src/wechat_qrcode.cpp` + `decodermgr` + `binarizermgr`
//!
//! Two further parts live in separate crates, since neither is specific to this
//! algorithm:
//!   - [`cvlite`] — the OpenCV functions used here
//!   - [`wxing`]  — the ZXing fork used by WeChat
//!
//! CNN inference sits behind the [`net::Net`] trait, and nothing in the
//! algorithm knows which library runs it. Two backends ship, both in
//! [`backend`]:
//!   - `tflite`, on by default, pulling in [`wxscan_tflite`]
//!   - `tract`, pure Rust, running the ONNX weights with no C dependency at all
//!
//! Without models the pipeline degrades to a plain decoder, since the CNN
//! stages are what this algorithm adds, which is why a backend is on by
//! default.
//!
//! For CoreML, NNAPI or anything else, set `default-features = false` and
//! implement [`net::Net`] for your own type; the trait is the whole contract.
//!
//! The weights themselves are not part of this crate. Enable the
//! `bundled-models` feature to get them through [`models`], where they are
//! grouped by format, or pass your own buffers to the constructor.

pub mod backend;
pub mod decodermgr;
pub mod frame;
pub mod detector;
pub mod net;
pub mod scale;
pub mod wechat_qrcode;

/// The tflite inference backend, available with the default `tflite` feature.
///
/// This is a re-export of [`wxscan_tflite`]; the implementation of [`net::Net`]
/// for its type lives in [`backend`].
#[cfg(feature = "tflite")]
pub use wxscan_tflite as tflite;

/// Prebuilt weights, available with the `bundled-models` feature.
///
/// Weights are grouped by format, which follows the backend in use: with the
/// default `tflite` feature they are at `models::tflite::{DETECT, SR}`, and
/// with `tract` at `models::onnx::{DETECT, SR}`.
#[cfg(feature = "bundled-models")]
pub use wxscan_models as models;

pub use wxing::error::{ZXError, ZXResult};
pub use wechat_qrcode::{QRCodeResult, WeChatQRCode};
