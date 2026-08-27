//! End to end through the pure Rust backend: the CNN stages actually run, so
//! this covers the detector and super resolution paths that `decode.rs` skips.
//!
//! The expected strings are what the tflite backend produces on the same
//! images, the two having been compared side by side; the point of the test is
//! that swapping the engine does not change the answer.
//!
//! The ONNX weights are not part of any crate; they come from the
//! wxscan-weights repository checked out beside this one, or from
//! `WXSCAN_WEIGHTS_DIR`. Without them the test skips.
//!
//! Run with `cargo test -p wxscan --no-default-features --features tract
//! --test tract`.
#![cfg(feature = "tract")]

mod common;

use wxscan::backend::tract::TractNet;
use wxscan::WeChatQRCode;

const URL: &str = "https://github.com/opencv/opencv_contrib";

fn scanner() -> Option<WeChatQRCode<TractNet>> {
    let detect = common::load("detect.onnx")?;
    let sr = common::load("sr.onnx")?;
    Some(WeChatQRCode::new(
        Some(TractNet::from_bytes(&detect).expect("load detect.onnx")),
        Some(TractNet::from_bytes(&sr).expect("load sr.onnx")),
    ))
}

fn decode(scanner: &WeChatQRCode<TractNet>, file: &str) -> Vec<String> {
    let path = format!("tests/data/{file}");
    let img = image::open(&path).expect("open image").to_luma8();
    let (w, h) = img.dimensions();
    scanner
        .detect_and_decode_gray(&img.into_raw(), w as usize, h as usize)
        .into_iter()
        .map(|r| r.text_lossy())
        .collect()
}

#[test]
fn decodes_the_sample_images() {
    let Some(s) = scanner() else {
        return common::skip("decodes_the_sample_images");
    };
    for (file, want) in [
        ("hello.png", "HELLO WORLD"),
        ("url.png", URL),
        ("rot15.png", URL),
        ("small.png", URL),
        ("numeric.png", "1234567890123456789012345678901234567890"),
        ("chinese.png", "微信二维码扫描器测试"),
    ] {
        assert_eq!(decode(&s, file), vec![want.to_string()], "{file}");
    }
    assert_eq!(decode(&s, "long.png"), vec!["A".repeat(300)]);
}

/// Super resolution runs on crops of whatever size the pipeline produces, so
/// the plan cache is asked for a shape it has not seen on almost every image.
/// Reusing one scanner across differently sized inputs is the case that has to
/// keep working.
#[test]
fn one_scanner_handles_many_input_sizes() {
    let Some(s) = scanner() else {
        return common::skip("one_scanner_handles_many_input_sizes");
    };
    for file in ["hello.png", "small.png", "long.png", "url.png", "hello.png"] {
        assert!(!decode(&s, file).is_empty(), "{file}: no result");
    }
}
