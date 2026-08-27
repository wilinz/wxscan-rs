//! End to end: read PNG, convert to grayscale, recognize. Corresponds to the
//! no-model path of the upstream `test/test_qrcode.cpp`.

use wxscan::net::NoNet;
use wxscan::WeChatQRCode;

fn load_gray(path: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(path).expect("open image").to_luma8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w as usize, h as usize)
}

fn decode_file(path: &str) -> Vec<String> {
    let (gray, w, h) = load_gray(path);
    let scanner: WeChatQRCode<NoNet> = WeChatQRCode::new(None, None);
    scanner
        .detect_and_decode_gray(&gray, w, h)
        .into_iter()
        .map(|r| r.text_lossy())
        .collect()
}

#[test]
fn decode_hello() {
    assert_eq!(decode_file("tests/data/hello.png"), vec!["HELLO WORLD".to_string()]);
}

#[test]
fn decode_url() {
    assert_eq!(
        decode_file("tests/data/url.png"),
        vec!["https://github.com/opencv/opencv_contrib".to_string()]
    );
}

#[test]
fn decode_numeric() {
    assert_eq!(
        decode_file("tests/data/numeric.png"),
        vec!["1234567890123456789012345678901234567890".to_string()]
    );
}

#[test]
fn decode_long() {
    assert_eq!(decode_file("tests/data/long.png"), vec!["A".repeat(300)]);
}

const URL: &str = "https://github.com/opencv/opencv_contrib";

#[test]
fn decode_rotated_15deg() {
    assert_eq!(decode_file("tests/data/rot15.png"), vec![URL.to_string()]);
}

#[test]
fn decode_rotated_90deg() {
    assert_eq!(decode_file("tests/data/rot90.png"), vec![URL.to_string()]);
}

#[test]
fn decode_downscaled() {
    assert_eq!(decode_file("tests/data/small.png"), vec![URL.to_string()]);
}

#[test]
fn decode_noisy() {
    assert_eq!(decode_file("tests/data/noise.png"), vec![URL.to_string()]);
}

#[test]
fn decode_blurred() {
    assert_eq!(decode_file("tests/data/blur.png"), vec![URL.to_string()]);
}

#[test]
fn decode_perspective() {
    assert_eq!(decode_file("tests/data/persp.png"), vec![URL.to_string()]);
}

/// Inverted image: after the normal image fails, `QRCodeReader` retries once
/// with the inverted matrix.
#[test]
fn decode_inverted() {
    assert_eq!(decode_file("tests/data/inverted.png"), vec![URL.to_string()]);
}

/// segno encodes Chinese text as a BYTE segment of UTF-8 bytes, the charset
/// is reported as UTF-8, and the bytes are returned unchanged.
#[test]
fn decode_chinese_byte_mode() {
    let (gray, w, h) = load_gray("tests/data/chinese.png");
    let scanner: WeChatQRCode<NoNet> = WeChatQRCode::new(None, None);
    let results = scanner.detect_and_decode_gray(&gray, w, h);
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert_eq!(r.charset, "UTF-8");
    assert_eq!(r.charset_mode, "BYTE");
    assert_eq!(
        r.bytes,
        "微信二维码扫描器测试".as_bytes()
    );
}
