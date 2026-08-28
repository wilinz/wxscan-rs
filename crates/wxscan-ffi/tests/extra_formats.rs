//! webp, bmp and tiff: the formats carried everywhere the platform has no
//! decoder to lend.
//!
//! Nobody photographs anything in these, but they arrive from elsewhere — webp
//! off the web, bmp out of a Windows screenshot — and Windows, Linux and
//! Android have nothing else to fall back on.
//!
//! Apple deliberately does not carry them: ImageIO reads all three already, and
//! a second copy would be 570 KB for nothing. So this file asserts the opposite
//! thing on Apple platforms — that they are *not* built in — which is what
//! keeps the target dependency from being quietly turned into a plain one.
#![cfg(feature = "image-io")]

use wxscan_ffi::{
    wxscan_results_free, wxscan_scan_bytes, wxscan_scanner_release, wxscan_scanner_new,
    WxScanScannerId, WxScanStatus,
};

fn bytes(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

unsafe fn plain_scanner() -> WxScanScannerId {
    let s = wxscan_scanner_new(std::ptr::null(), 0, std::ptr::null(), 0);
    assert_ne!(s, 0);
    s
}

/// Scans one file and reports whether the built-in decoders took it.
fn built_in_reads(name: &str) -> bool {
    unsafe {
        let scanner = plain_scanner();
        let data = bytes(name);
        let mut status = WxScanStatus::BadArgument;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        let ok = status == WxScanStatus::Ok;
        if ok {
            assert!(!out.is_null());
            // The same picture as the png fixture, so the same symbol and the
            // same size: a decoder that produced something else would pass a
            // mere "it decoded" check.
            assert_eq!(((*out).width, (*out).height), (320, 460), "{name}");
            assert_eq!((*out).results_len, 1, "{name}");
            wxscan_results_free(out);
        }
        wxscan_scanner_release(scanner);
        ok
    }
}

#[cfg(not(any(target_os = "ios", target_os = "macos")))]
#[test]
fn webp_bmp_and_tiff_are_carried_where_there_is_no_platform_decoder() {
    for name in ["upright.webp", "upright.bmp", "upright.tiff"] {
        assert!(built_in_reads(name), "{name} should be built in here");
    }
}

/// On Apple these are ImageIO's job, and carrying them too would be 570 KB
/// spent twice. Nothing breaks if that changes — the picture still decodes,
/// through the platform — so only a test notices.
#[cfg(any(target_os = "ios", target_os = "macos"))]
#[test]
fn webp_bmp_and_tiff_are_left_to_the_platform_on_apple() {
    for name in ["upright.webp", "upright.bmp", "upright.tiff"] {
        assert!(
            !built_in_reads(name),
            "{name} is built in on Apple, where ImageIO already reads it"
        );
    }
}

/// png, jpeg and gif are the floor and are carried everywhere, platform
/// decoder or not.
#[test]
fn the_three_a_camera_writes_are_everywhere() {
    assert!(built_in_reads("upright.png"));
}
