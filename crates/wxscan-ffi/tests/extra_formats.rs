//! webp, bmp and tiff: the formats that arrive from somewhere other than a
//! camera — webp off the web, bmp out of a Windows screenshot.
//!
//! Each is its own feature and each costs real size, so what this file
//! asserts is that the feature is what decides: on, the decoder is built in;
//! off, the bytes are not this library's to read and the answer is
//! `UnsupportedFormat` rather than a picture decoded by something that was
//! meant to be gone. A caller with a platform decoder to lend — Apple's
//! ImageIO reads all three — leaves them off and loses nothing.
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

/// Each of the three, on when its feature is and off when it is not. Written
/// as one test per format rather than a loop over three, so that a build
/// carrying two of them still says which.
macro_rules! format_follows_its_feature {
    ($name:ident, $feature:literal, $file:literal) => {
        #[test]
        fn $name() {
            if cfg!(feature = $feature) {
                assert!(
                    built_in_reads($file),
                    concat!($file, " should be built in with the ", $feature, " feature")
                );
            } else {
                assert!(
                    !built_in_reads($file),
                    concat!($file, " is built in without the ", $feature, " feature")
                );
            }
        }
    };
}

format_follows_its_feature!(webp_follows_its_feature, "webp", "upright.webp");
format_follows_its_feature!(bmp_follows_its_feature, "bmp", "upright.bmp");
format_follows_its_feature!(tiff_follows_its_feature, "tiff", "upright.tiff");

/// png is in `default` and is what every other test here reads, so a build
/// that has lost it should say so once, plainly, rather than as five failures
/// about something else.
#[cfg(feature = "png")]
#[test]
fn png_is_built_in() {
    assert!(built_in_reads("upright.png"));
}
