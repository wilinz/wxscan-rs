//! `wxscan_scan_bytes`: the same pictures, handed over as data rather than as
//! a path.
//!
//! The point of most of these is that they agree with `wxscan_scan_path`, so
//! they read the same files and compare against the same expectations. What
//! differs is what cannot happen here: there is nothing to open, so nothing is
//! ever `Unreadable`.
#![cfg(all(feature = "png", feature = "jpeg"))]

use wxscan_ffi::{
    wxscan_results_free, wxscan_scan_bytes, wxscan_scan_path, wxscan_scanner_release,
    wxscan_scanner_new, WxScanScannerId, WxScanStatus,
};

fn bytes(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// A scanner with no weights: enough to decode an ordinary symbol, and it keeps
/// the test from needing model files.
unsafe fn plain_scanner() -> WxScanScannerId {
    let s = wxscan_scanner_new(std::ptr::null(), 0, std::ptr::null(), 0);
    assert_ne!(s, 0, "the no-model scanner should always build");
    s
}

#[test]
fn decodes_a_picture_held_in_memory() {
    unsafe {
        let scanner = plain_scanner();
        let data = bytes("upright.png");
        let mut status = WxScanStatus::BadArgument;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        assert_eq!(status, WxScanStatus::Ok);
        assert!(!out.is_null());
        assert_eq!(((*out).width, (*out).height), (320, 460));
        assert_eq!((*out).results_len, 1);
        wxscan_results_free(out);
        wxscan_scanner_release(scanner);
    }
}

#[test]
fn applies_the_orientation_the_file_records() {
    unsafe {
        let scanner = plain_scanner();
        let data = bytes("exif_rot90.jpg");
        let mut status = WxScanStatus::BadArgument;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        assert_eq!(status, WxScanStatus::Ok);
        assert!(!out.is_null());
        // Stored 460x320 with the tag that says to turn it upright. Reaching
        // the bytes a different way must not change this.
        assert_eq!(((*out).width, (*out).height), (320, 460));
        assert_eq!((*out).results_len, 1);
        wxscan_results_free(out);
        wxscan_scanner_release(scanner);
    }
}

/// The two entry points are one decoder with two front doors, and this is what
/// says so: same file, same payload and same corners either way.
#[test]
fn agrees_with_the_path_it_could_have_been_read_from() {
    unsafe {
        let scanner = plain_scanner();
        let path = std::ffi::CString::new(format!(
            "{}/tests/data/exif_rot90.jpg",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let data = bytes("exif_rot90.jpg");

        let mut from_path_status = WxScanStatus::BadArgument;
        let from_path = wxscan_scan_path(scanner, path.as_ptr(), &mut from_path_status);
        let mut from_bytes_status = WxScanStatus::BadArgument;
        let from_bytes =
            wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut from_bytes_status);

        assert_eq!(from_path_status, from_bytes_status);
        assert!(!from_path.is_null() && !from_bytes.is_null());
        assert_eq!(
            ((*from_path).width, (*from_path).height),
            ((*from_bytes).width, (*from_bytes).height)
        );
        assert_eq!((*from_path).results_len, (*from_bytes).results_len);

        for i in 0..(*from_path).results_len {
            let a = &*(*from_path).results.add(i);
            let b = &*(*from_bytes).results.add(i);
            assert_eq!(
                std::ffi::CStr::from_ptr(a.text),
                std::ffi::CStr::from_ptr(b.text)
            );
            assert_eq!(a.points, b.points);
        }

        wxscan_results_free(from_path);
        wxscan_results_free(from_bytes);
        wxscan_scanner_release(scanner);
    }
}

#[test]
fn bytes_that_are_not_a_picture_say_so() {
    unsafe {
        let scanner = plain_scanner();
        let data = bytes("not_an_image.txt");
        let mut status = WxScanStatus::Ok;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        assert!(out.is_null());
        // Never Unreadable: there was nothing to open, only bytes to make
        // sense of, and they did not.
        assert_eq!(status, WxScanStatus::UnsupportedFormat);
        wxscan_scanner_release(scanner);
    }
}

#[test]
fn an_empty_buffer_is_a_format_question_rather_than_a_crash() {
    unsafe {
        let scanner = plain_scanner();
        let mut status = WxScanStatus::Ok;
        let out = wxscan_scan_bytes(scanner, [].as_ptr(), 0, &mut status);
        assert!(out.is_null());
        assert_eq!(status, WxScanStatus::UnsupportedFormat);
        wxscan_scanner_release(scanner);
    }
}

#[test]
fn a_null_buffer_is_rejected_rather_than_dereferenced() {
    unsafe {
        let scanner = plain_scanner();
        let mut status = WxScanStatus::Ok;
        assert!(wxscan_scan_bytes(scanner, std::ptr::null(), 0, &mut status).is_null());
        assert_eq!(status, WxScanStatus::BadArgument);
        // A null status pointer is allowed: a caller may not care why.
        assert!(
            wxscan_scan_bytes(scanner, std::ptr::null(), 0, std::ptr::null_mut()).is_null()
        );
        wxscan_scanner_release(scanner);
    }
}

#[test]
fn a_null_scanner_is_rejected() {
    unsafe {
        let data = bytes("upright.png");
        let mut status = WxScanStatus::Ok;
        assert!(
            wxscan_scan_bytes(0, data.as_ptr(), data.len(), &mut status)
                .is_null()
        );
        assert_eq!(status, WxScanStatus::BadArgument);
    }
}
