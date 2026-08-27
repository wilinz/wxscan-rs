//! `wxscan_scan_path`: reading the file itself rather than being handed pixels.
#![cfg(feature = "image-io")]

use std::ffi::CString;

use wxscan_ffi::{
    wxscan_results_free, wxscan_scan_path, wxscan_scanner_free, wxscan_scanner_new,
    WxScanScanner, WxScanStatus,
};

fn data(name: &str) -> CString {
    CString::new(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// A scanner with no weights: enough to decode an ordinary symbol, and it keeps
/// the test from needing model files.
unsafe fn plain_scanner() -> *mut WxScanScanner {
    let s = wxscan_scanner_new(std::ptr::null(), 0, std::ptr::null(), 0);
    assert!(!s.is_null(), "the no-model scanner should always build");
    s
}

#[test]
fn reads_a_file_and_decodes_it() {
    unsafe {
        let scanner = plain_scanner();
        let mut status = WxScanStatus::BadArgument;
        let out = wxscan_scan_path(scanner, data("upright.png").as_ptr(), &mut status);
        assert_eq!(status, WxScanStatus::Ok);
        assert!(!out.is_null());
        assert_eq!((*out).width, 320);
        assert_eq!((*out).height, 460);
        assert_eq!((*out).results_len, 1);
        wxscan_results_free(out);
        wxscan_scanner_free(scanner);
    }
}

#[test]
fn applies_the_orientation_the_file_records() {
    unsafe {
        let scanner = plain_scanner();
        let mut status = WxScanStatus::BadArgument;
        let out = wxscan_scan_path(scanner, data("exif_rot90.jpg").as_ptr(), &mut status);
        assert_eq!(status, WxScanStatus::Ok);
        assert!(!out.is_null());
        // Stored 460x320 with the tag that says to turn it upright, which is
        // how a phone writes a picture taken sideways. Ignoring the tag would
        // give the stored dimensions back and coordinates that do not match
        // what anyone saw on screen.
        assert_eq!(((*out).width, (*out).height), (320, 460));
        assert_eq!((*out).results_len, 1);
        wxscan_results_free(out);
        wxscan_scanner_free(scanner);
    }
}

#[test]
fn a_missing_file_is_not_a_picture_without_a_code_in_it() {
    unsafe {
        let scanner = plain_scanner();
        let mut status = WxScanStatus::Ok;
        let out = wxscan_scan_path(scanner, data("nothing_here.png").as_ptr(), &mut status);
        assert!(out.is_null());
        assert_eq!(status, WxScanStatus::Unreadable);
        wxscan_scanner_free(scanner);
    }
}

#[test]
fn a_file_that_is_not_an_image_says_so() {
    unsafe {
        let scanner = plain_scanner();
        let mut status = WxScanStatus::Ok;
        let out = wxscan_scan_path(scanner, data("not_an_image.txt").as_ptr(), &mut status);
        assert!(out.is_null());
        assert_eq!(status, WxScanStatus::UnsupportedFormat);
        wxscan_scanner_free(scanner);
    }
}

#[test]
fn a_null_path_is_rejected_rather_than_dereferenced() {
    unsafe {
        let scanner = plain_scanner();
        let mut status = WxScanStatus::Ok;
        assert!(wxscan_scan_path(scanner, std::ptr::null(), &mut status).is_null());
        assert_eq!(status, WxScanStatus::BadArgument);
        // A null status pointer is allowed: a caller may not care why.
        assert!(wxscan_scan_path(scanner, std::ptr::null(), std::ptr::null_mut()).is_null());
        wxscan_scanner_free(scanner);
    }
}
