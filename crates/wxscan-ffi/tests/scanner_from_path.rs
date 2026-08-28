//! `wxscan_scanner_new_path`: the three ways a path can be wrong.
//!
//! A caller with a path has made one of three mistakes when nothing comes
//! back, and they are fixed in different places: a typo in the string, a
//! download that has not happened, or a file that is not a model at all. The
//! status is the only thing that tells them apart, so it is what this checks.

use std::ffi::CString;

use wxscan_ffi::{
    wxscan_scanner_new_path, wxscan_scanner_release, WxScanScannerId, WxScanStatus,
};

/// A scanner built from paths, with the status it reported.
unsafe fn from_paths(detect: Option<&str>, sr: Option<&str>) -> (WxScanScannerId, WxScanStatus) {
    let d = detect.map(|p| CString::new(p).unwrap());
    let s = sr.map(|p| CString::new(p).unwrap());
    let mut status = WxScanStatus::Ok;
    let id = wxscan_scanner_new_path(
        d.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
        s.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
        &mut status,
    );
    (id, status)
}

#[test]
fn no_paths_at_all_is_the_mode_without_models() {
    unsafe {
        let (id, status) = from_paths(None, None);
        assert_ne!(id, 0, "a scanner with no models is still a scanner");
        assert_eq!(status, WxScanStatus::Ok);
        wxscan_scanner_release(id);
    }
}

#[test]
fn a_file_that_is_not_there_is_unreadable() {
    unsafe {
        let (id, status) = from_paths(Some("/nowhere/detect.tflite"), None);
        assert_eq!(id, 0);
        assert_eq!(
            status,
            WxScanStatus::Unreadable,
            "a missing file is the caller's to fix, and must not look like weights that failed"
        );
    }
}

#[test]
fn a_file_that_is_not_weights_is_refused() {
    unsafe {
        let dir = std::env::temp_dir().join("wxscan-not-a-model.bin");
        std::fs::write(&dir, b"this is not a tflite model").unwrap();
        let (id, status) = from_paths(Some(dir.to_str().unwrap()), None);
        assert_eq!(id, 0);
        assert_eq!(
            status,
            WxScanStatus::WeightsRefused,
            "the file was read; it is its contents that are wrong"
        );
        std::fs::remove_file(&dir).ok();
    }
}

#[test]
fn the_second_path_is_checked_too() {
    unsafe {
        let (id, status) = from_paths(None, Some("/nowhere/sr.tflite"));
        assert_eq!(id, 0);
        assert_eq!(status, WxScanStatus::Unreadable);
    }
}
