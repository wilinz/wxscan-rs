//! Two holders, one scanner.
//!
//! The case these exist for is a Dart application holding a scanner for still
//! images while a camera plugin decodes frames with the same handle. Neither
//! side can see the other's lifetime, so whichever finishes first must leave
//! the scanner standing.
//!
//! A release that frees too early is a use-after-free rather than a failure, so
//! what these assert is that the scanner still works after it — visible here,
//! and caught outright under Miri or a sanitizer.

use wxscan_ffi::{
    wxscan_results_free, wxscan_scan_gray, wxscan_scanner_has_detector, wxscan_scanner_new,
    wxscan_scanner_release, wxscan_scanner_retain, wxscan_scanner_scale_factor,
    wxscan_scanner_set_scale_factor, WxScanScannerId,
};

/// A scanner with no weights: enough to exercise the counting, and it keeps the
/// test from needing model files.
unsafe fn plain_scanner() -> WxScanScannerId {
    let s = wxscan_scanner_new(std::ptr::null(), 0, std::ptr::null(), 0);
    assert_ne!(s, 0, "the no-model scanner should always build");
    s
}

/// Scans a blank image. The result is uninteresting; that it runs at all is the
/// assertion, since it resolves the handle and takes the scanner's lock.
unsafe fn touch(scanner: WxScanScannerId) {
    let gray = vec![0u8; 64 * 64];
    let out = wxscan_scan_gray(scanner, gray.as_ptr(), 64, 64);
    assert!(!out.is_null(), "a blank image still produces an empty result");
    wxscan_results_free(out);
}

#[test]
fn a_retained_scanner_survives_the_first_release() {
    unsafe {
        let scanner = plain_scanner();
        assert_eq!(wxscan_scanner_retain(scanner), scanner, "retain returns the same handle");

        // The application lets go while the camera is still decoding.
        wxscan_scanner_release(scanner);
        touch(scanner);

        // And the state the second holder set is its own, not a fresh scanner.
        wxscan_scanner_set_scale_factor(scanner, 0.5);
        assert!((wxscan_scanner_scale_factor(scanner) - 0.5).abs() < 1e-6);

        wxscan_scanner_release(scanner);
    }
}

#[test]
fn several_holders_each_take_their_own() {
    unsafe {
        let scanner = plain_scanner();
        for _ in 0..8 {
            wxscan_scanner_retain(scanner);
        }
        for _ in 0..8 {
            wxscan_scanner_release(scanner);
            touch(scanner);
        }
        wxscan_scanner_release(scanner);
    }
}

#[test]
fn holders_on_other_threads_count_too() {
    unsafe {
        let scanner = plain_scanner();
        let address = scanner;

        let borrowers: Vec<_> = (0..4)
            .map(|_| {
                wxscan_scanner_retain(scanner);
                std::thread::spawn(move || {
                    touch(address);
                    wxscan_scanner_release(address);
                })
            })
            .collect();

        // The creator drops out first, which is the ordering that would be a
        // use-after-free without the count.
        wxscan_scanner_release(scanner);
        for t in borrowers {
            t.join().unwrap();
        }
    }
}

#[test]
fn a_handle_naming_nothing_is_refused_rather_than_followed() {
    // The point of the table: these are ordinary failures, not reads of freed
    // or arbitrary memory. Zero is never a scanner, and a number nobody handed
    // out is not one either.
    assert_eq!(wxscan_scanner_retain(0), 0);
    wxscan_scanner_release(0);
    assert_eq!(wxscan_scanner_retain(usize::MAX), 0);
    wxscan_scanner_release(usize::MAX);
    assert_eq!(wxscan_scanner_has_detector(usize::MAX), 0);
}

#[test]
fn a_released_handle_is_dead_and_stays_dead() {
    unsafe {
        let scanner = plain_scanner();
        wxscan_scanner_release(scanner);

        // What a camera binding does with a stale handle after the application
        // it borrowed from has gone away. Nothing is dereferenced.
        assert_eq!(wxscan_scanner_retain(scanner), 0);
        assert_eq!(wxscan_scanner_has_detector(scanner), 0);
        let gray = vec![0u8; 64 * 64];
        assert!(wxscan_scan_gray(scanner, gray.as_ptr(), 64, 64).is_null());

        // And the number is never handed out again, so it cannot come to mean
        // a different scanner later.
        let next = plain_scanner();
        assert_ne!(next, scanner);
        wxscan_scanner_release(next);
    }
}

#[test]
fn a_scanner_reports_itself_after_being_handed_over() {
    unsafe {
        let scanner = plain_scanner();
        wxscan_scanner_retain(scanner);
        wxscan_scanner_release(scanner);
        // No weights were given, so there is no detector; the point is that the
        // question can still be asked of it.
        assert_eq!(wxscan_scanner_has_detector(scanner), 0);
        wxscan_scanner_release(scanner);
    }
}
