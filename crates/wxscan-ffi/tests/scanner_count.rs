//! `wxscan_scanner_count`, alone in this binary.
//!
//! The count is process-wide, so a test that asserts an exact number cannot
//! share a process with anything else that creates a scanner — and cargo runs
//! the tests in one file on parallel threads. One test per binary is what keeps
//! this honest.

use wxscan_ffi::{
    wxscan_scanner_count, wxscan_scanner_new, wxscan_scanner_release, wxscan_scanner_retain,
};

#[test]
fn the_count_follows_the_holders() {
    unsafe {
        assert_eq!(wxscan_scanner_count(), 0, "nothing has been created yet");

        let scanner = wxscan_scanner_new(std::ptr::null(), 0, std::ptr::null(), 0);
        assert_ne!(scanner, 0);
        assert_eq!(wxscan_scanner_count(), 1);

        // A second holder is not a second scanner.
        wxscan_scanner_retain(scanner);
        assert_eq!(wxscan_scanner_count(), 1);

        wxscan_scanner_release(scanner);
        assert_eq!(wxscan_scanner_count(), 1, "one holder is left");

        wxscan_scanner_release(scanner);
        assert_eq!(wxscan_scanner_count(), 0, "and now none");

        // A leak is exactly what this is for: created and never given back.
        let leaked = wxscan_scanner_new(std::ptr::null(), 0, std::ptr::null(), 0);
        assert_ne!(leaked, 0);
        assert_eq!(wxscan_scanner_count(), 1, "a holder that never let go shows up");
        wxscan_scanner_release(leaked);
    }
}
