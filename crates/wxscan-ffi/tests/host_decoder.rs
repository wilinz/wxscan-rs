//! A decoder lent by the host, for formats this library does not carry.
//!
//! The real ones are CGImageSource and ImageDecoder; the one here is a stand-in
//! that recognises a made-up format, which is enough to say the three things
//! that matter: it is asked, it is asked *second*, and its buffer is given
//! back.
#![cfg(feature = "image-io")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use wxscan_ffi::{
    wxscan_results_free, wxscan_scan_bytes, wxscan_scanner_release, wxscan_scanner_new,
    wxscan_set_image_decoder, WxScanImageDecoder, WxScanScannerId, WxScanStatus,
};

/// A 2x2 grey square, in the format the fake host decoder hands back.
static PIXELS: [u8; 4] = [0xff, 0xff, 0xff, 0xff];

static DECODE_CALLS: AtomicUsize = AtomicUsize::new(0);
static RELEASE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Recognises anything starting with "FAKE" and nothing else, which is how a
/// real one behaves too: it answers 0 for what it does not know.
unsafe extern "C" fn fake_decode(
    data: *const u8,
    len: usize,
    out_pixels: *mut *const u8,
    out_width: *mut u32,
    out_height: *mut u32,
    out_format: *mut i32,
    ctx: *mut std::ffi::c_void,
) -> i32 {
    DECODE_CALLS.fetch_add(1, Ordering::SeqCst);
    assert_eq!(ctx as usize, 0x1234, "the context comes back untouched");
    let bytes = std::slice::from_raw_parts(data, len);
    if !bytes.starts_with(b"FAKE") {
        return 0;
    }
    *out_pixels = PIXELS.as_ptr();
    *out_width = 2;
    *out_height = 2;
    *out_format = 0; // Gray
    1
}

unsafe extern "C" fn fake_release(pixels: *const u8, _ctx: *mut std::ffi::c_void) {
    RELEASE_CALLS.fetch_add(1, Ordering::SeqCst);
    assert_eq!(pixels, PIXELS.as_ptr(), "released what decode handed over");
}

/// The decoder is one global slot, and `cargo test` runs these in parallel by
/// default, so they would install over each other and count each other's
/// calls. Every test holds this for its duration instead.
static SERIAL: Mutex<()> = Mutex::new(());

/// Takes the lock and clears the counters, so a test starts from nothing
/// whatever the one before it did. A failing test poisons the lock; that is
/// not a reason for the rest to fail too, so the poison is stepped over.
fn exclusively() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    DECODE_CALLS.store(0, Ordering::SeqCst);
    RELEASE_CALLS.store(0, Ordering::SeqCst);
    guard
}

fn install() {
    let decoder = WxScanImageDecoder {
        decode: Some(fake_decode),
        release: Some(fake_release),
        ctx: 0x1234 as *mut std::ffi::c_void,
    };
    unsafe { wxscan_set_image_decoder(&decoder) };
}

fn uninstall() {
    unsafe { wxscan_set_image_decoder(std::ptr::null()) };
}

unsafe fn plain_scanner() -> WxScanScannerId {
    let s = wxscan_scanner_new(std::ptr::null(), 0, std::ptr::null(), 0);
    assert_ne!(s, 0);
    s
}

fn bytes(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// The whole point: bytes that were UnsupportedFormat become a scan.
///
/// A 2x2 grey square has no symbol in it, so this comes back Ok and empty —
/// which is exactly the distinction the status exists to draw, and proof the
/// pixels reached the scanner rather than the bytes being rejected.
#[test]
fn a_host_decoder_is_consulted_for_what_this_build_cannot_read() {
    let _serial = exclusively();
    install();
    unsafe {
        let scanner = plain_scanner();
        let data = b"FAKE and then some payload".to_vec();
        let mut status = WxScanStatus::BadArgument;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        assert_eq!(status, WxScanStatus::Ok);
        assert!(!out.is_null());
        assert_eq!(((*out).width, (*out).height), (2, 2));
        assert_eq!((*out).results_len, 0);
        wxscan_results_free(out);
        wxscan_scanner_release(scanner);
    }
    assert_eq!(DECODE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(RELEASE_CALLS.load(Ordering::SeqCst), 1, "the buffer is given back");
    uninstall();
}

/// Second, never first. A png must be read by the built-in decoder whether or
/// not a host has lent one, or installing a decoder would quietly change how
/// every ordinary picture is read.
#[test]
fn the_built_in_decoders_answer_first() {
    let _serial = exclusively();
    install();
    unsafe {
        let scanner = plain_scanner();
        let data = bytes("upright.png");
        let mut status = WxScanStatus::BadArgument;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        assert_eq!(status, WxScanStatus::Ok);
        assert_eq!((*out).results_len, 1);
        wxscan_results_free(out);
        wxscan_scanner_release(scanner);
    }
    assert_eq!(
        DECODE_CALLS.load(Ordering::SeqCst),
        0,
        "the host should never have been asked about a png"
    );
    uninstall();
}

/// A host that does not recognise the bytes either leaves the answer where it
/// was, rather than turning a format question into something else.
#[test]
fn a_host_that_declines_leaves_the_old_answer() {
    let _serial = exclusively();
    install();
    unsafe {
        let scanner = plain_scanner();
        let data = b"neither a picture nor the fake format".to_vec();
        let mut status = WxScanStatus::Ok;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        assert!(out.is_null());
        assert_eq!(status, WxScanStatus::UnsupportedFormat);
        wxscan_scanner_release(scanner);
    }
    uninstall();
}

/// Clearing it puts the library back exactly as it was.
#[test]
fn taking_the_decoder_back_restores_the_old_behaviour() {
    let _serial = exclusively();
    install();
    uninstall();
    unsafe {
        let scanner = plain_scanner();
        let data = b"FAKE and then some payload".to_vec();
        let mut status = WxScanStatus::Ok;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        assert!(out.is_null());
        assert_eq!(status, WxScanStatus::UnsupportedFormat);
        wxscan_scanner_release(scanner);
    }
}

/// Half a decoder is not a decoder, and is refused at the door rather than
/// being checked for on every picture.
#[test]
fn a_decoder_missing_half_of_itself_is_not_installed() {
    let _serial = exclusively();
    unsafe {
        let half = WxScanImageDecoder {
            decode: Some(fake_decode),
            release: None,
            ctx: 0x1234 as *mut std::ffi::c_void,
        };
        wxscan_set_image_decoder(&half);

        let scanner = plain_scanner();
        let data = b"FAKE and then some payload".to_vec();
        let mut status = WxScanStatus::Ok;
        let out = wxscan_scan_bytes(scanner, data.as_ptr(), data.len(), &mut status);
        assert!(out.is_null(), "a half-installed decoder must not be called");
        assert_eq!(status, WxScanStatus::UnsupportedFormat);
        wxscan_scanner_release(scanner);
    }
}
