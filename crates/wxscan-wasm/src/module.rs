//! The module itself: what a wasm build exports, and what it needs from the
//! platform that a native library gets for free.
//!
//! Off wasm this is not compiled at all. `malloc` and `free` would take over
//! the host allocator's names if it were, which `cargo test` would do.

use std::alloc::{alloc, dealloc, Layout};
use std::os::raw::c_void;

/// Bytes reserved before every allocation to record its size.
///
/// `free` in C takes only a pointer, while Rust's deallocator wants the layout
/// back. The size goes in a header, and the header is a full 16 bytes so that
/// what the caller receives keeps the alignment the allocator promised.
const HEADER: usize = 16;

/// Allocate `size` bytes and return a pointer the host can write to.
///
/// Returns null when `size` is zero or the allocation fails, which is what a C
/// caller expects. Release it with [`free`].
#[no_mangle]
pub extern "C" fn malloc(size: usize) -> *mut c_void {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let Some(total) = size.checked_add(HEADER) else {
        return std::ptr::null_mut();
    };
    let Ok(layout) = Layout::from_size_align(total, HEADER) else {
        return std::ptr::null_mut();
    };
    // SAFETY: the layout has a non-zero size, having just been given a header.
    unsafe {
        let base = alloc(layout);
        if base.is_null() {
            return std::ptr::null_mut();
        }
        (base as *mut usize).write(total);
        base.add(HEADER) as *mut c_void
    }
}

/// Release a pointer that came from [`malloc`]. A null pointer is ignored.
///
/// # Safety
/// `ptr` must have come from [`malloc`] and must not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let base = (ptr as *mut u8).sub(HEADER);
    let total = (base as *mut usize).read();
    if let Ok(layout) = Layout::from_size_align(total, HEADER) {
        dealloc(base, layout);
    }
}

/// Randomness for `getrandom`, which arrives through tract's dependencies.
///
/// Nothing in the scanning path draws from it — the graphs are loaded and run,
/// never initialised randomly — so rather than reach into JavaScript for
/// `crypto.getRandomValues` and make the module depend on a host binding, the
/// request fails. A caller that hits this gets an error instead of numbers that
/// only look random.
fn no_randomness(_: &mut [u8]) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}

getrandom::register_custom_getrandom!(no_randomness);

/// Function pointers to the exported ABI, so that it survives to the module's
/// export list.
///
/// Nothing in this crate calls into `wxscan_ffi`, and `#[no_mangle]` in a
/// dependency is not on its own a reason for the linker to keep the code.
/// Naming each one here is.
#[used]
static EXPORTED: Kept<15> = Kept([
    wxscan_ffi::wxscan_scanner_new as *const c_void,
    wxscan_ffi::wxscan_scanner_free as *const c_void,
    wxscan_ffi::wxscan_scanner_set_scale_factor as *const c_void,
    wxscan_ffi::wxscan_scanner_scale_factor as *const c_void,
    wxscan_ffi::wxscan_scanner_set_confidence_threshold as *const c_void,
    wxscan_ffi::wxscan_scanner_confidence_threshold as *const c_void,
    wxscan_ffi::wxscan_scanner_set_nms_threshold as *const c_void,
    wxscan_ffi::wxscan_scanner_nms_threshold as *const c_void,
    wxscan_ffi::wxscan_scanner_has_detector as *const c_void,
    wxscan_ffi::wxscan_scanner_has_super_resolution as *const c_void,
    wxscan_ffi::wxscan_scan_gray as *const c_void,
    wxscan_ffi::wxscan_scan_pixels as *const c_void,
    wxscan_ffi::wxscan_scan_frame as *const c_void,
    wxscan_ffi::wxscan_results_free as *const c_void,
    wxscan_ffi::wxscan_ping as *const c_void,
]);

/// The host backend's constructor, kept the same way.
#[cfg(all(feature = "host-net", target_arch = "wasm32"))]
#[used]
static EXPORTED_HOST: Kept<1> = Kept([wxscan_ffi::wxscan_scanner_new_host as *const c_void]);

struct Kept<const N: usize>([*const c_void; N]);

// SAFETY: the array holds function addresses. Nothing reads it; it exists so
// that the linker counts each function as used. Raw pointers are not `Sync` on
// their own, and a static has to be.
unsafe impl<const N: usize> Sync for Kept<N> {}

/// Panic reporting, for development.
///
/// A panic in a wasm module reaches the host as `RuntimeError: unreachable`,
/// with the message dropped on the floor — there is no stderr to print it to.
/// With this feature on, the module asks the host for one function and hands
/// the message to it.
#[cfg(feature = "debug-log")]
mod debug_log {
    #[link(wasm_import_module = "wxscan")]
    extern "C" {
        fn wxscan_host_log(ptr: *const u8, len: usize);
    }

    /// Installs the hook. Exported so the host can call it before anything else.
    #[no_mangle]
    pub extern "C" fn wxscan_install_panic_hook() {
        std::panic::set_hook(Box::new(|info| {
            let message = info.to_string();
            // SAFETY: the host supplies this import, and the slice outlives the call.
            unsafe { wxscan_host_log(message.as_ptr(), message.len()) };
        }));
    }
}

/// Pack a document into the one `u64` a wasm export can return: its address
/// in the upper half, its length in the lower.
fn packed(document: String) -> u64 {
    let bytes = document.into_bytes().into_boxed_slice();
    let len = bytes.len() as u64;
    let address = Box::into_raw(bytes) as *mut u8 as u64;
    (address << 32) | len
}

/// Run `scan`, and hand back the results as JSON rather than as C structs.
///
/// A worker cannot post the struct graph the C ABI returns — it is pointers
/// into this module's memory — so the browser binding takes a document
/// instead. The shape is the one the platform bindings already produce for
/// camera frames, which is what lets the Dart side parse both with one
/// function.
///
/// # Safety
/// `scanner` must come from one of the constructors, and the closure must not
/// read outside the buffer it is given.
unsafe fn scan_to_json(
    scanner: *const wxscan_ffi::WxScanScanner,
    scan: impl FnOnce(&wxscan_ffi::WxScanScanner) -> (Vec<wxscan::QRCodeResult>, Vec<wxscan::detector::ssd_detector::QuadPoints>, u32, u32),
) -> u64 {
    if scanner.is_null() {
        return 0;
    }
    let (results, candidates, width, height) = scan(&*scanner);
    packed(crate::json::document(&results, &candidates, width, height))
}

/// Scan an upright, tightly packed grayscale image. See [`scan_to_json`].
///
/// Returns `(address << 32) | length`, or zero. Release it with
/// [`wxscan_wasm_string_free`].
///
/// # Safety
/// `data` must point to at least `width * height` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn wxscan_wasm_scan_gray_json(
    scanner: *const wxscan_ffi::WxScanScanner,
    data: *const u8,
    width: i32,
    height: i32,
) -> u64 {
    if data.is_null() || width <= 0 || height <= 0 {
        return 0;
    }
    let (w, h) = (width as usize, height as usize);
    let gray = std::slice::from_raw_parts(data, w * h);
    scan_to_json(scanner, |s| {
        let (r, c) = s.scan_upright(gray, w, h);
        (r, c, width as u32, height as u32)
    })
}

/// Scan a colour image, converting it to grayscale first. See
/// [`wxscan_wasm_scan_gray_json`]; `format` is a `WxScanPixelFormat`.
///
/// # Safety
/// `data` must hold `width * height * bytes_per_pixel` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn wxscan_wasm_scan_pixels_json(
    scanner: *const wxscan_ffi::WxScanScanner,
    data: *const u8,
    width: i32,
    height: i32,
    format: i32,
) -> u64 {
    let Some(bytes_per_pixel) = pixel_stride(format) else {
        return 0;
    };
    if data.is_null() || width <= 0 || height <= 0 {
        return 0;
    }
    let (w, h) = (width as usize, height as usize);
    let pixels = std::slice::from_raw_parts(data, w * h * bytes_per_pixel);
    let gray = to_gray(pixels, w, h, format);
    scan_to_json(scanner, |s| {
        let (r, c) = s.scan_upright(&gray, w, h);
        (r, c, width as u32, height as u32)
    })
}

/// Scan a camera frame: a Y plane with a row stride, rotated upright first.
/// See [`wxscan_wasm_scan_gray_json`].
///
/// # Safety
/// `data` must hold `row_stride * height` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn wxscan_wasm_scan_frame_json(
    scanner: *const wxscan_ffi::WxScanScanner,
    data: *const u8,
    width: i32,
    height: i32,
    row_stride: i32,
    rotation: i32,
    mirror: i32,
) -> u64 {
    if data.is_null() || width <= 0 || height <= 0 || row_stride < width {
        return 0;
    }
    let plane = std::slice::from_raw_parts(data, row_stride as usize * height as usize);
    let (upright, ow, oh) = wxscan::frame::upright_gray(
        plane,
        width as usize,
        height as usize,
        row_stride as usize,
        rotation,
    );
    scan_to_json(scanner, |s| {
        let (mut r, mut c) = s.scan_upright(&upright, ow, oh);
        if mirror != 0 {
            let flip = |p: &mut (f32, f32)| p.0 = ow as f32 - p.0;
            for result in &mut r {
                result.points.iter_mut().for_each(flip);
            }
            for quad in &mut c {
                quad.iter_mut().for_each(flip);
            }
        }
        (r, c, ow as u32, oh as u32)
    })
}

/// Bytes per pixel for a `WxScanPixelFormat`, or `None` if it is not one.
fn pixel_stride(format: i32) -> Option<usize> {
    Some(match format {
        0 => 1,
        1 | 3 => 3,
        2 | 4 => 4,
        _ => return None,
    })
}

/// Convert to grayscale the way the algorithm does internally.
fn to_gray(pixels: &[u8], width: usize, height: usize, format: i32) -> Vec<u8> {
    match format {
        0 => pixels.to_vec(),
        1 => cvlite::color::rgb_to_gray(pixels, width, height),
        2 => cvlite::color::rgba_to_gray(pixels, width, height),
        3 => cvlite::color::bgr_to_gray(pixels, width, height),
        _ => {
            // BGRA, which cvlite does not have: drop alpha into the BGR path.
            let mut bgr = Vec::with_capacity(width * height * 3);
            for chunk in pixels.chunks_exact(4) {
                bgr.extend_from_slice(&chunk[..3]);
            }
            cvlite::color::bgr_to_gray(&bgr, width, height)
        }
    }
}

/// Release a document from [`wxscan_wasm_scan_gray_json`].
///
/// # Safety
/// `address` and `len` must be the halves of one value that call returned, and
/// must not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn wxscan_wasm_string_free(address: *mut u8, len: usize) {
    if !address.is_null() && len != 0 {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(address, len)));
    }
}
