//! The wxscan C ABI, compiled to WebAssembly for the browser.
//!
//! Everything callable is in [`wxscan_ffi`]; this crate adds the three things a
//! wasm module needs that a native library gets from its platform, and produces
//! the `cdylib` that a browser can instantiate.
//!
//! * `malloc` and `free`, so the host can put an image into linear memory and
//!   take a result out. A wasm module has no other way to be handed bytes.
//! * A source of randomness for `getrandom`, which tract pulls in transitively.
//! * A reference to each exported function, so the linker does not discard the
//!   ABI as unreachable — nothing inside this crate calls it.
//!
//! By default inference is the host's job — see [`wxscan_ffi::host_net`] — which
//! is what keeps this module at a quarter of a megabyte. The `tract` feature
//! compiles an ONNX engine in instead, for a module of about twelve.

// Nothing here means anything off wasm, and `malloc` and `free` would take
// over the host allocator's names if this crate were ever linked natively —
// which `cargo test` does. Compile to nothing instead.
#![cfg(target_arch = "wasm32")]

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
