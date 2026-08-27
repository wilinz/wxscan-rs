//! The exported C entry points for scanning.
//!
//! `wxscan_scan_gray` takes an already upright, tightly packed grayscale image.
//! `wxscan_scan_frame` additionally takes a row stride and a rotation and
//! prepares the frame first, which is what camera pipelines need.

use wxscan::frame::upright_gray;
use crate::results::{into_c, WxScanResults};
use crate::scanner::WxScanScanner;

/// Slice a caller-provided buffer after validating the geometry.
///
/// # Safety
/// `data` must point to at least `row_stride * height` readable bytes.
unsafe fn frame_slice<'a>(
    data: *const u8,
    width: i32,
    height: i32,
    row_stride: i32,
) -> Option<&'a [u8]> {
    if data.is_null() || width <= 0 || height <= 0 || row_stride < width {
        return None;
    }
    Some(std::slice::from_raw_parts(
        data,
        row_stride as usize * height as usize,
    ))
}

/// Run detection and decoding on an upright, tightly packed grayscale image.
///
/// Returns NULL if the scanner or the buffer is invalid; an empty result set
/// means nothing was found. The result must be released with
/// [`crate::results::wxscan_results_free`].
///
/// # Safety
/// `scanner` must come from [`crate::scanner::wxscan_scanner_new`] and `data`
/// must point to at least `width * height` readable bytes that stay valid for
/// the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn wxscan_scan_gray(
    scanner: *const WxScanScanner,
    data: *const u8,
    width: i32,
    height: i32,
) -> *mut WxScanResults {
    wxscan_scan_frame(scanner, data, width, height, width, 0, 0)
}

/// The layout of a colour buffer handed to [`wxscan_scan_pixels`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WxScanPixelFormat {
    /// One byte per pixel, already grayscale.
    Gray = 0,
    /// Three bytes per pixel, red first. What an image decoder usually gives.
    Rgb = 1,
    /// Four bytes per pixel, red first; the alpha channel is ignored.
    Rgba = 2,
    /// Three bytes per pixel, blue first.
    Bgr = 3,
    /// Four bytes per pixel, blue first; the alpha channel is ignored.
    Bgra = 4,
}

impl WxScanPixelFormat {
    fn from_raw(v: i32) -> Option<Self> {
        Some(match v {
            0 => Self::Gray,
            1 => Self::Rgb,
            2 => Self::Rgba,
            3 => Self::Bgr,
            4 => Self::Bgra,
            _ => return None,
        })
    }

    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }
}

/// Run detection and decoding on a colour image, converting it to grayscale
/// first.
///
/// This exists so that a caller decoding a PNG or a JPEG does not have to
/// convert the pixels itself; the conversion here is the same one the algorithm
/// applies internally.
///
/// `format` is a [`WxScanPixelFormat`]. Rows are tightly packed, so the buffer
/// must hold `width * height * bytes_per_pixel` bytes.
///
/// Returns NULL on invalid input. The result must be released with
/// [`crate::results::wxscan_results_free`].
///
/// # Safety
/// `scanner` must come from [`crate::scanner::wxscan_scanner_new`] and `data`
/// must point to at least that many readable bytes, valid for the duration of
/// the call.
#[no_mangle]
pub unsafe extern "C" fn wxscan_scan_pixels(
    scanner: *const WxScanScanner,
    data: *const u8,
    width: i32,
    height: i32,
    format: i32,
) -> *mut WxScanResults {
    let Some(format) = WxScanPixelFormat::from_raw(format) else {
        return std::ptr::null_mut();
    };
    if scanner.is_null() || data.is_null() || width <= 0 || height <= 0 {
        return std::ptr::null_mut();
    }
    let (w, h) = (width as usize, height as usize);
    let Some(len) = w.checked_mul(h).and_then(|n| n.checked_mul(format.bytes_per_pixel())) else {
        return std::ptr::null_mut();
    };
    let src = std::slice::from_raw_parts(data, len);

    let gray = match format {
        WxScanPixelFormat::Gray => return wxscan_scan_gray(scanner, data, width, height),
        WxScanPixelFormat::Rgb => cvlite::color::rgb_to_gray(src, w, h),
        WxScanPixelFormat::Rgba => cvlite::color::rgba_to_gray(src, w, h),
        WxScanPixelFormat::Bgr => cvlite::color::bgr_to_gray(src, w, h),
        WxScanPixelFormat::Bgra => bgra_to_gray(src, w, h),
    };
    wxscan_scan_gray(scanner, gray.as_ptr(), width, height)
}

/// BGRA to grayscale. cvlite covers the other four layouts; this one is BGR
/// with an ignored alpha, so it reads the three channels it needs in one pass
/// rather than copying the buffer to swap them.
fn bgra_to_gray(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    // The same fixed-point coefficients cvlite uses, so every layout converts
    // identically.
    const B2Y: u32 = 3735;
    const G2Y: u32 = 19235;
    const R2Y: u32 = 9798;
    const SHIFT: u32 = 14;

    let mut dst = vec![0u8; width * height];
    for (out, px) in dst.iter_mut().zip(src.chunks_exact(4)) {
        let (b, g, r) = (px[0] as u32, px[1] as u32, px[2] as u32);
        *out = ((b * B2Y + g * G2Y + r * R2Y + (1 << (SHIFT - 1))) >> SHIFT) as u8;
    }
    dst
}

/// Prepare a camera frame and scan it.
///
/// * `row_stride` is the byte distance between rows of the Y plane and may
///   exceed `width`.
/// * `rotation` is the clockwise angle in degrees needed to bring the frame
///   upright.
/// * `mirror_output`, when non-zero, mirrors the returned x coordinates about
///   the vertical axis of the upright frame. The frame itself is never
///   mirrored: the CNN detector is trained on unmirrored input, so mirroring it
///   lowers the detection rate. Use this when the preview is displayed
///   mirrored, as front-facing camera previews usually are.
///
/// Returns NULL on invalid input. The result must be released with
/// [`crate::results::wxscan_results_free`].
///
/// # Safety
/// `scanner` must come from [`crate::scanner::wxscan_scanner_new`] and `data`
/// must point to at least `row_stride * height` readable bytes that stay valid
/// for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn wxscan_scan_frame(
    scanner: *const WxScanScanner,
    data: *const u8,
    width: i32,
    height: i32,
    row_stride: i32,
    rotation: i32,
    mirror_output: i32,
) -> *mut WxScanResults {
    let Some(scanner) = scanner.as_ref() else {
        return std::ptr::null_mut();
    };
    let Some(bytes) = frame_slice(data, width, height, row_stride) else {
        return std::ptr::null_mut();
    };

    let (gray, ow, oh) = upright_gray(
        bytes,
        width as usize,
        height as usize,
        row_stride as usize,
        rotation,
    );

    let (results, candidates) = scanner.scan_upright(&gray, ow, oh);

    let flip_x = (mirror_output != 0).then_some(ow as f32);
    into_c(results, candidates, ow as u32, oh as u32, flip_x)
}

/// Link probe: returns 1 when the library is linked in correctly.
#[no_mangle]
pub extern "C" fn wxscan_ping() -> i32 {
    1
}
