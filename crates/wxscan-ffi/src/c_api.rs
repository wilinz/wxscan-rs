//! The exported C entry points for scanning.
//!
//! `wxscan_scan_gray` takes an already upright, tightly packed grayscale image.
//! `wxscan_scan_frame` additionally takes a row stride and a rotation and
//! prepares the frame first, which is what camera pipelines need.

#[cfg(feature = "image-io")]
use std::ffi::c_char;
#[cfg(feature = "image-io")]
use image::ImageDecoder;
#[cfg(feature = "image-io")]
use std::io::{BufRead, Seek};

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
        WxScanPixelFormat::Bgra => cvlite::color::bgra_to_gray(src, w, h),
    };
    wxscan_scan_gray(scanner, gray.as_ptr(), width, height)
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

/// Why [`wxscan_scan_path`] returned nothing.
///
/// A scan that finds no symbol is [`WxScanStatus::Ok`] with an empty result
/// set, which is a different thing from a file that could not be read at all.
/// Collapsing the two is how a picture the library never even saw comes back
/// looking like a picture with no code in it.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WxScanStatus {
    /// The file was read and scanned. The result set may still be empty.
    Ok = 0,
    /// A null pointer, a path that is not UTF-8, or a null scanner.
    BadArgument = 1,
    /// The path could not be opened or read.
    Unreadable = 2,
    /// The bytes were read but are not an image this build can decode. PNG,
    /// JPEG and GIF are — the `image` dependency enables those three and no
    /// others, because they are what a photo picker actually writes and the
    /// rest cost 570 KB. HEIC is not, and nor is anything else needing a
    /// system framework.
    ///
    /// A photo library is mostly HEIC, but a picker generally does not hand it
    /// over that way: on iOS, `image_picker` sniffs the first byte, finds
    /// neither JPEG nor PNG nor GIF, and re-encodes to JPEG on its way to disk.
    /// The paths worth worrying about are the ones that come from somewhere
    /// else — a file shared into the application, say — and a caller that has
    /// to read those needs the platform's own decoder and
    /// [`wxscan_scan_pixels`].
    UnsupportedFormat = 3,
}

/// Read an image file and scan it.
///
/// This exists so that a caller holding a path does not have to decode the
/// picture itself and hand over the pixels: a 12 megapixel photograph is 48 MB
/// as RGBA, and a caller that crosses a thread or an isolate boundary pays for
/// that buffer more than once. Here the file is read, decoded and reduced to
/// grayscale without any of it crossing the boundary.
///
/// `status`, when not NULL, is set to a [`WxScanStatus`] saying what happened.
/// Returns NULL for anything other than [`WxScanStatus::Ok`]. The result must
/// be released with [`crate::results::wxscan_results_free`].
///
/// The orientation recorded in the file is applied, so a photograph taken with
/// the phone turned sideways is scanned upright and the coordinates come back
/// in the picture as it is meant to be seen.
///
/// # Safety
/// `scanner` must come from [`crate::scanner::wxscan_scanner_new`], `path` must
/// be a NUL terminated string, and `status`, when not NULL, must point to a
/// writable [`WxScanStatus`].
#[cfg(feature = "image-io")]
#[no_mangle]
pub unsafe extern "C" fn wxscan_scan_path(
    scanner: *const WxScanScanner,
    path: *const c_char,
    status: *mut WxScanStatus,
) -> *mut WxScanResults {
    let set = |s: WxScanStatus| {
        if !status.is_null() {
            *status = s;
        }
    };

    let Some(scanner) = scanner.as_ref() else {
        set(WxScanStatus::BadArgument);
        return std::ptr::null_mut();
    };
    if path.is_null() {
        set(WxScanStatus::BadArgument);
        return std::ptr::null_mut();
    }
    let Ok(path) = std::ffi::CStr::from_ptr(path).to_str() else {
        set(WxScanStatus::BadArgument);
        return std::ptr::null_mut();
    };

    // Opening and decoding are separate failures on purpose: a path that is not
    // there and a file this build has no decoder for want different answers
    // from the caller, and only the second is worth retrying another way.
    let reader = match image::ImageReader::open(path).and_then(|r| r.with_guessed_format()) {
        Ok(r) => r,
        Err(_) => {
            set(WxScanStatus::Unreadable);
            return std::ptr::null_mut();
        }
    };
    decode_and_scan(scanner, reader, set)
}

/// Decode an encoded image already in memory and scan it.
///
/// The same work as [`wxscan_scan_path`] for a caller that has the bytes
/// rather than a path: an image picked from a photo library and handed over as
/// data, a download, an asset, or a browser, which has no filesystem to give a
/// path into at all.
///
/// `data` is the *encoded* file — PNG, JPEG or GIF, the three this build
/// carries decoders for — not pixels. For pixels use [`wxscan_scan_pixels`].
/// The format is sniffed from the bytes, so no caller has to say which it is.
///
/// `status`, when not NULL, is set to a [`WxScanStatus`] saying what happened.
/// [`WxScanStatus::Unreadable`] cannot arise here — there is nothing to open —
/// so a buffer that is not a picture this build decodes comes back as
/// [`WxScanStatus::UnsupportedFormat`]. Returns NULL for anything other than
/// [`WxScanStatus::Ok`]. The result must be released with
/// [`crate::results::wxscan_results_free`].
///
/// The orientation recorded in the file is applied, exactly as for a path.
///
/// # Safety
/// `scanner` must come from [`crate::scanner::wxscan_scanner_new`], `data`
/// must point to at least `len` readable bytes that stay valid for the
/// duration of the call, and `status`, when not NULL, must point to a writable
/// [`WxScanStatus`].
#[cfg(feature = "image-io")]
#[no_mangle]
pub unsafe extern "C" fn wxscan_scan_bytes(
    scanner: *const WxScanScanner,
    data: *const u8,
    len: usize,
    status: *mut WxScanStatus,
) -> *mut WxScanResults {
    let set = |s: WxScanStatus| {
        if !status.is_null() {
            *status = s;
        }
    };

    let Some(scanner) = scanner.as_ref() else {
        set(WxScanStatus::BadArgument);
        return std::ptr::null_mut();
    };
    if data.is_null() {
        set(WxScanStatus::BadArgument);
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data, len);

    // with_guessed_format on a Cursor reads from memory and cannot fail for
    // want of I/O, so the only way past here is a format question.
    let Ok(reader) =
        image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()
    else {
        set(WxScanStatus::UnsupportedFormat);
        return std::ptr::null_mut();
    };
    decode_and_scan(scanner, reader, set)
}

/// Decode an image and scan it, however its bytes were reached.
///
/// Shared by [`wxscan_scan_path`] and [`wxscan_scan_bytes`]: once there is a
/// reader, a file and a buffer are the same problem, and the EXIF handling
/// below is the part worth having in one place rather than two.
#[cfg(feature = "image-io")]
fn decode_and_scan<R: BufRead + Seek>(
    scanner: &WxScanScanner,
    reader: image::ImageReader<R>,
    set: impl Fn(WxScanStatus),
) -> *mut WxScanResults {
    let Ok(decoder) = reader.into_decoder() else {
        set(WxScanStatus::UnsupportedFormat);
        return std::ptr::null_mut();
    };
    // Read before the decoder is consumed; a photograph usually stores its
    // pixels in the sensor's orientation and the rotation as a tag beside them.
    let mut decoder = decoder;
    let orientation = decoder.orientation().ok();
    let Ok(image) = image::DynamicImage::from_decoder(decoder) else {
        set(WxScanStatus::UnsupportedFormat);
        return std::ptr::null_mut();
    };
    let mut image = image;
    if let Some(orientation) = orientation {
        image.apply_orientation(orientation);
    }

    let gray = image.into_luma8();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let (results, candidates) = scanner.scan_upright(&gray.into_raw(), w, h);
    set(WxScanStatus::Ok);
    into_c(results, candidates, w as u32, h as u32, None)
}

/// Link probe: returns 1 when the library is linked in correctly.
#[no_mangle]
pub extern "C" fn wxscan_ping() -> i32 {
    1
}
