//! The scanner handle: construction, destruction, and its inference backend.

use std::sync::Mutex;

use wxscan::net::{Net, NetOutput};
use wxscan::detector::ssd_detector::QuadPoints;
use wxscan::{QRCodeResult, WeChatQRCode};

/// Inference backend. Empty when no backend feature is on, in which case a
/// scanner can only be created without models.
///
/// With both `tflite` and `tract` on, the weights decide: the model buffer is
/// tried as TFLite first and then as ONNX, so one binary takes either format.
pub(crate) enum Backend {
    #[cfg(feature = "tflite")]
    Tflite(wxscan::tflite::TfliteNet),
    #[cfg(feature = "tract")]
    Tract(wxscan::backend::tract::TractNet),
    /// Inference happens outside this library; see [`crate::host_net`].
    #[cfg(all(feature = "host-net", target_arch = "wasm32"))]
    Host(crate::host_net::HostNet),
}

impl Net for Backend {
    #[allow(unused_variables)]
    fn forward(&self, input: &[f32], shape: &[usize]) -> Result<Vec<NetOutput>, String> {
        match self {
            #[cfg(feature = "tflite")]
            Backend::Tflite(n) => n.forward(input, shape),
            #[cfg(feature = "tract")]
            Backend::Tract(n) => n.forward(input, shape),
            #[cfg(all(feature = "host-net", target_arch = "wasm32"))]
            Backend::Host(n) => n.forward(input, shape),
            #[cfg(not(any(feature = "tflite", feature = "tract", all(feature = "host-net", target_arch = "wasm32"))))]
            _ => Err("wxscan: built without an inference backend".to_string()),
        }
    }
}

/// A scanner instance. Opaque to the C side.
///
/// Scanning takes `&self`, but the underlying algorithm keeps mutable state
/// (decoder rotation, connected-component caches), so one instance scans one
/// frame at a time. Create several instances to scan in parallel.
pub struct WxScanScanner {
    pub(crate) inner: Mutex<WeChatQRCode<Backend>>,
}

impl WxScanScanner {
    /// Scan an upright, tightly packed grayscale image, returning the decoded
    /// results together with the detector candidates.
    ///
    /// This is for Rust callers that link the crate as a library, such as a
    /// platform binding that needs the results in a form the C ABI does not
    /// carry. C callers go through [`crate::wxscan_scan_gray`] instead.
    ///
    /// Returns empty vectors if the scanner is already in use on another thread.
    pub fn scan_upright(
        &self,
        gray: &[u8],
        width: usize,
        height: usize,
    ) -> (Vec<QRCodeResult>, Vec<QuadPoints>) {
        match self.inner.lock() {
            Ok(inner) => inner.detect_and_decode_gray_with_candidates(gray, width, height),
            Err(_) => (Vec::new(), Vec::new()),
        }
    }
}

/// Create a scanner from in-memory model buffers.
///
/// Passing NULL for both models selects the image-processing-only mode. It
/// still decodes, but the detection rate for small or distant symbols is
/// considerably lower, since that is what the CNN stage contributes.
///
/// Returns NULL if a model fails to load. Release with [`wxscan_scanner_free`].
///
/// # Safety
/// `detect` and `sr`, when not NULL, must point to at least the corresponding
/// number of readable bytes.
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_new(
    detect: *const u8,
    detect_len: usize,
    sr: *const u8,
    sr_len: usize,
) -> *mut WxScanScanner {
    let load = |p: *const u8, n: usize| -> Result<Option<Backend>, ()> {
        if p.is_null() || n == 0 {
            return Ok(None);
        }
        let bytes = std::slice::from_raw_parts(p, n);
        // Each enabled backend gets a turn, so a build with both takes either
        // weight format and the caller does not have to say which it has.
        #[cfg(feature = "tflite")]
        if let Ok(n) = wxscan::tflite::TfliteNet::from_bytes(bytes) {
            return Ok(Some(Backend::Tflite(n)));
        }
        #[cfg(feature = "tract")]
        if let Ok(n) = wxscan::backend::tract::TractNet::from_bytes(bytes) {
            return Ok(Some(Backend::Tract(n)));
        }
        let _ = bytes;
        Err(())
    };

    let (detect, sr) = match (load(detect, detect_len), load(sr, sr_len)) {
        (Ok(d), Ok(s)) => (d, s),
        _ => return std::ptr::null_mut(),
    };

    Box::into_raw(Box::new(WxScanScanner {
        inner: Mutex::new(WeChatQRCode::new(detect, sr)),
    }))
}

/// Create a scanner whose inference runs in the host, for the wasm build.
///
/// The weights are never passed in: the host holds them, and each flag says
/// only whether that network is available. Passing zero for both is the mode
/// without models, exactly as for [`wxscan_scanner_new`].
///
/// Release with [`wxscan_scanner_free`].
#[cfg(all(feature = "host-net", target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn wxscan_scanner_new_host(has_detector: i32, has_sr: i32) -> *mut WxScanScanner {
    let detect = (has_detector != 0).then(crate::host_net::HostNet::detector);
    let sr = (has_sr != 0).then(crate::host_net::HostNet::super_resolution);
    Box::into_raw(Box::new(WxScanScanner {
        inner: Mutex::new(WeChatQRCode::new(
            detect.map(Backend::Host),
            sr.map(Backend::Host),
        )),
    }))
}

/// Set the downscale factor applied before detection.
///
/// Values outside `(0, 1]` restore the default, which targets a 400x400 area.
///
/// # Safety
/// `s` must come from [`wxscan_scanner_new`].
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_set_scale_factor(s: *mut WxScanScanner, v: f32) {
    if let Some(s) = s.as_ref() {
        if let Ok(mut inner) = s.inner.lock() {
            inner.set_scale_factor(v);
        }
    }
}

/// Destroy a scanner. Passing NULL is a no-op.
///
/// # Safety
/// The pointer must come from [`wxscan_scanner_new`] and must be freed at most once.
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_free(s: *mut WxScanScanner) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// The downscale factor applied before detection, as set by
/// [`wxscan_scanner_set_scale_factor`]. A negative value means the default,
/// which targets a 400x400 area.
///
/// # Safety
/// `s` must come from [`wxscan_scanner_new`].
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_scale_factor(s: *const WxScanScanner) -> f32 {
    match s.as_ref().and_then(|s| s.inner.lock().ok()) {
        Some(inner) => inner.scale_factor(),
        None => -1.0,
    }
}

/// How confident the detector must be to report a candidate, 0.2 by default.
///
/// Lower recalls more weak symbols along with more false positives; higher does
/// the reverse. Values outside `(0, 1)` are ignored. Without models there is no
/// detector and this does nothing.
///
/// # Safety
/// `s` must come from [`wxscan_scanner_new`].
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_set_confidence_threshold(s: *mut WxScanScanner, v: f32) {
    if !(v > 0.0 && v < 1.0) {
        return;
    }
    if let Some(mut inner) = s.as_ref().and_then(|s| s.inner.lock().ok()) {
        if let Some(params) = inner.detection_params_mut() {
            params.confidence_threshold = v;
        }
    }
}

/// The confidence threshold in use, or a negative value when no detector is
/// loaded.
///
/// # Safety
/// `s` must come from [`wxscan_scanner_new`].
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_confidence_threshold(s: *const WxScanScanner) -> f32 {
    match s.as_ref().and_then(|s| s.inner.lock().ok()) {
        Some(inner) => inner.detection_params().map_or(-1.0, |p| p.confidence_threshold),
        None => -1.0,
    }
}

/// The IoU above which two overlapping candidates are treated as one symbol,
/// 0.45 by default. Values outside `(0, 1)` are ignored.
///
/// # Safety
/// `s` must come from [`wxscan_scanner_new`].
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_set_nms_threshold(s: *mut WxScanScanner, v: f32) {
    if !(v > 0.0 && v < 1.0) {
        return;
    }
    if let Some(mut inner) = s.as_ref().and_then(|s| s.inner.lock().ok()) {
        if let Some(params) = inner.detection_params_mut() {
            params.nms_threshold = v;
        }
    }
}

/// The NMS threshold in use, or a negative value when no detector is loaded.
///
/// # Safety
/// `s` must come from [`wxscan_scanner_new`].
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_nms_threshold(s: *const WxScanScanner) -> f32 {
    match s.as_ref().and_then(|s| s.inner.lock().ok()) {
        Some(inner) => inner.detection_params().map_or(-1.0, |p| p.nms_threshold),
        None => -1.0,
    }
}

/// Whether the detector network is loaded.
///
/// # Safety
/// `s` must come from [`wxscan_scanner_new`].
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_has_detector(s: *const WxScanScanner) -> i32 {
    match s.as_ref().and_then(|s| s.inner.lock().ok()) {
        Some(inner) => i32::from(inner.has_detector()),
        None => 0,
    }
}

/// Whether the super resolution network is loaded.
///
/// # Safety
/// `s` must come from [`wxscan_scanner_new`].
#[no_mangle]
pub unsafe extern "C" fn wxscan_scanner_has_super_resolution(s: *const WxScanScanner) -> i32 {
    match s.as_ref().and_then(|s| s.inner.lock().ok()) {
        Some(inner) => i32::from(inner.has_super_resolution()),
        None => 0,
    }
}
