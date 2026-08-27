# wxscan-ffi

The C ABI for [`wxscan`](https://crates.io/crates/wxscan), for callers in C,
C++, Swift, Kotlin, Python and elsewhere.

`include/wxscan.h` is generated from these sources with cbindgen and committed,
so consumers need neither cbindgen nor a Rust toolchain.

```c
#include "wxscan.h"

// Both models may be NULL, which decodes without the CNN stages.
WxScanScanner *scanner = wxscan_scanner_new(detect, detect_len, sr, sr_len);

// An upright, tightly packed grayscale image.
WxScanResults *out = wxscan_scan_gray(scanner, gray, width, height);
for (size_t i = 0; i < out->results_len; i++) {
    printf("%s\n", out->results[i].text);
}
wxscan_results_free(out);

wxscan_scanner_free(scanner);
```

For a picture on disk, `wxscan_scan_path` reads and decodes the file itself, so
a caller holding a path never has to materialise the pixels: a 12 megapixel
photograph is 48 MB as RGBA, and a caller that hands that across a thread or an
isolate boundary pays for the buffer more than once.

```c
WxScanStatus status;
WxScanResults *out = wxscan_scan_path(scanner, "/tmp/photo.jpg", &status);
// status distinguishes a file that could not be read from a picture with no
// code in it; the second is Ok with an empty result set.
```

The orientation recorded in the file is applied, so a photograph taken with the
phone turned sideways scans upright and its coordinates match what was on
screen. PNG, JPEG and GIF are decoded, which is everything a photo picker
writes; HEIC is not, and a caller that must read one needs the platform's
decoder and `wxscan_scan_pixels`. The entry point is behind the `image-io`
feature, on by default; it costs 436 KB of decoders, so a build that has no use
for it can leave it out.

For camera frames, `wxscan_scan_frame` additionally takes a row stride, a
rotation, and a flag that mirrors the returned x coordinates. The frame itself
is never mirrored, because the detector is trained on unmirrored input; the flag
exists so coordinates line up with a preview that is displayed mirrored, as
front-facing previews usually are.

Results are plain C structs. Serialization, if a caller needs it, belongs in
that caller's binding layer.

The scanner is an explicit handle rather than a global, so several can coexist
with different models and calls do not contend on one lock. One instance scans
one frame at a time.

## Linking

The crate builds as a static library and an rlib. It deliberately produces no
cdylib: that would require resolving the TFLite symbols at build time, while the
design leaves them to the host build system's final link. A caller that wants a
shared object wraps this crate in one of its own and provides the search path
there — see the `wxscan_core` package in
[wxscan](https://github.com/wilinz/wxscan) for a worked example covering five
platforms.

Part of [wxscan-rs](https://github.com/wilinz/wxscan-rs). Apache-2.0.
