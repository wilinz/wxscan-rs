//! Results as JSON, for crossing a worker boundary.
//!
//! A scan produces pointers into this module's memory: strings, byte arrays,
//! coordinate arrays. None of that survives `postMessage`, and copying the
//! whole heap to the page would be worse than the scan. So the browser binding
//! serializes, exactly as the Swift and Kotlin bindings already do for camera
//! frames — and to the same document, so `parseFrameJson` on the Dart side
//! reads both without knowing which produced it.
//!
//! The C ABI itself stays as it is. Serialization belongs to a binding, and
//! this crate is the browser's.

use std::fmt::Write as _;

use wxscan::detector::ssd_detector::QuadPoints;
use wxscan::QRCodeResult;

/// Escapes the few characters JSON forbids in a string.
fn escape(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

/// Interpret payload bytes according to the charset the decoder reported.
///
/// The same conversion the C ABI performs for its `text` field: GB2312 content
/// is decoded, everything else is read as UTF-8 with invalid sequences
/// replaced.
fn text_of(result: &QRCodeResult) -> String {
    if result.charset.eq_ignore_ascii_case("GB2312") {
        let (cow, _, _) = encoding_rs::GBK.decode(&result.bytes);
        return cow.into_owned();
    }
    String::from_utf8_lossy(&result.bytes).into_owned()
}

fn write_points(out: &mut String, points: &QuadPoints) {
    out.push('[');
    for (i, (x, y)) in points.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{x},{y}");
    }
    out.push(']');
}

/// The document `parseFrameJson` reads.
pub(crate) fn document(
    results: &[QRCodeResult],
    candidates: &[QuadPoints],
    width: u32,
    height: u32,
) -> String {
    let mut out = String::with_capacity(256 + results.len() * 192);
    let _ = write!(out, "{{\"w\":{width},\"h\":{height},\"results\":[");
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"text\":\"");
        escape(&mut out, &text_of(r));
        out.push_str("\",\"raw\":[");
        for (j, b) in r.bytes.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            let _ = write!(out, "{b}");
        }
        out.push_str("],\"charset\":\"");
        escape(&mut out, &r.charset);
        out.push_str("\",\"points\":");
        write_points(&mut out, &r.points);
        let _ = write!(out, ",\"version\":{}", r.qrcode_version);
        out.push_str(",\"ecLevel\":\"");
        escape(&mut out, &r.ec_level);
        out.push_str("\",\"charsetMode\":\"");
        escape(&mut out, &r.charset_mode);
        let _ = write!(out, "\",\"binaryMethod\":{}}}", r.binary_method);
    }
    out.push_str("],\"candidates\":[");
    for (i, c) in candidates.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_points(&mut out, c);
    }
    out.push_str("]}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(bytes: &[u8], charset: &str) -> QRCodeResult {
        QRCodeResult {
            bytes: bytes.to_vec(),
            charset: charset.to_string(),
            points: [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)],
            qrcode_version: 3,
            ec_level: "M".to_string(),
            charset_mode: "BYTE".to_string(),
            binary_method: 0,
        }
    }

    #[test]
    fn writes_the_document_the_dart_side_reads() {
        let doc = document(&[result(b"hi", "UTF-8")], &[[(0.0, 0.0); 4]], 640, 480);
        assert!(doc.starts_with("{\"w\":640,\"h\":480,\"results\":["));
        assert!(doc.contains("\"text\":\"hi\""));
        assert!(doc.contains("\"raw\":[104,105]"));
        assert!(doc.contains("\"points\":[1,2,3,4,5,6,7,8]"));
        assert!(doc.contains("\"version\":3"));
        assert!(doc.ends_with("\"candidates\":[[0,0,0,0,0,0,0,0]]}"));
    }

    #[test]
    fn escapes_what_json_forbids() {
        let doc = document(&[result(b"a\"b\\c\nd", "UTF-8")], &[], 1, 1);
        assert!(doc.contains(r#""text":"a\"b\\c\nd""#));
    }

    #[test]
    fn decodes_gb2312_for_the_text_field() {
        // 中文 in GBK, which is not valid UTF-8.
        let doc = document(&[result(&[0xd6, 0xd0, 0xce, 0xc4], "GB2312")], &[], 1, 1);
        assert!(doc.contains("\"text\":\"中文\""), "{doc}");
    }
}
