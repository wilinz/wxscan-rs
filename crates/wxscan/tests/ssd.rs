//! Checks that the Rust `PriorBox` / `DetectionOutput` implementations match
//! the output of OpenCV's caffe layers.
//!
//! The reference tensors are exported by
//! `tools/model_conversion/ref_dump.py` using the caffe importer of
//! OpenCV 4.x.

use wxscan::detector::detection_output::{forward, DetectionOutputParams};
use wxscan::detector::priorbox::generate_all_priors;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn check_case(tag: &str, img_w: usize, img_h: usize) {
    let base = format!("tests/data/ssd/{tag}");
    let loc = read_f32(&format!("{base}_loc.bin"));
    let conf = read_f32(&format!("{base}_conf.bin"));
    let prior_ref = read_f32(&format!("{base}_priorbox.bin"));
    let det_ref = read_f32(&format!("{base}_detout.bin"));

    // ── prior boxes ──
    let priors = generate_all_priors(img_w, img_h);
    assert_eq!(priors.len() * 4, loc.len(), "{tag}: prior count");
    let half = prior_ref.len() / 2;
    let mut max_box_diff = 0f32;
    let mut max_var_diff = 0f32;
    for (i, p) in priors.iter().enumerate() {
        for k in 0..4 {
            max_box_diff = max_box_diff.max((p.bbox[k] - prior_ref[i * 4 + k]).abs());
            max_var_diff = max_var_diff.max((p.variance[k] - prior_ref[half + i * 4 + k]).abs());
        }
    }
    assert!(max_box_diff < 1e-6, "{tag}: prior box diff {max_box_diff}");
    assert!(max_var_diff < 1e-6, "{tag}: prior variance diff {max_var_diff}");

    // ── detection output ──
    let dets = forward(&loc, &conf, &priors, &DetectionOutputParams::default());
    let expected: Vec<&[f32]> = det_ref
        .chunks_exact(7)
        .filter(|r| r[2] > 0.0)
        .collect();
    assert_eq!(dets.len(), expected.len(), "{tag}: detection count");
    for (got, want) in dets.iter().zip(expected.iter()) {
        assert_eq!(got.label, want[1], "{tag}: label");
        assert!((got.score - want[2]).abs() < 1e-5, "{tag}: score {} vs {}", got.score, want[2]);
        for (k, v) in [got.xmin, got.ymin, got.xmax, got.ymax].iter().enumerate() {
            assert!(
                (v - want[3 + k]).abs() < 1e-5,
                "{tag}: coord {k}: {v} vs {}",
                want[3 + k]
            );
        }
    }
}

#[test]
fn ssd_postprocess_matches_opencv_384() {
    check_case("det", 384, 384);
}

#[test]
fn ssd_postprocess_matches_opencv_320x224() {
    check_case("det2", 320, 224);
}
