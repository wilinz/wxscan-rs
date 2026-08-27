//! The caffe-SSD `PriorBox` layer.
//!
//! The tflite graph does not contain this layer, since it depends only on the
//! feature map sizes and a few constants, so it is computed on the Rust side.
//! The parameters come from `detect.prototxt`: five feature layers with six
//! prior boxes each (min, sqrt(min*max) and four aspect ratios).

/// Prior box configuration for a single feature layer.
#[derive(Clone, Copy, Debug)]
pub struct PriorBoxSpec {
    pub min_size: f32,
    pub max_size: f32,
    /// Excludes 1.0; caffe places the ratio 1.0 first internally
    pub aspect_ratios: [f32; 4],
    pub step: f32,
    pub variance: [f32; 4],
    pub offset: f32,
}

/// The five parameter sets from `detect.prototxt`, in the concat order of
/// mbox_priorbox.
pub const DETECT_PRIOR_SPECS: [PriorBoxSpec; 5] = [
    PriorBoxSpec { min_size: 50.0,  max_size: 100.0, aspect_ratios: [2.0, 0.5, 3.0, 1.0 / 3.0], step: 16.0, variance: [0.1, 0.1, 0.2, 0.2], offset: 0.5 },
    PriorBoxSpec { min_size: 100.0, max_size: 150.0, aspect_ratios: [2.0, 0.5, 3.0, 1.0 / 3.0], step: 32.0, variance: [0.1, 0.1, 0.2, 0.2], offset: 0.5 },
    PriorBoxSpec { min_size: 150.0, max_size: 200.0, aspect_ratios: [2.0, 0.5, 3.0, 1.0 / 3.0], step: 32.0, variance: [0.1, 0.1, 0.2, 0.2], offset: 0.5 },
    PriorBoxSpec { min_size: 200.0, max_size: 300.0, aspect_ratios: [2.0, 0.5, 3.0, 1.0 / 3.0], step: 32.0, variance: [0.1, 0.1, 0.2, 0.2], offset: 0.5 },
    PriorBoxSpec { min_size: 300.0, max_size: 400.0, aspect_ratios: [2.0, 0.5, 3.0, 1.0 / 3.0], step: 32.0, variance: [0.1, 0.1, 0.2, 0.2], offset: 0.5 },
];

/// Prior boxes per location: min, sqrt(min*max) and four aspect ratios.
pub const PRIORS_PER_LOCATION: usize = 6;

/// One prior box: normalized [xmin, ymin, xmax, ymax] and its variance.
#[derive(Clone, Copy, Debug)]
pub struct Prior {
    pub bbox: [f32; 4],
    pub variance: [f32; 4],
}

/// Generates the prior boxes for one layer, in the order used by caffe's
/// `PriorBoxLayer::Forward_cpu`.
///
/// `layer_w/h` are the feature map dimensions, `img_w/h` the network input
/// dimensions.
pub fn generate_priors_for_layer(
    spec: &PriorBoxSpec,
    layer_w: usize,
    layer_h: usize,
    img_w: usize,
    img_h: usize,
    out: &mut Vec<Prior>,
) {
    let (step_w, step_h) = if spec.step > 0.0 {
        (spec.step, spec.step)
    } else {
        (img_w as f32 / layer_w as f32, img_h as f32 / layer_h as f32)
    };
    let img_wf = img_w as f32;
    let img_hf = img_h as f32;

    for h in 0..layer_h {
        for w in 0..layer_w {
            let center_x = (w as f32 + spec.offset) * step_w;
            let center_y = (h as f32 + spec.offset) * step_h;

            let push = |bw: f32, bh: f32, out: &mut Vec<Prior>| {
                out.push(Prior {
                    bbox: [
                        (center_x - bw / 2.0) / img_wf,
                        (center_y - bh / 2.0) / img_hf,
                        (center_x + bw / 2.0) / img_wf,
                        (center_y + bh / 2.0) / img_hf,
                    ],
                    variance: spec.variance,
                });
            };

            // 1) Aspect ratio 1, side length min_size
            push(spec.min_size, spec.min_size, out);
            // 2) Aspect ratio 1, side length sqrt(min*max)
            let s = (spec.min_size * spec.max_size).sqrt();
            push(s, s, out);
            // 3) The remaining aspect ratios (caffe skips ar == 1)
            for &ar in spec.aspect_ratios.iter() {
                if (ar - 1.0).abs() < 1e-6 {
                    continue;
                }
                let sq = ar.sqrt();
                push(spec.min_size * sq, spec.min_size / sq, out);
            }
        }
    }
}

/// Derives the five feature layer sizes from the network input size and
/// generates all prior boxes.
///
/// The feature layers are downsampled from the input by 16, 32, 32, 32 and 32
/// (stage4_8, stage5_4, stage6_2, stage7_2, stage8_2; the last four share a
/// resolution).
pub fn generate_all_priors(img_w: usize, img_h: usize) -> Vec<Prior> {
    let strides = [16usize, 32, 32, 32, 32];
    let mut out = Vec::new();
    for (spec, stride) in DETECT_PRIOR_SPECS.iter().zip(strides.iter()) {
        let lw = img_w.div_ceil(*stride);
        let lh = img_h.div_ceil(*stride);
        generate_priors_for_layer(spec, lw, lh, img_w, img_h, &mut out);
    }
    out
}
