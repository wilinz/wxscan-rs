# Parity with the upstream C++ implementation

The same images are fed to OpenCV's `wechat_qrcode` (the original C++
implementation) and to the Rust port in this repository, and the decoded text
and the four corner coordinates are compared image by image.

## Environment

```bash
python3 -m venv venv  && ./venv/bin/pip install opencv-contrib-python numpy segno pillow
# Reference implementation: OpenCV 5.x removed the Caffe importer, so parity
# runs with models require 4.x
python3 -m venv venv4 && ./venv4/bin/pip install opencv-contrib-python==4.10.0.84 numpy
```

## Running

```bash
# 1. Generate the corpus (160 images: several versions and error correction
#    levels, with rotation, scaling, blur, noise, perspective and inversion)
./venv/bin/python gen_corpus.py
# 2. C++ reference results, without models
./venv/bin/python run_cpp.py corpus cpp_corpus.json
# 3. Rust results
(cd ../../crates/wxscan && cargo run --release --quiet --example dump -- ../../tools/parity/corpus/*.png) > rust_corpus.json
# 4. Compare
./venv/bin/python compare.py cpp_corpus.json rust_corpus.json corpus/manifest.json
```

For the path that uses the CNN models: `gen_scene.py` generates small codes
inside large images, `run_cpp_nn.py` produces the reference output (it needs
venv4 and the upstream Caffe models, which `tools/1_download_models.sh` in
[wxscan-weights](https://github.com/wilinz/wxscan-weights) fetches; pass their
directory as its third argument if that checkout is not next to this one), and the Rust
side runs
`cargo run --features tflite --example dump_nn -- <detect.tflite> <sr.tflite> scenes/*.png`.

## Current results

| Corpus | Agreement | Notes |
|---|---|---|
| 160 images, no models | text 159/160 | The one difference is an inverted image the Rust port decodes and the C++ implementation does not; the decoded text matches ground truth. Corners are bit-identical on 152/154, with sub-pixel differences on 2. |
| 24 scene images, with models | text 24/24 | Maximum corner difference 2.9 px, from the SSD boxes shifting slightly because TFLite and Caffe accumulate floating point in a different order. |

The sub-pixel differences originate in the Gaussian blur inside
`cv::adaptiveThreshold`: OpenCV uses a fixed-point implementation for 8U images
while this port accumulates in f32 (see the comments in
`src/threshold.rs` in the cvlite repository).
