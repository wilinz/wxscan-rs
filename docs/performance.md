# Performance work

Every optimization here was made under one constraint: the algorithm's output
must not change. Verification is a full comparison of the Rust output before and
after on the 160-image parity corpus — zero text differences, zero corner
differences, maximum corner delta `0.000000 px`.

## Measuring

- **Local profiling**:
  `cargo run --release --features tflite,profiling --example profile -- <detect.tflite> <sr.tflite> <image>`
  prints per-stage timings. The instrumentation sits behind the `profiling`
  feature and is not compiled by default: on a noisy frame the flood fill runs
  200,000 times, and the `Instant::now()` calls alone would distort the result
  by 3 to 5 ms.
- **Against the C++ original**: `pip install opencv-contrib-python==4.10.0.84`
  and the upstream Caffe models, which `tools/convert.py download` in
  [wxscan-weights](https://github.com/wilinz/wxscan-weights) fetches, run on the
  same image and machine.
- **Parity**: `tools/parity/`, see the README there.

## Test samples

Four frames derived from real camera output, each exercising a different
pathological path:

| Sample | Characteristic | Bottleneck it triggers |
|---|---|---|
| Idle frame | no SSD candidates | detection only, the fastest path |
| Small pathological box | 217x224 candidate, decode fails | the decode ladder |
| Large pathological box | 599x533 candidate, decode fails | decode ladder and connected components |
| Noisy frame | sensor noise added, 1300+ false finder patterns | finder candidate explosion |

## Results

| Case | Before | After | C++ original |
|---|---|---|---|
| Idle frame | 4.8 ms | 4.6 ms | 4.9 ms |
| Small pathological box | 31.7 ms | 20.7 ms | 30.3 ms |
| Large pathological box | 55.3 ms | 29.2 ms | 42.5 ms |
| Noisy frame | — | 116.7 ms | — |

The large-box case went from 1.4x slower than the C++ implementation to 1.45x
faster.

## What was done

### 1. UnicomBlock: per-pixel BFS to scanline fill

`src/common/unicomblock.rs` in the wxing repository. The original did four neighbour checks
and one queue push/pop per pixel. The replacement marks a whole same-colour run
along a row at once and only pushes seeds for new runs on the rows above and
below, leaving roughly one byte read and one label write per pixel, with
sequential rather than random access.

**Why it is equivalent**: `index` is the component label, `count` is the
component size plus one, and min/max are bounding box extremes. None of the
three depends on visit order. The number of pixels visited is identical before
and after (5,077,154), which is evidence that coverage did not change.

### 2. UnicomBlock: component attributes stored per label, not per pixel

Same file. `count`, `min_pnt` and `max_pnt` used to be stored per pixel, so
filling a component meant writing the same value back to every pixel in it — one
`decode_more` wrote 3 x W x H x 4 bytes, a dozen times per frame. Indexing them
by component label removes the write-back entirely; reading takes one lookup
through the label already stored on the pixel.

### 3. Finder pattern detection: borrow the row run records instead of copying

`src/qrcode/detector/finder_pattern_finder.rs` in the wxing repository. Two `to_vec()`
calls meant two heap allocations plus a full copy per row, tens of thousands of
times per frame. The cause was that an accessor taking `&mut self` can only hand
out one slice at a time, so holding two forced a copy. `BitMatrix` gained three
read-only accessors — `row_records_at`, `row_records_offset_at` and
`row_counter_offset_end_at` — which callers use after `ensure_row_records`.

### 4. BitMatrix: reuse the working matrix

`BitMatrix::reuse_from` and `QRCodeReader::scratch`. One decode passes through a
dozen bitmaps (four binarizers x normal/inverted x several scales), each
allocating the bitmap itself plus three W x H row-run buffers. The companion
change was to explicitly zero `row_counters[base]` in `set_row_records`, the one
value that relied on a zeroed buffer, so the three buffers no longer need
clearing at all — only a flag array of `height` entries does.

This showed no measurable gain on macOS, where large allocations come from mmap
with lazily zeroed pages. It is kept because the Android allocator does a real
malloc plus memset.

## Tried and reverted

### Running the four binarizers in parallel

They are independent, so parallelising them looks obvious. Measured on a
Snapdragon 865 it took 400 ms of wall clock against 358 ms serial. The per-thread
times were 45/398/54/79 ms: the parallelism worked (wall clock equals the
longest thread), but almost all the time sits in FastWindow alone, and running
them concurrently only moved it onto a little core (4x1.8 GHz little, 3x2.4 GHz,
1x3.2 GHz), where that one binarizer went from 176 ms to 398 ms.

The cancellation plumbing (`find_cancellable`, `detect_cancellable`,
`decode_cancellable`) was kept, defaulting to `never_cancel()`, in case it is
useful later.

## Known bottlenecks, unresolved

- **FastWindow binarization amplifies noise.** It is a local mean over a 6-pixel
  window, so in flat low-contrast regions it turns sensor noise into a speckle
  field, and finder candidates go from a normal 3 to 10 up to 600 to 1300. The
  upstream C++ behaves the same way (`decodeImage` runs all four with
  `tryBinarizeTime = 4`); this is not something the port introduced. It accounts
  for 52% of decode time on an idle frame.
- **An idle frame still runs the whole decode ladder.** The SSD thresholds
  (confidence 0.2, NMS 0.45, top_k 100) match `detection_output_param` in
  `detect.prototxt` exactly, so the false-positive candidates are upstream
  behaviour. Removing them means changing detection semantics — decoding at a
  lower rate, skipping FastWindow, or capping the candidate count — and each
  costs detection rate.

## Inference backends

The `tract` feature runs the same models in pure Rust, with no C dependency.
It costs speed: on a 960x1280 frame with a small code in it, measured with the
`bench` example on an M-series Mac,

| | tflite | tract |
|---|---|---|
| SSD forward, 266x355 input | 1.8 ms | 13.5 ms |
| Whole frame, detect + super res | 3.3 ms | 20.8 ms |

The gap is the forward pass alone — everything around it is shared code, and the
results are identical. tflite's kernels are XNNPACK's, threaded and hand tuned
for exactly the depthwise convolutions this detector is made of; tract runs
single-threaded. So tract is the choice when a portable build matters more than
latency: no library to ship per platform, no linker configuration, `cargo build`
anywhere.
