# Model conversion

The upstream models are Caffe. This directory rebuilds them, layer by layer,
in the two formats the Rust backends run: TFLite for `wxscan-tflite`, ONNX for
tract. Neither path retrains or alters a weight; both are checked against the
Caffe output before being copied into `crates/wxscan-models/models/`.

There is no generic Caffe importer here, only the layers these two models use.
That is a deliberate limit: it keeps each converter short enough to read against
the prototxt, which is what makes the "did the conversion change anything"
question answerable.

| File | Role |
|---|---|
| `download_models.sh` | Fetches the four Caffe files at the commit `opencv_contrib` pins, and checks the MD5 |
| `ref_dump.py` | Reference output from OpenCV 4.x's Caffe importer, saved as `ref_outputs.npz` |
| `caffe2tf.py` / `export_tflite.py` | The TFLite path: rebuild as Keras, export, compare |
| `caffe2onnx.py` / `export_onnx.py` | The ONNX path: build the graph directly, export, compare |
| `check_tf.py` | Intermediate-tensor diff against Caffe, on the TF side, for when a whole-model comparison fails |

## Running

The reference and the two conversions want incompatible pins (OpenCV 5 dropped
the Caffe importer), so keep them in separate virtualenvs:

```sh
./download_models.sh

# reference output, needs OpenCV 4.x
pip install -r requirements-ref.txt
python ref_dump.py

# ONNX, pure Python and the lighter of the two
pip install grpcio-tools numpy onnx onnxruntime
python -m grpc_tools.protoc -I. --python_out=. caffe.proto   # generates caffe_pb2.py
python export_onnx.py            # writes onnx_out/{detect,sr}.onnx

# TFLite, needs TensorFlow
pip install -r requirements.txt
python export_tflite.py          # writes tflite_out/{detect,sr}.tflite
```

Both export scripts print the maximum absolute difference against the Caffe
reference; it should stay around 1e-6, the noise floor of float32 accumulating
in a different order. Anything larger means the graph is not equivalent, not
that it is imprecise.

## Why ONNX is the simpler of the two

ONNX is NCHW, like Caffe, and its Conv/ConvTranspose take Caffe's weight
layout, groups, dilation and symmetric padding as they are; MaxPool has a
`ceil_mode` for Caffe's window clipping. So `caffe2onnx.py` is close to a
transcription. The TFLite path has to move every tensor to NHWC, expand
dilation into sparse kernels, and pad explicitly because TF's `same` is
asymmetric at stride 2 — which is why `caffe2tf.py` is three times the length
for the same models.

Two things are left to the Rust side in both formats, as upstream's own
`cv::dnn` graph does them in layers this port implements itself: `PriorBox` and
`DetectionOutput`. The exported graphs end at `mbox_loc` and
`mbox_conf_flatten`.

## Shapes

Height and width stay symbolic. Super resolution runs on crops of whatever size
the pipeline produces, and the detector's input follows the frame. The ONNX
export deliberately skips `shape_inference`: recording every intermediate shape
in terms of the symbols H and W only gives a runtime that pins them to a
concrete size more facts to reconcile, and tract fails to unify them.
