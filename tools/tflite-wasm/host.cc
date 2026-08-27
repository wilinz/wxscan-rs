// The host side of wxscan's host_net protocol: two interpreters, and enough
// accessors for the JS glue to move tensors between the two wasm modules.
#include <cstdio>
#include <cstring>
#include <emscripten.h>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace {

struct Net {
  TfLiteModel* model = nullptr;
  TfLiteInterpreter* interp = nullptr;
  TfLiteDelegate* xnn = nullptr;
};

Net nets[2];  // 0 = detector, 1 = super resolution

bool build(Net& n, const char* data, int len) {
  n.model = TfLiteModelCreate(data, len);
  if (!n.model) return false;
  TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(opts, 1);
  // The C API applies no delegate on its own, so XNNPACK is asked for here.
  TfLiteXNNPackDelegateOptions xo = TfLiteXNNPackDelegateOptionsDefault();
  xo.num_threads = 1;
  n.xnn = TfLiteXNNPackDelegateCreate(&xo);
  if (n.xnn) TfLiteInterpreterOptionsAddDelegate(opts, n.xnn);
  n.interp = TfLiteInterpreterCreate(n.model, opts);
  TfLiteInterpreterOptionsDelete(opts);
  return n.interp != nullptr;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int tf_load(const char* det, int det_len, const char* sr, int sr_len) {
  if (!build(nets[0], det, det_len)) return 0;
  if (!build(nets[1], sr, sr_len)) return 0;
  return 1;
}

/// Resize network `net` to an h x w single-channel input and return the
/// address its data should be written to, or 0.
EMSCRIPTEN_KEEPALIVE
int tf_prepare(int net, int h, int w) {
  TfLiteInterpreter* it = nets[net].interp;
  int dims[4] = {1, h, w, 1};
  if (TfLiteInterpreterResizeInputTensor(it, 0, dims, 4) != kTfLiteOk) return 0;
  if (TfLiteInterpreterAllocateTensors(it) != kTfLiteOk) return 0;
  return (int)(intptr_t)TfLiteTensorData(TfLiteInterpreterGetInputTensor(it, 0));
}

EMSCRIPTEN_KEEPALIVE
int tf_invoke(int net) {
  TfLiteInterpreter* it = nets[net].interp;
  if (TfLiteInterpreterInvoke(it) != kTfLiteOk) return -1;
  return TfLiteInterpreterGetOutputTensorCount(it);
}

EMSCRIPTEN_KEEPALIVE
int tf_out_ptr(int net, int i) {
  return (int)(intptr_t)TfLiteTensorData(TfLiteInterpreterGetOutputTensor(nets[net].interp, i));
}

EMSCRIPTEN_KEEPALIVE
int tf_out_floats(int net, int i) {
  return (int)(TfLiteTensorByteSize(TfLiteInterpreterGetOutputTensor(nets[net].interp, i)) / 4);
}

EMSCRIPTEN_KEEPALIVE
int tf_out_rank(int net, int i) {
  return TfLiteTensorNumDims(TfLiteInterpreterGetOutputTensor(nets[net].interp, i));
}

EMSCRIPTEN_KEEPALIVE
int tf_out_dim(int net, int i, int d) {
  return TfLiteTensorDim(TfLiteInterpreterGetOutputTensor(nets[net].interp, i), d);
}

}  // extern "C"
