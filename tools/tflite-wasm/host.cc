// The host side of wxscan's host_net protocol: two interpreters, and enough
// accessors for the JS glue to move tensors between the two wasm modules.
//
// It uses the C++ API rather than the C one, and registers operators by hand,
// for size. `TfLiteInterpreterCreate` resolves operators with
// `BuiltinOpResolver`, which registers about 150 of them and so links every
// kernel TensorFlow Lite has; these two models between them use sixteen.
// Dropping the C API takes the module from 2.9 MB to 1.3 MB with no change in
// what it computes or how long it takes.
//
// The list below therefore follows the weights. If a model is replaced by one
// using an operator that is not here, it fails at AllocateTensors rather than
// silently doing something else. To regenerate it, build `dump_ops.cc`
// (`./build.sh --ops`) and run it over the new weights; it prints the operator
// codes each one contains.

#include <cstdio>
#include <cstring>
#include <memory>
#include <emscripten.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace {
namespace ops = tflite::ops::builtin;

struct Net {
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interp;
  TfLiteDelegate* xnn = nullptr;
};
Net nets[2];

// The union of what detect.tflite and sr.tflite contain, and nothing else:
// the resolver is what drags every other kernel into the binary.
void registerOps(tflite::MutableOpResolver& r) {
  auto add = [&](tflite::BuiltinOperator op, TfLiteRegistration* reg) {
    r.AddBuiltin(op, reg, 1, 10);
  };
  add(tflite::BuiltinOperator_ADD, ops::Register_ADD());
  add(tflite::BuiltinOperator_CONCATENATION, ops::Register_CONCATENATION());
  add(tflite::BuiltinOperator_CONV_2D, ops::Register_CONV_2D());
  add(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, ops::Register_DEPTHWISE_CONV_2D());
  add(tflite::BuiltinOperator_MAX_POOL_2D, ops::Register_MAX_POOL_2D());
  add(tflite::BuiltinOperator_MUL, ops::Register_MUL());
  add(tflite::BuiltinOperator_RESHAPE, ops::Register_RESHAPE());
  add(tflite::BuiltinOperator_SOFTMAX, ops::Register_SOFTMAX());
  add(tflite::BuiltinOperator_PAD, ops::Register_PAD());
  add(tflite::BuiltinOperator_PADV2, ops::Register_PADV2());
  add(tflite::BuiltinOperator_SUB, ops::Register_SUB());
  add(tflite::BuiltinOperator_STRIDED_SLICE, ops::Register_STRIDED_SLICE());
  add(tflite::BuiltinOperator_TRANSPOSE_CONV, ops::Register_TRANSPOSE_CONV());
  add(tflite::BuiltinOperator_SHAPE, ops::Register_SHAPE());
  add(tflite::BuiltinOperator_PACK, ops::Register_PACK());
  add(tflite::BuiltinOperator_LEAKY_RELU, ops::Register_LEAKY_RELU());
}

bool build(Net& n, const char* data, int len) {
  n.model = tflite::FlatBufferModel::BuildFromBuffer(data, len);
  if (!n.model) return false;
  tflite::MutableOpResolver resolver;
  registerOps(resolver);
  tflite::InterpreterBuilder builder(*n.model, resolver);
  builder.SetNumThreads(1);
  if (builder(&n.interp) != kTfLiteOk || !n.interp) return false;

  // Only BuiltinOpResolver carries the delegate creator the builder looks for,
  // and registering every operator is what this file exists to avoid. So the
  // delegate is asked for by name instead — without it the operators fall back
  // to the reference kernels, which cost about four times the time.
  TfLiteXNNPackDelegateOptions o = TfLiteXNNPackDelegateOptionsDefault();
  o.num_threads = 1;
  n.xnn = TfLiteXNNPackDelegateCreate(&o);
  if (!n.xnn || n.interp->ModifyGraphWithDelegate(n.xnn) != kTfLiteOk) {
    printf("wxscan: XNNPACK did not attach; falling back to reference kernels\n");
  }
  return true;
}
}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
int tf_load(const char* det, int det_len, const char* sr, int sr_len) {
  return build(nets[0], det, det_len) && build(nets[1], sr, sr_len) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
int tf_prepare(int net, int h, int w) {
  auto& it = *nets[net].interp;
  if (it.ResizeInputTensor(it.inputs()[0], {1, h, w, 1}) != kTfLiteOk) return 0;
  if (it.AllocateTensors() != kTfLiteOk) return 0;
  return (int)(intptr_t)it.typed_input_tensor<float>(0);
}

EMSCRIPTEN_KEEPALIVE
int tf_invoke(int net) {
  auto& it = *nets[net].interp;
  if (it.Invoke() != kTfLiteOk) return -1;
  return (int)it.outputs().size();
}

EMSCRIPTEN_KEEPALIVE
int tf_out_ptr(int net, int i) {
  auto& it = *nets[net].interp;
  return (int)(intptr_t)it.tensor(it.outputs()[i])->data.f;
}

EMSCRIPTEN_KEEPALIVE
int tf_out_floats(int net, int i) {
  auto& it = *nets[net].interp;
  return (int)(it.tensor(it.outputs()[i])->bytes / 4);
}

EMSCRIPTEN_KEEPALIVE
int tf_out_rank(int net, int i) {
  auto& it = *nets[net].interp;
  return it.tensor(it.outputs()[i])->dims->size;
}

EMSCRIPTEN_KEEPALIVE
int tf_out_dim(int net, int i, int d) {
  auto& it = *nets[net].interp;
  return it.tensor(it.outputs()[i])->dims->data[d];
}

}  // extern "C"
