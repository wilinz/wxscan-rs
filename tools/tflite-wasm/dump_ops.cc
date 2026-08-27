// Prints the operators a .tflite model contains, for host.cc's registration
// list. Built by `./build.sh --ops`; see the note at the top of host.cc.
#include <cstdio>
#include <memory>
#include <emscripten.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"

extern "C" EMSCRIPTEN_KEEPALIVE
int dump_ops(const char* data, int len) {
  auto model = tflite::FlatBufferModel::BuildFromBuffer(data, len);
  if (!model) {
    printf("not a model\n");
    return 0;
  }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder(*model, resolver)(&interp);
  if (!interp) {
    printf("no interpreter\n");
    return 0;
  }
  int seen[256] = {0};
  for (size_t i = 0; i < interp->nodes_size(); i++) {
    const int code = interp->node_and_registration((int)i)->second.builtin_code;
    if (code >= 0 && code < 256) seen[code]++;
  }
  printf("%zu nodes, operators (code x count, against builtin_ops.h):",
         interp->nodes_size());
  for (int c = 0; c < 256; c++) {
    if (seen[c]) printf(" %d(x%d)", c, seen[c]);
  }
  printf("\n");
  return 1;
}
