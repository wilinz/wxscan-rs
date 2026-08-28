# 给浏览器用的 TFLite

[English](README.md) · **简体中文**

`build.sh` 产出 `wxscan_tflite.js` 和 `wxscan_tflite.wasm`：带 XNNPACK delegate 的
TensorFlow Lite C 运行时，编译成 WebAssembly，外加 [`host.cc`](host.cc) 里那层小小的
垫片。跟它对话的是 [`wxscan-ffi` 的 host 后端](../../crates/wxscan-ffi/src/host_net.rs)。

版本号写在仓库顶层的 [`depversion.toml`](../../depversion.toml) 里，`build.sh` 和 CI
工作流都读它。只有一处，因为它曾经有两处，而没有任何东西保证那两处一致；放在根目录，
因为这是关于这个库的事实，不是关于这次构建的。它必须和 wxscan 的 `tool/tflite.lock`
里钉死的桌面构建对上，这样浏览器跑的才是和其它每个平台相同的运行时、相同的 `.tflite`
权重。

它旁边还有一个 `patch`，那是我们自己的，不是 TensorFlow 的：下面那些补丁、以及施加它们
的脚本的修订号。上游不构建这个配置，所以从某个给定版本里出来的东西，由它们决定的程度
不亚于由版本号决定；而且版本号不动的时候，它们会动。它们动了，就把它抬上去。

改动其中任何一个，就是一次版本升级的全部内容。打一个 `tflite-<version>-p<patch>` 的
tag 会构建它，并作为那个 tag 的 release 发布；tag 叫别的名字则直接失败，不会把一堆字节
放在一个描述不了它们的名字底下。

它有自己的 release，不搭扫描器的顺风车，因为它是一个按自己节奏变化的依赖：它一年变几次，
扫描器每天都在变，把一份拷贝塞进每一个 `v*` release 就是同样的 1.3 MB 一遍遍重复。
下游分别钉死这两者，wxscan 的 `tool/web.lock` 里各占一行。

```sh
source /path/to/emsdk/emsdk_env.sh
./build.sh                       # 写到 ./out
```

第一次要花上一刻钟：它会克隆 TensorFlow、抓十几个依赖，编译大约一千个 XNNPACK 微内核。
在一棵全新的树上，**第一次 `cmake` 预期就是会失败**，原因见下面的补丁说明，脚本是有意
越过它继续往下走的。

    ./build.sh --ops        # 改为构建 dump_ops，权重变化时用

## 为什么需要打补丁

上游不构建这个配置，所以 `build.sh` 在抓取之后施加两个补丁。两个都在
[`patches/`](patches) 里。

| | |
|---|---|
| `0001-tensorflow-std-abs` | 从 libc++ 19 起 `std::abs<float>` 不再是模板。TF 2.17 早于那之前，emscripten 工具链不是。两行代码改成 lambda。 |
| `0002-xnnpack-wasm` | XNNPACK 的 CMake 构建不支持 wasm：它生成了 wasm 微内核清单却从不把它们接上，拒绝把 `Emscripten` 当作系统名，而且 `src/xnnpack/math.h` 调用 `rint` 却没有 include `math.h`。它的 wasm 构建走的是 Bazel。 |

第二个补丁就是脚本要配置两遍的原因。XNNPACK 是在一次 configure 过程中被拉下来的，那时
它还没打补丁、会拒绝这个目标，所以第一遍在它把补丁需要的东西抓下来之后，被允许失败一次。
这个补丁还是每次运行都施加，不是只施加一次：FetchContent 每次都重新 checkout 那个依赖，
把本地改动一并带走。

依赖构建在这个目标上还有另外两处做得不对，那两处在脚本里处理，没有打补丁，因为它们各自
只是单个目标文件：flatbuffers 按翻译单元决定是否区域无关，结果没有定义一个 TFLite 引用
到的符号；cpuinfo 则从不编译它的 emscripten 后端。

## 代价，以及为什么值得

测的是一张 1920x1080 的场景图，一个二维码，Node 22，M 系列 Mac：

| 检测器输入 | 参考内核 | 加上 XNNPACK + SIMD |
|---|---|---|
| 224x320 | 10.2 ms | 2.0 ms |
| 384x384 | 20.8 ms | 4.1 ms |
| 480x640 | 43.6 ms | 8.4 ms |

XNNPACK 接管了检测器 139 个节点里的 137 个，剩下的执行计划只有三个节点。

**这里没有任何东西会自动施加这个 delegate。** C API 从头到尾就没提过 XNNPACK，
`c_api.cc` 里根本没有它的踪影；而 C++ 的 builder 只通过 `BuiltinOpResolver` 施加它，
那恰恰是 `host.cc` 刻意避开的东西。所以它是按名字创建、再挂上去的。漏了这一步，模块
照样能用、照样解得对，只是悄悄跑在参考内核上。唯一的外部迹象是少了那行
`Created TensorFlow Lite XNNPACK delegate`——推理在一帧里占比小到足以把四倍于自身的
开销藏起来。

### 体积

| | 模块 | Gzip 后 |
|---|---|---|
| 走 C API，注册全部算子 | 2.90 MB | 838 KB |
| 只注册这两个模型用到的十六个 | **1.34 MB** | **418 KB** |

`-Oz`、LTO 和一个更小的分配器加起来大约值 70 KB。那些静态库在链接看到它们之前就已经按
`-O3` 编译好了，所以体积取决于链接进来了什么，不取决于怎么链接。`BuiltinOpResolver`
会注册大约 150 个算子，注册一个就会把它的内核链接进来。细节见 `host.cc` 顶部的注释，
权重变化时用 `--ops` 重新生成清单。

## 另一条路，以及为什么不走

`wxscan-wasm` 也可以把 tract 编进模块，完全不要宿主就跑 ONNX 权重。那个模块是 11.9 MB，
这条路是 3.0 MB；同一帧那边要 347 ms，这边 332 ms。它不需要 JavaScript，这是它唯一真正
的优势。

LiteRT.js 在这里用不了：它拒绝带符号维度的模型，也不提供 resize，而 `detect.tflite`
是**有意**把高和宽保持为符号的。TFLite 的 C API 有
`TfLiteInterpreterResizeInputTensor`，这条路走得通，根本原因就在这里。
