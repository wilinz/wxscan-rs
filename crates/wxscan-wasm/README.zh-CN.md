# wxscan-wasm

[English](README.md) · **简体中文**

把 [wxscan](../wxscan) 的 C ABI 编成一个 WebAssembly 模块，给浏览器用。产物是一个
`.wasm` 文件；本 crate 不发布。

```sh
# 小体积构建：由宿主来跑推理（见下）
cargo build -p wxscan-wasm --target wasm32-unknown-unknown --profile wasm

# 自包含构建：由 tract 在模块内部跑 ONNX 权重
cargo build -p wxscan-wasm --target wasm32-unknown-unknown --profile wasm \
  --no-default-features --features tract
```

两种构建都值得加上 `RUSTFLAGS="-C target-feature=+simd128"`：实测它在正确性上没有
任何代价，却把扫描时间砍掉了约 28%。

## 两个后端，各自的代价

在一张 1920x1080、里面有一个二维码的场景图上测量，Node 22，M 系列 Mac，开启 simd128：

| | 模块 | Gzip 后 | 扫描 | 需要 |
|---|---|---|---|---|
| host（默认） | 433 KB | 221 KB | 332 ms | 宿主侧的一个引擎：[tools/tflite-wasm](../../tools/tflite-wasm)，另外 3.0 MB |
| `tract` | 12.5 MB | 2.9 MB | 347 ms | 无 |
| 完全不带模型 | 242 KB | 155 KB | 20 ms | 无，但能找到的符号少得多 |

推理只占那 332 ms 里较小的一半：在 host 后端背后用 TFLite 加 XNNPACK 时它是其中的
8 ms，其余是解码器——和原生构建里的比例相同，那边整帧 135 ms、检测器占 3.7 ms。
浏览器在这里大约比原生慢两倍半，而这个差距和 wasm 边界没有任何关系。

那 12 MB 是一个 ONNX 运行时，不是这个算法：扫描器、解码器和 imgproc 函数加在一起
就是 242 KB 那一行。

## 模块对宿主的要求

每种构建都导出 `malloc` 和 `free`——因为一个 wasm 模块没有别的办法拿到一张图片——
以及它的 `memory`。其余的导出就是
[`include/wxscan.h`](../wxscan-ffi/include/wxscan.h) 里的 C ABI，原样不变。

每种构建都会在模块 `wxscan` 下**导入** `wxscan_host_now_us`，默认构建还会在同一个
模块下多导入两个。它们都不是可选的：缺了模块就实例化不了。

| 导入 | |
|---|---|
| `wxscan_host_forward(net, input, len, shape, rank) -> bytes` | 用网络 `net`（0 是检测器，1 是超分辨率）在一个 NCHW 的 f32 输入上做前向，返回它准备好的结果的字节数，或者 0 |
| `wxscan_host_fetch(dst, len) -> ok` | 把那个结果写进模块的内存 |
| `wxscan_host_now_us() -> f64` | 从任意固定原点算起的微秒数；只有差值会被读取。浏览器回答 `performance.now() * 1000` |

那个时钟不是锦上添花。`std::time::Instant::now()` 在 `wasm32-unknown-unknown` 上会
panic，所以没有宿主时钟的话，`wxscan_wasm_take_stages` 背后的各阶段计时器全都报零，
而你在浏览器里就没有任何办法知道一帧的时间花在哪了。没什么可提供的宿主返回一个常数
即可，得到的也正是这个结果。

`wxscan_wasm_take_stages(out, len)` 最多写出 `len` 个（共十一个）`u32` 微秒计数——
`wxscan_wasm_stage_count()` 会说一共有几个——并把它们清零：检测器的准备、前向、
先验框与 NMS 解码，然后是它的输入宽高；接着是超分辨率、zxing、解码尝试次数，以及
最后一个候选的宽和高。检测器的前向计数**包含**花在 `wxscan_host_forward` 里的时间，
所以它和宿主在那里量到的是重叠的，而不是相加的。

分成两次调用而不是一次，是为了让内存全部由模块来分配和释放；宿主从不持有需要由模块
去释放的内存。写回来的那块数据是小端 32 位字：先是输出的个数，然后对每个输出是它的
rank 及其各维度，最后是所有 f32 数据一个输出接一个输出地排列。
[`host_net.rs`](../wxscan-ffi/src/host_net.rs) 是这份契约的另一半，也是它可读的
版本。

这个后端的扫描器来自 `wxscan_scanner_new_host(has_detector, has_sr)` 而不是
`wxscan_scanner_new`：权重留在宿主那边，模块只需要知道它可以请求哪几个网络。

**这些导入是同步的，而 JavaScript 的推理 API 不是。** 这就排除了 LiteRT.js——它的
`run` 在每个后端上都返回 promise；也排除了用 Asyncify 去弥合这个落差——实测那会让
模块大 37%，运行时间变成 2.3 倍。行得通的宿主是第二个 wasm 模块：
[`tools/tflite-wasm`](../../tools/tflite-wasm) 为浏览器构建 TensorFlow Lite C
运行时，它的 API 是同步的，吃的是和原生构建一样的 `.tflite` 权重，并且能 resize 到
一帧产生的任何形状。

`--features debug-log` 会多加一个导入 `wxscan_host_log`，以及一个导出的
`wxscan_install_panic_hook`。不加它，Rust 的 panic 到达宿主时是
`RuntimeError: unreachable`，消息被丢掉了——因为模块无处可打印；加上它，消息就能送到。
仅供开发使用。
