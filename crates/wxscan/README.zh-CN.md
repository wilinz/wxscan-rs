# wxscan

[English](README.md) · **简体中文**

OpenCV contrib 里 `wechat_qrcode` 算法的 Rust 移植：基于 CNN 的检测、超分辨率，
以及解码。不依赖 OpenCV。

```rust
use wxscan::WeChatQRCode;

let detect = wxscan::tflite::TfliteNet::from_bytes(detect_bytes)?;
let sr = wxscan::tflite::TfliteNet::from_bytes(sr_bytes)?;
let scanner = WeChatQRCode::new(Some(detect), Some(sr));

for result in scanner.detect_and_decode_gray(&gray, width, height) {
    // 负载是原始字节；`charset` 说明该怎么解释它。
    println!("{} ({})", result.text_lossy(), result.charset);
}
```

`detect_and_decode_gray_with_candidates` 会额外返回检测器找到的东西。有候选但没有
结果意味着定位到了一个符号却解不出来，通常是因为它太小或者太糊——调用方可以据此提示
放大，而不是报告一次失败。

## 它由什么组成

那些并非本算法专有的部分是独立的 crate：
[`cvlite`](https://github.com/wilinz/cvlite) 提供用到的 OpenCV 函数，
[`wxing`](https://github.com/wilinz/wxing) 提供解码器所出自的 ZXing 分支。本 crate
装的是专有的那些：SSD 检测器、超分辨率阶段，以及围绕它们的编排。

## 模型

权重不属于本 crate，也不属于任何其它 crate。从
[wxscan-weights](https://github.com/wilinz/wxscan-weights) 取预构建的那些——它们按
格式分组，跟随所用的后端——或者传入你自己的缓冲：

```rust
let detect = std::fs::read("detect.tflite")?;
let sr = std::fs::read("sr.tflite")?;
```

没有模型时，这条流水线**降级**成一个普通解码器，而不是失效。它照样能读普通的码；
失去的是小码和远处的码的检出率。

## 推理后端

CNN 推理藏在 `net::Net` trait 后面，算法的任何部分都不知道是哪个库在跑它。随仓库
提供两个后端，都在 `backend` 模块里：

| Feature | 引擎 | 权重 | C 依赖 |
|---|---|---|---|
| `tflite`（默认） | [`wxscan-tflite`](../wxscan-tflite) | `detect.tflite`、`sr.tflite` | libtensorflowlite_c |
| `tract` | [tract](https://crates.io/crates/tract-onnx) | `detect.onnx`、`sr.onnx` | 无 |

布局转换也放在 tflite 适配器里：tflite 是 NHWC，而 trait 约定的是 NCHW。tract 不需要
转换——ONNX 就是 NCHW，和两种格式共同的来源 Caffe 模型一致——而且 cargo 能到的地方
它都能构建：

```toml
wxscan = { git = "https://github.com/wilinz/wxscan-rs", default-features = false, features = ["tract"] }
```

不论开哪些 feature，在 `wasm32-unknown-unknown` 上都有一件事必须做：宿主必须在模块
`wxscan` 下提供一个 `wxscan_host_now_us() -> f64` 导入。`std::time::Instant::now()`
在那个目标上会 panic，所以各阶段的计时器改读宿主的时钟——浏览器回答
`performance.now() * 1000` 即可。没有时钟可借的宿主返回一个常数即可，那样每个阶段
报告的耗时都是零。

两个都关掉，剩下的核心没有推理、也没有 C 依赖，用普通的 `cargo build` 就能编译：

```toml
wxscan = { git = "https://github.com/wilinz/wxscan-rs", default-features = false }
```

一个后端就是一个方法。这个 trait 就在这里，所以你自己的 crate 可以为自己的类型
实现它：

```rust
use wxscan::net::{Net, NetOutput};

impl Net for MyNet {
    fn forward(&self, input: &[f32], shape: &[usize]) -> Result<Vec<NetOutput>, String> {
        // 输入是 NCHW；返回也用 NCHW
    }
}
```

即便开着默认 feature，那个库本身仍然不是内置的：把 `TFLITE_LIB_DIR` 指向一个含有它
的目录，或者让最终的链接步骤去解析这些符号——Apple 平台通常就是这么做的。

## Features

| Feature | 默认 | 作用 |
|---|---|---|
| `tflite` | 是 | `net::Net` 的 libtensorflowlite_c 实现。 |
| `profiling` | 否 | 热点路径上的埋点，供 `examples/profile` 使用。 |

[wxscan-rs](https://github.com/wilinz/wxscan-rs) 的一部分。Apache-2.0。
