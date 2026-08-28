# wxscan-rs

[English](README.md) · **简体中文**

OpenCV contrib 里的 `wechat_qrcode` 算法，用 Rust 重写了一遍。先用一个 CNN 在画面里
找出码的位置，再用第二个网络把每一块放大，最后交给 ZXing 的一个分支去解。不依赖
OpenCV，没有 C++，用 `tract` 后端时连 C 都没有。

<img src="https://raw.githubusercontent.com/wilinz/wxscan/main/docs/demo.webp" width="300"
     alt="一帧里两个二维码都被框出，点开其中一个显示解出的中文文本，按 UTF-8 读取。">

*这套算法干活的样子，由 [wxscan](https://github.com/wilinz/wxscan) 里的 Flutter
包驱动：一帧里两个码，其中一个是转过的，隔着一张桌子从笔记本屏幕上读到。*

解码是简单的那一半。难的是在 1080p 的一帧里找到那个又小又远、光照还差的码，那两级 CNN
就是干这个的。微信的扫描器能隔着半个房间扫到码，普通解码器却要你把码怼到镜头前，差别
就在这里。

**它跟原实现对过账，不只是有测试。** 在一个固定语料上：不带模型时，160 张图里有 159
张的解码文本和 OpenCV 的 C++ 实现一致；带模型时 24 张场景图全对。角点坐标除两张外逐位
相同，那两张差在亚像素级。为此少掉的自由度是有意的。这个移植逐行跟着[上游源码][upstream]
走，包括那些看起来不对的地方。移植要是悄悄改进了原实现，就没法再拿去跟它比对了。
见 [一致性](#一致性)。

**cargo 能编的地方它都能编。** 默认后端要 TFLite 的 C 库；`tract` 后端除了 Rust 什么
都不要，交叉编译就是一句 `cargo build --target`。两者跑的是同一批权重。

[upstream]: https://github.com/opencv/opencv_contrib/tree/4.x/modules/wechat_qrcode

## 使用

还没发到 crates.io，所以依赖从 git 引：

```toml
[dependencies]
wxscan = { git = "https://github.com/wilinz/wxscan-rs" }
```

发布之后：

```sh
cargo add wxscan
```

git 这种写法跟着默认分支走，加 `tag`、`branch` 或 `rev` 可以钉死一个。

**需要什么**

| | 版本 |
|---|---|
| Rust | 依赖这些 crate 要 1.75 或更新；直接 checkout 构建用的是 `rust-toolchain.toml` 钉死的 1.95.0，rustup 第一次跑时自动装 |
| libtensorflowlite_c | 只有默认的 `tflite` 后端要，见 [TFLite 库](#tflite-库)。`tract` 后端除了 Rust 什么都不要 |

不内置任何二进制，构建脚本也不联网。

`cvlite` 和 `wxing` 同样还没发布，而本 crate 是把它们当普通依赖写的，所以构建时得告诉
cargo 去哪里找：

```toml
[patch.crates-io]
cvlite = { git = "https://github.com/wilinz/cvlite" }
wxing = { git = "https://github.com/wilinz/wxing" }
```

权重不在任何 crate 里。去
[wxscan-weights](https://github.com/wilinz/wxscan-weights) 下 `detect.tflite` 和
`sr.tflite`，或者用你自己的，然后按字节读进来：

```rust
use wxscan::WeChatQRCode;

let detect_bytes = std::fs::read("detect.tflite")?;
let sr_bytes = std::fs::read("sr.tflite")?;

// 两个模型都可以传 None，那就是不带 CNN 阶段的纯解码模式。
let detect = wxscan::tflite::TfliteNet::from_bytes(&detect_bytes)?;
let sr = wxscan::tflite::TfliteNet::from_bytes(&sr_bytes)?;
let scanner = WeChatQRCode::new(Some(detect), Some(sr));

for result in scanner.detect_and_decode_gray(&gray, width, height) {
    // 负载是原始字节；怎么解释它看 `charset`。
    println!("{} ({})", result.text_lossy(), result.charset);
}
```

`gray` 是 8 位灰度，每像素一字节，一行接一行。

## Crate 怎么分的

只按一条界线分：这块东西是不是这个算法专有的。不专有的那两块各自放在自己的仓库里，
因为离开这里它们照样有用：

| Crate | 仓库 | 内容 |
|---|---|---|
| [`cvlite`](https://github.com/wilinz/cvlite) | 独立 | 这里用到的那些 OpenCV `imgproc` 函数：resize、自适应阈值、颜色转换、blob。跟二维码无关；aarch64 上有 NEON 路径。 |
| [`wxing`](https://github.com/wilinz/wxing) | 独立 | 微信用的那个 ZXing 分支：二值化器、定位图形、解码器。跟 CNN 阶段无关，所以它自己就能解码。 |

下面这些是 WeChat 算法专有的，三个一起走同一个版本。tflite 绑定是检测器跑在上面的
后端，C ABI 是它对外的那张脸：

| Crate | 内容 |
|---|---|
| [`wxscan`](crates/wxscan) | CNN 检测、超分辨率，以及围绕它们的编排。完整的算法就是这个。 |
| [`wxscan-tflite`](crates/wxscan-tflite) | tflite 绑定，默认的推理后端。单独拆出来是为了把唯一那个 C 依赖圈在里面。 |
| [`wxscan-ffi`](crates/wxscan-ffi) | C ABI，给 Rust 之外的调用方。`include/wxscan.h` 由 cbindgen 生成。 |

## 推理后端

CNN 推理藏在 `net::Net` trait 后面，算法的哪一部分都不知道是哪个库在跑它。仓库自带
两个后端，都在 `wxscan::backend` 里：

| Feature | 引擎 | 权重 | C 依赖 |
|---|---|---|---|
| `tflite`（默认） | `wxscan-tflite` | `detect.tflite`、`sr.tflite` | libtensorflowlite_c |
| `tract` | [tract](https://crates.io/crates/tract-onnx) | `detect.onnx`、`sr.onnx` | 无 |

布局转换也放在 tflite 适配器里：tflite 用 NHWC，trait 约定的是 NCHW。tract 那边不用
转，ONNX 本来就是 NCHW，跟两种格式共同的来源 Caffe 模型一致。

```toml
wxscan = { git = "https://github.com/wilinz/wxscan-rs", default-features = false, features = ["tract"] }
```

发布之后：

```sh
cargo add wxscan --no-default-features --features tract
```

tract 在 cargo 能到的任何地方都能编能跑，代价是比 tflite 的 XNNPACK 内核慢一些。两个
feature 都关掉，剩下的核心完全没有推理能力，但普通的 `cargo build` 照样编得过、测得过。

不管开哪些 feature，在 `wasm32-unknown-unknown` 上都有一件事必须做：宿主得在模块
`wxscan` 下提供一个 `wxscan_host_now_us() -> f64` 导入。`std::time::Instant::now()`
在那个目标上会 panic，所以各阶段的计时器改成读宿主的时钟，浏览器回一个
`performance.now() * 1000` 就行。没有时钟可借的宿主返回常数即可，那样每个阶段报出来的
耗时都是零。

实现一个后端就是实现一个方法。trait 在 `wxscan` 里，所以仓库之外的 crate 也可以给自己
的类型实现：

```rust
use wxscan::net::{Net, NetOutput};

struct MyNet(/* CoreML、NNAPI 或者别的什么引擎 */);

impl Net for MyNet {
    fn forward(&self, input: &[f32], shape: &[usize]) -> Result<Vec<NetOutput>, String> {
        // 输入是 NCHW；返回也用 NCHW
    }
}
```

没有模型时，这条流水线**降级**成一个普通解码器，不是失效。普通的码它照样读得出来，
丢掉的是小码和远处的码的检出率，也就是 CNN 阶段贡献的那部分。

## TFLite 库

仓库不内置任何二进制。把 `TFLITE_LIB_DIR` 指向一个装着 libtensorflowlite_c 的目录，
或者把这些符号留给最终的链接步骤去解析——Apple 平台通常这么做。库的名字随分发渠道
变：C API 的桌面构建叫 `libtensorflowlite_c`，Google 面向 Android 的 LiteRT 分发把
同一套 API 叫 `libLiteRt`。

```sh
TFLITE_LIB_DIR=/path/to/libs cargo test --workspace
```

## 一致性

`tools/parity` 把同一批图片分别喂给 OpenCV 的 `wechat_qrcode` 和这个移植，比解码文本，
也比角点坐标。当前结果在
[`tools/parity/README.zh-CN.md`](tools/parity/README.zh-CN.md)：不带模型时文本 160
张对上 159 张，带模型时场景图 24/24 全对，角点坐标除两张外逐位相同，那两张差在亚像素级。

剩下的差异都能追到 `cv::adaptiveThreshold` 上：OpenCV 对 8U 图像走定点的可分离滤波，
这个移植在 f32 上累加。

[wxscan-weights](https://github.com/wilinz/wxscan-weights) 仓库存着预构建的权重，也存
着从公开的 Caffe 模型重新生成它们的脚本。

## 性能

[`docs/performance.md`](docs/performance.md) 记了优化过什么、怎么测的，还有试过又回退
掉的东西。那里每一项改动都验证过：一致性语料上的输出逐字节不变。

## 绑定

C ABI 的使用方是 [`wxscan`](https://github.com/wilinz/wxscan) 里的 Flutter 包，那也是
「怎么从一个平台绑定里驱动它」的参考实现。

## 许可

Apache-2.0，跟它移植自的上游实现一致。
