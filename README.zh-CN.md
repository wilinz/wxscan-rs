# wxscan-rs

[English](README.md) · **简体中文**

OpenCV contrib 里的 `wechat_qrcode` 算法，用 Rust 写成：一个 CNN 在画面里定位符号，
第二个网络把每一块裁剪放大，再由一个 ZXing 的分支去解码。不依赖 OpenCV，没有 C++，
用 `tract` 后端时连 C 都没有。

解一个二维码是简单的那部分。在一帧 1080p 里找到一个小的、远的或者光照很差的码才是
那两级 CNN 存在的理由，也正是微信扫描器能隔着一个房间读到码，而普通解码器要你把码
怼到镜头前的原因。

**它是对着原实现校验过的，不只是有测试。** 在一个固定语料上，无模型时 160 张图里有
159 张的解码文本与 OpenCV 的 C++ 实现一致，有模型时 24 张场景图 24 张一致，角点坐标
除两张外逐位相同，那两张的差异在亚像素级别。为此付出的自由度是有意的：这个移植逐行
跟随[上游源码][upstream]，包括那些看起来不对的地方，因为一个悄悄改进了原实现的移植
就没法再和它比对了。见 [一致性](#一致性)。

**cargo 能构建的地方它都能构建。** 默认后端需要 TFLite C 库；`tract` 后端不需要
Rust 之外的任何东西，所以交叉编译就是一句 `cargo build --target`。两者跑的是同一批
权重。

[upstream]: https://github.com/opencv/opencv_contrib/tree/4.x/modules/wechat_qrcode

## 使用

还没发到 crates.io。两种写法都列在这里，等到发布那天切换只是改一行：

```toml
[dependencies]
wxscan = { git = "https://github.com/wilinz/wxscan-rs" }
# wxscan = "0.1"                    # 发布后从 crates.io 引入
```

git 那种写法跟随默认分支；加 `tag`、`branch` 或 `rev` 可以固定一个。

`cvlite` 和 `wxing` 同样没有发布，而本 crate 是把它们当普通依赖来写的，所以构建时
必须告诉 cargo 它们在哪：

```toml
[patch.crates-io]
cvlite = { git = "https://github.com/wilinz/cvlite" }
wxing = { git = "https://github.com/wilinz/wxing" }
```

权重不在任何 crate 里。从
[wxscan-weights](https://github.com/wilinz/wxscan-weights) 下载 `detect.tflite`
和 `sr.tflite`——或者用你自己的——然后以字节读入：

```rust
use wxscan::WeChatQRCode;

let detect_bytes = std::fs::read("detect.tflite")?;
let sr_bytes = std::fs::read("sr.tflite")?;

// 两个模型都可以是 None，那样就是在没有 CNN 阶段的模式下解码。
let detect = wxscan::tflite::TfliteNet::from_bytes(&detect_bytes)?;
let sr = wxscan::tflite::TfliteNet::from_bytes(&sr_bytes)?;
let scanner = WeChatQRCode::new(Some(detect), Some(sr));

for result in scanner.detect_and_decode_gray(&gray, width, height) {
    // 负载是原始字节；`charset` 说明该怎么解释它。
    println!("{} ({})", result.text_lossy(), result.charset);
}
```

`gray` 是 8 位灰度，每像素一字节，一行接一行。

## Crate 划分

划分只依据一条界线：一块东西是不是这个算法专有的。那两块不专有的放在各自的仓库里，
因为脱离这里它们也有用：

| Crate | 仓库 | 内容 |
|---|---|---|
| [`cvlite`](https://github.com/wilinz/cvlite) | 独立 | 这里用到的 OpenCV `imgproc` 函数：resize、自适应阈值、颜色转换、blob。与二维码无关；在 aarch64 上有 NEON 路径。 |
| [`wxing`](https://github.com/wilinz/wxing) | 独立 | 微信所用的 ZXing 分支：二值化器、定位图形、解码器。与 CNN 阶段无关，所以它自己就能解码。 |

下面这些都是 WeChat 算法专有的，三者作为一个版本一起走：tflite 绑定是检测器运行其上
的后端，而 C ABI 是它的对外表面。

| Crate | 内容 |
|---|---|
| [`wxscan`](crates/wxscan) | CNN 检测、超分辨率，以及围绕它们的编排。这就是完整的算法。 |
| [`wxscan-tflite`](crates/wxscan-tflite) | tflite 绑定，用作默认推理后端。单独拆出来，是为了把唯一的 C 依赖限制在它里面。 |
| [`wxscan-ffi`](crates/wxscan-ffi) | C ABI，给 Rust 之外的调用方。用 cbindgen 生成 `include/wxscan.h`。 |

## 推理后端

CNN 推理藏在 `net::Net` trait 后面，算法的任何部分都不知道是哪个库在跑它。随仓库
提供两个后端，都在 `wxscan::backend` 里：

| Feature | 引擎 | 权重 | C 依赖 |
|---|---|---|---|
| `tflite`（默认） | `wxscan-tflite` | `detect.tflite`、`sr.tflite` | libtensorflowlite_c |
| `tract` | [tract](https://crates.io/crates/tract-onnx) | `detect.onnx`、`sr.onnx` | 无 |

布局转换也放在 tflite 适配器里，因为 tflite 是 NHWC 而 trait 约定的是 NCHW。tract
不需要转换：ONNX 就是 NCHW，和两种格式共同的来源 Caffe 模型一致。

```toml
wxscan = { git = "https://github.com/wilinz/wxscan-rs", default-features = false, features = ["tract"] }
# wxscan = { version = "0.1", default-features = false, features = ["tract"] }   # 发布后
```

tract 在 cargo 能到的任何地方都能构建和运行，代价是相对 tflite 的 XNNPACK 内核慢一些。
把两个 feature 都关掉，剩下的核心完全没有推理能力，但它照样能用普通的 `cargo build`
编译和测试。

不论开哪些 feature，在 `wasm32-unknown-unknown` 上都有一件事必须做：宿主必须在模块
`wxscan` 下提供一个 `wxscan_host_now_us() -> f64` 导入。`std::time::Instant::now()`
在那个目标上会 panic，所以各阶段的计时器改读宿主的时钟——浏览器回答
`performance.now() * 1000` 即可。没有时钟可借的宿主返回一个常数即可，那样每个阶段
报告的耗时都是零。

实现一个后端就是实现一个方法。这个 trait 在 `wxscan` 里，所以仓库之外的 crate 也可以
为自己的类型实现它：

```rust
use wxscan::net::{Net, NetOutput};

struct MyNet(/* CoreML、NNAPI 或者别的什么引擎 */);

impl Net for MyNet {
    fn forward(&self, input: &[f32], shape: &[usize]) -> Result<Vec<NetOutput>, String> {
        // 输入是 NCHW；返回也用 NCHW
    }
}
```

没有模型时，这条流水线**降级**成一个普通解码器，而不是失效。它照样能读普通的码；
失去的是小码和远处的码的检出率——那正是 CNN 阶段贡献的东西。

## TFLite 库

本仓库不内置任何二进制。把 `TFLITE_LIB_DIR` 指向一个含有 libtensorflowlite_c 的
目录，或者让最终的链接步骤去解析这些符号——Apple 平台通常就是这么做的。库的名字随
分发渠道而不同：C API 的桌面构建叫 `libtensorflowlite_c`，而 Google 面向 Android 的
LiteRT 分发把同一套 API 叫作 `libLiteRt`。

```sh
TFLITE_LIB_DIR=/path/to/libs cargo test --workspace
```

## 一致性

`tools/parity` 把同一批图片分别喂给 OpenCV 的 `wechat_qrcode` 和这个移植，比较解码
文本和角点坐标。当前结果在
[`tools/parity/README.zh-CN.md`](tools/parity/README.zh-CN.md)：无模型时文本在 160 张里对上
159 张，有模型时场景图 24/24 全对，角点坐标除两张外逐位相同，那两张的差异在亚像素
级别。

剩下的差异可以追溯到 `cv::adaptiveThreshold`：OpenCV 对 8U 图像使用定点的可分离滤波，
而这个移植在 f32 上累加。

[wxscan-weights](https://github.com/wilinz/wxscan-weights) 仓库存放预构建的权重，以及
从公开的 Caffe 模型重新生成它们的脚本。

## 性能

[`docs/performance.md`](docs/performance.md) 记录了优化了什么、怎么测的，以及试过又
回退掉的东西。那里的每一项改动都验证过在一致性语料上输出逐字节不变。

## 绑定

C ABI 的使用方是 [`wxscan`](https://github.com/wilinz/wxscan) 里的 Flutter 包，那也是
「如何从一个平台绑定里驱动它」的参考实现。

## 许可

Apache-2.0，与它移植自的上游实现一致。
