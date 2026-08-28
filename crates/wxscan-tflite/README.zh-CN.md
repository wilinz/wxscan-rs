# wxscan-tflite

[English](README.md) · **简体中文**

对 libtensorflowlite_c（LiteRT 的 C API）的一层最小 FFI 绑定。它是
[`wxscan`](../wxscan) 的默认推理后端，也可以单独当作一个小巧的 tflite 绑定来用。

```rust
let net = wxscan_tflite::TfliteNet::from_bytes(model_bytes)?;
// 维度用模型自己的布局，这两个模型是 NHWC
let outputs = net.run(&input, &[1, height, width, 1])?;
```

解释器按输入形状缓存，形状变化时重建，因为 XNNPACK 构建的是静态图，在 delegate 生效
之后再 resize 会失败。

本 crate 说的是 tflite 的规矩：张量布局是 NHWC，形状就是模型声明的那样。这里不做任何
转换，这是有意的——适配到某个推理抽象上，属于定义那个抽象的那一层。`wxscan` 在它的
`net::Net` trait 后面做了这件事，所以算法的任何部分都不依赖本 crate。

## TFLite 库

不内置任何二进制。把 `TFLITE_LIB_DIR` 指向一个含有该库的目录，或者让最终的链接步骤
去解析这些符号——Apple 平台通常就是这么做的。名字随分发渠道而不同：C API 的桌面构建
叫 `libtensorflowlite_c`，而 Google 面向 Android 的 LiteRT 分发把同一套 API 叫作
`libLiteRt`。

[wxscan-rs](https://github.com/wilinz/wxscan-rs) 的一部分。Apache-2.0。
