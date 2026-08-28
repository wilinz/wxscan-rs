# 与上游 C++ 实现的一致性

[English](README.md) · **简体中文**

同一批图片分别喂给 OpenCV 的 `wechat_qrcode`（原始的 C++ 实现）和本仓库里的 Rust
移植，然后逐张比解码文本和四个角点坐标。

## 环境

```bash
python3 -m venv venv  && ./venv/bin/pip install opencv-contrib-python numpy segno pillow
# 参考实现：OpenCV 5.x 移除了 Caffe importer，所以带模型的一致性比对需要 4.x
python3 -m venv venv4 && ./venv4/bin/pip install opencv-contrib-python==4.10.0.84 numpy
```

## 运行

```bash
# 1. 生成语料（160 张图：多种版本和纠错等级，附带旋转、缩放、模糊、噪声、
#    透视变换和反色）
./venv/bin/python gen_corpus.py
# 2. C++ 参考结果，不带模型
./venv/bin/python run_cpp.py corpus cpp_corpus.json
# 3. Rust 结果
(cd ../../crates/wxscan && cargo run --release --quiet --example dump -- ../../tools/parity/corpus/*.png) > rust_corpus.json
# 4. 比较
./venv/bin/python compare.py cpp_corpus.json rust_corpus.json corpus/manifest.json
```

用到 CNN 模型的那条路径分三步。`gen_scene.py` 在大图里生成小尺寸的码；`run_cpp_nn.py`
产出参考输出，它要 venv4 和上游的 Caffe 模型，用
[wxscan-weights](https://github.com/wilinz/wxscan-weights) 里的
`tools/convert.py download` 下载，那个 checkout 要是不在本仓库旁边，把它的目录当第三个
参数传进去；Rust 那一侧跑
`cargo run --features tflite --example dump_nn -- <detect.tflite> <sr.tflite> scenes/*.png`。

## 当前结果

| 语料 | 一致程度 | 说明 |
|---|---|---|
| 160 张图，不带模型 | 文本 159/160 | 唯一的差异是一张反色图片，Rust 移植解出来了，C++ 实现没有；解出的文本和真值一致。角点在 152/154 上逐位相同，另外 2 张是亚像素级差异。 |
| 24 张场景图，带模型 | 文本 24/24 | 角点最大差 2.9 px，来自 SSD 框的轻微偏移，因为 TFLite 和 Caffe 的浮点累加顺序不同。 |

那些亚像素差异出自 `cv::adaptiveThreshold` 内部的高斯模糊：OpenCV 对 8U 图像走定点
实现，这个移植在 f32 上累加。cvlite 仓库里 `src/threshold.rs` 的注释写了细节。
